#pragma once

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "WaveformSampleCompressor.h"
#include "wavfile.h"

namespace compressor_ddsp
{

	// The constants intentionally mirror WaveformSampleCompressor.h.  This class
	// does not modify or call the original DSP; it is a differentiable equivalent.
	constexpr int64_t kNumSynths = NumWfSynths;
	constexpr int64_t kWaveformSize = WaveformSize;
	constexpr int64_t kNonlinearLayers = NumNonlinearLayer;
	constexpr int64_t kBlockSize = BlockSize;
	constexpr int64_t kEvalWindowSize = EvalWindowSize;
	constexpr int64_t kEvalHopSize = EvalHopSize;
	constexpr float kBasePitchHz = 130.81F * 1.0F;//130.81=C4
	constexpr float kExpectedSampleRate = 48000.0F;
	constexpr double kDefaultFrequencyEnvelopeCoefficient = 0.001;

	struct LoadedMonoSample
	{
		torch::Tensor samples;
		float sample_rate = 0.0F;
	};

	inline LoadedMonoSample LoadLeftChannel(const std::string& wav_path)
	{
		WavReader reader;
		if (!reader.OpenWAV(wav_path))
			throw std::runtime_error("Failed to open WAV file: " + wav_path);

		const uint64_t sample_count_u64 = reader.GetNumSamples();
		if (sample_count_u64 == 0 ||
			sample_count_u64 >
			static_cast<uint64_t>(std::numeric_limits<int>::max()))
		{
			throw std::runtime_error("Unsupported WAV length: " + wav_path);
		}

		const int sample_count = static_cast<int>(sample_count_u64);
		std::vector<float> left(static_cast<size_t>(sample_count));
		std::vector<float> right(static_cast<size_t>(sample_count));
		const int samples_read =
			reader.ReadBlock(left.data(), right.data(), sample_count);

		if (samples_read != sample_count)
			throw std::runtime_error("Could not read the complete WAV file");

		LoadedMonoSample result;
		result.sample_rate = static_cast<float>(reader.GetSampleRate());
		result.samples =
			torch::from_blob(
				left.data(),
				{ static_cast<int64_t>(sample_count) },
				torch::TensorOptions().dtype(torch::kFloat32))
			.clone();
		return result;
	}

	inline void WriteMonoSample(
		const std::string& wav_path,
		const torch::Tensor& audio,
		float sample_rate)
	{
		torch::NoGradGuard no_grad;
		const torch::Tensor cpu_audio =
			audio.detach()
			.to(torch::Device(torch::kCPU), torch::kFloat32)
			.contiguous();

		WavWriter writer;
		writer.CreateWAV(wav_path, sample_rate);
		const int sample_count =
			static_cast<int>(cpu_audio.numel());
		const float* samples = cpu_audio.data_ptr<float>();
		writer.WriteBlock(samples, samples, sample_count);
		writer.Close();
	}

	class CompressorDDSP : public torch::nn::Module
	{
	public:
		explicit CompressorDDSP(
			double frequency_envelope_coefficient =
				kDefaultFrequencyEnvelopeCoefficient)
			: frequency_envelope_coefficient_(
				frequency_envelope_coefficient)
		{
			if (!std::isfinite(frequency_envelope_coefficient_) ||
				frequency_envelope_coefficient_ < 0.0)
			{
				throw std::invalid_argument(
					"frequency_envelope_coefficient must be finite and non-negative");
			}

			// Deterministic small asymmetry is necessary because the ten original
			// synths otherwise have exactly equal values and equal gradients.
			torch::manual_seed(0);

			const auto options =
				torch::TensorOptions().dtype(torch::kFloat32);

			const torch::Tensor phase =
				torch::arange(kWaveformSize, options) *
				(2.0 * std::acos(-1.0) /
					static_cast<double>(kWaveformSize));
			const torch::Tensor initial_waveform =
				torch::sin(phase)
				.unsqueeze(0)
				.repeat({ kNumSynths, 1 });

			waveforms_ = register_parameter(
				"waveforms",
				initial_waveform.clone());

			modf_a_ = RegisterMatrix("modf_a", 0.0F);
			modf_b_ = register_parameter(
				"modf_b",
				1.0e-3F *
				torch::randn(
					{ kNumSynths, kNonlinearLayers },
					options));
			modf_c_ = register_parameter(
				"modf_c",
				1.0e-2F *
				torch::randn(
					{ kNumSynths, kNonlinearLayers },
					options));
			modf_g_ = RegisterVector("modf_g", 0.0F);
			// Frequency DC is persisted for compatibility with existing text
			// parameter files, but it is fixed at 1 and is not trainable.
			modf_dc_ = register_parameter(
				"modf_dc",
				torch::ones({ kNumSynths }, options),
				false);

			moda_a_ = RegisterMatrix("moda_a", 0.0F);
			moda_b_ = register_parameter(
				"moda_b",
				1.0e-3F *
				torch::randn(
					{ kNumSynths, kNonlinearLayers },
					options));
			moda_c_ = register_parameter(
				"moda_c",
				1.0e-2F *
				torch::randn(
					{ kNumSynths, kNonlinearLayers },
					options));
			moda_g_ = RegisterVector("moda_g", 0.0F);
			moda_dc_ = RegisterVector("moda_dc", 1.0F);
		}

		torch::Tensor forward(
			int64_t sample_count,
			float sample_rate)
		{
			return torch::sum(
				forward_components(
					sample_count,
					sample_rate),
				0);
		}

		torch::Tensor forward_components(
			int64_t sample_count,
			float sample_rate)
		{
			if (sample_count <= 0)
				throw std::invalid_argument(
					"sample_count must be positive");
			if (sample_rate <= 0.0F)
				throw std::invalid_argument(
					"sample_rate must be positive");

			const auto options = waveforms_.options();

			// The original runtime evaluates Nonlinear only at block boundaries
			// and linearly interpolates between adjacent boundary values.
			const int64_t node_count =
				sample_count / kBlockSize + 2;
			const torch::Tensor node_time =
				torch::arange(node_count, options);

			const torch::Tensor frequency_a =
				(0.1 * frequency_envelope_coefficient_) *
				torch::tanh(modf_a_);
			const torch::Tensor frequency_g =
				(0.001 * frequency_envelope_coefficient_) *
				torch::tanh(modf_g_);
			const torch::Tensor frequency_nodes =
				EvaluateNonlinear(
					node_time,
					frequency_a,
					modf_b_,
					modf_c_,
					frequency_g,
					modf_dc_);
			const torch::Tensor amplitude_nodes =
				EvaluateNonlinear(
					node_time,
					moda_a_,
					moda_b_,
					moda_c_,
					moda_g_,
					moda_dc_);

			// Original sample zero uses k = 1 / BlockSize, so the interpolation
			// coordinate starts at one rather than zero.
			const torch::Tensor sample_position =
				torch::arange(
					1,
					sample_count + 1,
					options) /
				static_cast<double>(kBlockSize);
			const torch::Tensor left_indices =
				torch::floor(sample_position)
				.to(torch::kLong);
			const torch::Tensor interpolation =
				sample_position -
				left_indices.to(
					sample_position.scalar_type());

			const torch::Tensor frequency =
				InterpolateNodes(
					frequency_nodes,
					left_indices,
					interpolation);
			const torch::Tensor amplitude =
				InterpolateNodes(
					amplitude_nodes,
					left_indices,
					interpolation);

			// Match the original fixed oscillator phase increment, using the
			// selected kBasePitchHz constant and the fixed 44100 Hz sample rate.
			const double phase_increment_scale =
				static_cast<double>(kBasePitchHz) /
				static_cast<double>(kExpectedSampleRate);
			const torch::Tensor oscillator_phase =
				torch::cumsum(
					frequency * phase_increment_scale,
					1);

			const torch::Tensor oscillator =
				ReadWaveforms(oscillator_phase);
			return oscillator * amplitude;
		}

	private:
		torch::Tensor RegisterMatrix(
			const std::string& name,
			float value)
		{
			return register_parameter(
				name,
				torch::full(
					{ kNumSynths, kNonlinearLayers },
					value,
					torch::TensorOptions().dtype(
						torch::kFloat32)));
		}

		torch::Tensor RegisterVector(
			const std::string& name,
			float value)
		{
			return register_parameter(
				name,
				torch::full(
					{ kNumSynths },
					value,
					torch::TensorOptions().dtype(
						torch::kFloat32)));
		}

		static torch::Tensor EvaluateNonlinear(
			const torch::Tensor& x,
			const torch::Tensor& a,
			const torch::Tensor& b,
			const torch::Tensor& c,
			const torch::Tensor& g,
			const torch::Tensor& dc)
		{
			const torch::Tensor expanded_x =
				x.view({ 1, -1, 1 });
			const torch::Tensor nonlinear_terms =
				a.unsqueeze(1) *
				torch::tanh(
					b.unsqueeze(1) * expanded_x +
					c.unsqueeze(1));

			return dc.unsqueeze(1) +
				g.unsqueeze(1) * x.unsqueeze(0) +
				torch::sum(nonlinear_terms, 2);
		}

		static torch::Tensor InterpolateNodes(
			const torch::Tensor& nodes,
			const torch::Tensor& left_indices,
			const torch::Tensor& interpolation)
		{
			const torch::Tensor left =
				nodes.index_select(1, left_indices);
			const torch::Tensor right =
				nodes.index_select(1, left_indices + 1);
			return left +
				(right - left) *
				interpolation.unsqueeze(0);
		}

		torch::Tensor ReadWaveforms(
			const torch::Tensor& phase) const
		{
			// This reproduces ReadWaveform: wrap the phase, convert it to a
			// 256-entry table coordinate, and linearly interpolate adjacent
			// samples.  Index selection is discrete, while the fractional path
			// preserves the piecewise gradient with respect to phase.
			const torch::Tensor wrapped =
				phase - torch::floor(phase);
			const torch::Tensor table_position =
				wrapped *
				static_cast<double>(kWaveformSize);
			const torch::Tensor raw_index =
				torch::floor(table_position);
			const torch::Tensor index0 =
				torch::remainder(
					raw_index.to(torch::kLong),
					kWaveformSize);
			const torch::Tensor index1 =
				torch::remainder(
					index0 + 1,
					kWaveformSize);
			const torch::Tensor fraction =
				table_position - raw_index;

			const torch::Tensor value0 =
				waveforms_.gather(1, index0);
			const torch::Tensor value1 =
				waveforms_.gather(1, index1);
			return value0 +
				(value1 - value0) * fraction;
		}

	private:
		double frequency_envelope_coefficient_ =
			kDefaultFrequencyEnvelopeCoefficient;
		torch::Tensor waveforms_;

		torch::Tensor modf_a_;
		torch::Tensor modf_b_;
		torch::Tensor modf_c_;
		torch::Tensor modf_g_;
		torch::Tensor modf_dc_;

		torch::Tensor moda_a_;
		torch::Tensor moda_b_;
		torch::Tensor moda_c_;
		torch::Tensor moda_g_;
		torch::Tensor moda_dc_;
	};

	class MagnitudeSpectrumLoss
	{
	public:
		explicit MagnitudeSpectrumLoss(
			const torch::Tensor& target)
			: target_magnitude_(
				CalculateMagnitudeBlocks(
					target.detach()))
		{
		}

		torch::Tensor operator()(
			const torch::Tensor& synthesized) const
		{
			const torch::Tensor synthesized_magnitude =
				CalculateMagnitudeBlocks(synthesized);
			const torch::Tensor difference =
				target_magnitude_ - synthesized_magnitude;

			// Match EvalLoss::DoEval: sum((magA - magB)^2 * 0.01).
			return torch::sum(
				torch::square(difference)) *
				0.01;
		}

	private:
		static torch::Tensor CalculateMagnitudeBlocks(
			const torch::Tensor& audio)
		{
			if (audio.dim() != 1)
				throw std::invalid_argument(
					"MagnitudeSpectrumLoss expects mono 1-D audio");

			const int64_t sample_count = audio.size(0);
			const int64_t frame_count =
				sample_count / kEvalHopSize + 1;
			const int64_t required_samples =
				(frame_count - 1) * kEvalHopSize +
				kEvalWindowSize;
			const int64_t right_padding =
				required_samples - sample_count;

			torch::Tensor padded = audio;
			if (right_padding > 0)
			{
				padded = torch::cat(
					{
						audio,
						torch::zeros(
							{right_padding},
							audio.options())
					},
					0);
			}

			const torch::Tensor frames =
				padded.unfold(
					0,
					kEvalWindowSize,
					kEvalHopSize);

			// periodic Hann window:
			// 0.5 - 0.5*cos(2*pi*n/window_size)
			const torch::Tensor window_position =
				torch::arange(
					kEvalWindowSize,
					audio.options());
			const torch::Tensor window =
				0.5 -
				0.5 *
				torch::cos(
					window_position *
					(2.0 * std::acos(-1.0) /
						static_cast<double>(
							kEvalWindowSize)));

			const torch::Tensor spectrum =
				torch::fft::rfft(frames * window);

			// rfft includes the Nyquist bin, while the original evaluator keeps
			// exactly EvalWindowSize / 2 bins and therefore excludes Nyquist.
			return torch::abs(spectrum)
				.narrow(
					1,
					0,
					kEvalWindowSize / 2);
		}

	private:
		torch::Tensor target_magnitude_;
	};

	enum class OptimizerKind
	{
		Adam,
		LBFGS
	};

	struct OptimizationOptions
	{
		int iterations = 500;
		OptimizerKind optimizer = OptimizerKind::Adam;
		bool use_last_best_parameters = false;

		double frequency_envelope_coefficient =
			kDefaultFrequencyEnvelopeCoefficient;
		double synth_learning_rate = 2.0e-2;
		double lbfgs_learning_rate = 0.1;
		double maximum_gradient_norm = 10.0;

		bool automatic_synth_learning_rate = true;
		int synth_lr_warmup_iterations = 100;
		double synth_lr_reduction_factor = 0.9;
		int synth_lr_patience = 100;
		double synth_lr_threshold = 1.0e-6;
		int synth_lr_cooldown = 100;
		double minimum_synth_learning_rate = 1.0e-6;

		std::string output_wav_path = "sample.wav";
		std::string best_parameters_path =
			"compressor_ddsp_best.txt";
	};

	inline void SaveBestParametersText(
		const CompressorDDSP& model,
		double best_loss,
		const std::string& path)
	{
		std::ofstream output(
			path,
			std::ios::out | std::ios::trunc);
		if (!output.is_open())
			throw std::runtime_error(
				"Failed to create parameter file: " + path);

		const auto parameters =
			model.named_parameters(true);

		output << "COMPRESSOR_DDSP_PARAMETERS_V1\n";
		output << "best_loss "
			<< std::setprecision(
				std::numeric_limits<double>::max_digits10)
			<< best_loss
			<< '\n';
		output << "tensor_count "
			<< parameters.size()
			<< '\n';

		output << std::setprecision(
			std::numeric_limits<float>::max_digits10);

		for (const auto& item : parameters)
		{
			const torch::Tensor tensor =
				item.value()
				.detach()
				.to(
					torch::Device(torch::kCPU),
					torch::kFloat32)
				.contiguous();

			output << "tensor "
				<< item.key()
				<< ' '
				<< tensor.dim();
			for (const int64_t size : tensor.sizes())
				output << ' ' << size;
			output << ' ' << tensor.numel() << '\n';

			const float* values =
				tensor.data_ptr<float>();
			for (int64_t i = 0; i < tensor.numel(); ++i)
			{
				output << values[i];
				if ((i + 1) % 8 == 0 ||
					i + 1 == tensor.numel())
				{
					output << '\n';
				}
				else
				{
					output << ' ';
				}
			}
		}

		if (!output)
			throw std::runtime_error(
				"Failed while writing parameter file: " + path);
	}

	inline double LoadBestParametersText(
		CompressorDDSP& model,
		const std::string& path)
	{
		std::ifstream input(path);
		if (!input.is_open())
			throw std::runtime_error(
				"Best parameter file does not exist: " + path);

		std::string token;
		input >> token;
		if (token != "COMPRESSOR_DDSP_PARAMETERS_V1")
			throw std::runtime_error(
				"Unsupported parameter file format: " + path);

		double best_loss = 0.0;
		input >> token >> best_loss;
		if (token != "best_loss" ||
			!std::isfinite(best_loss))
		{
			throw std::runtime_error(
				"Invalid best_loss in parameter file: " + path);
		}

		size_t tensor_count = 0;
		input >> token >> tensor_count;
		if (token != "tensor_count")
			throw std::runtime_error(
				"Missing tensor_count in parameter file: " + path);

		auto parameters =
			model.named_parameters(true);
		if (tensor_count != parameters.size())
			throw std::runtime_error(
				"Parameter count does not match the DDSP model");

		torch::NoGradGuard no_grad;
		for (size_t parameter_index = 0;
			parameter_index < tensor_count;
			++parameter_index)
		{
			std::string name;
			int64_t dimension_count = 0;
			input >> token >> name >> dimension_count;
			if (token != "tensor" ||
				dimension_count < 0)
			{
				throw std::runtime_error(
					"Invalid tensor header in parameter file");
			}

			const auto& item = parameters[parameter_index];
			torch::Tensor parameter = item.value();
			if (name != item.key() ||
				dimension_count != parameter.dim())
			{
				throw std::runtime_error(
					"Parameter name or dimension mismatch for: " +
					name);
			}

			std::vector<int64_t> sizes(
				static_cast<size_t>(dimension_count));
			for (int64_t dimension = 0;
				dimension < dimension_count;
				++dimension)
			{
				input >> sizes[static_cast<size_t>(dimension)];
				if (sizes[static_cast<size_t>(dimension)] !=
					parameter.size(dimension))
				{
					throw std::runtime_error(
						"Parameter shape mismatch for: " + name);
				}
			}

			int64_t value_count = 0;
			input >> value_count;
			if (value_count != parameter.numel())
				throw std::runtime_error(
					"Parameter value count mismatch for: " + name);

			std::vector<float> values(
				static_cast<size_t>(value_count));
			for (float& value : values)
			{
				input >> value;
				if (!input || !std::isfinite(value))
					throw std::runtime_error(
						"Invalid value for parameter: " + name);
			}

			const torch::Tensor loaded =
				torch::from_blob(
					values.data(),
					sizes,
					torch::TensorOptions().dtype(
						torch::kFloat32))
				.clone();
			if (name == "modf_dc")
			{
				parameter.fill_(1.0);
			}
			else
			{
				parameter.copy_(
					loaded.to(parameter.device()));
			}
		}

		return best_loss;
	}

	inline void RenderBestSynthComponents(
		const std::string& target_wav_path,
		const std::string& best_parameters_path,
		const std::string& output_directory,
		double frequency_envelope_coefficient =
			kDefaultFrequencyEnvelopeCoefficient)
	{
		const LoadedMonoSample target =
			LoadLeftChannel(target_wav_path);

		CompressorDDSP model(
			frequency_envelope_coefficient);
		const double best_loss =
			LoadBestParametersText(
				model,
				best_parameters_path);

		std::filesystem::create_directories(
			output_directory);

		torch::NoGradGuard no_grad;
		const torch::Tensor components =
			model.forward_components(
				target.samples.numel(),
				target.sample_rate);

		if (components.dim() != 2 ||
			components.size(0) != kNumSynths)
		{
			throw std::runtime_error(
				"Unexpected DDSP component tensor shape");
		}

		for (int64_t synth_index = 0;
			synth_index < kNumSynths;
			++synth_index)
		{
			const std::filesystem::path output_path =
				std::filesystem::path(output_directory) /
				("wf" +
					std::to_string(synth_index) +
					".wav");
			WriteMonoSample(
				output_path.string(),
				components.select(0, synth_index),
				target.sample_rate);
			std::cout
				<< "Wrote "
				<< output_path.string()
				<< '\n';
		}

		std::cout
			<< "Rendered "
			<< kNumSynths
			<< " best-parameter synth components"
			<< " | saved best loss = "
			<< best_loss
			<< '\n';
	}

	inline void ObserveOptimizationCandidate(
		const CompressorDDSP& model,
		const torch::Tensor& synthesized,
		double loss_value,
		int iteration,
		float sample_rate,
		const OptimizationOptions& options,
		double& best_loss)
	{
		if (iteration % 10 == 0)
		{
			std::cout
				<< "Iteration "
				<< iteration
				<< " | loss = "
				<< loss_value
				<< '\n';
		}

		if (loss_value >= best_loss)
			return;

		best_loss = loss_value;
		WriteMonoSample(
			options.output_wav_path,
			synthesized,
			sample_rate);
		SaveBestParametersText(
			model,
			best_loss,
			options.best_parameters_path);

		std::cout
			<< "Iteration "
			<< iteration
			<< " | new best loss = "
			<< best_loss
			<< " | wrote "
			<< options.output_wav_path
			<< " and "
			<< options.best_parameters_path
			<< '\n';
	}

	inline double OptimizeSingleSample(
		const std::string& target_wav_path,
		const OptimizationOptions& options = {})
	{
		if (options.iterations <= 0)
			throw std::invalid_argument(
				"iterations must be positive");
		if (options.automatic_synth_learning_rate &&
			(options.synth_lr_warmup_iterations < 0 ||
				options.synth_lr_reduction_factor <= 0.0 ||
				options.synth_lr_reduction_factor >= 1.0 ||
				options.synth_lr_patience < 0 ||
				options.synth_lr_threshold < 0.0 ||
				options.synth_lr_cooldown < 0 ||
				options.minimum_synth_learning_rate < 0.0))
		{
			throw std::invalid_argument(
				"Invalid automatic synth learning-rate settings");
		}

		LoadedMonoSample target =
			LoadLeftChannel(target_wav_path);

		if (std::abs(
			target.sample_rate -
			kExpectedSampleRate) >
			0.5F)
		{
			std::cerr
				<< "Warning: the original DSP uses 44100 Hz, but the "
				<< "target WAV is "
				<< target.sample_rate
				<< " Hz. The DDSP keeps the original fixed 44100 Hz "
				<< "phase coefficient.\n";
		}

		CompressorDDSP model(
			options.frequency_envelope_coefficient);
		double best_loss =
			std::numeric_limits<double>::infinity();
		if (options.use_last_best_parameters)
		{
			best_loss = LoadBestParametersText(
				model,
				options.best_parameters_path);
			std::cout
				<< "Loaded last best parameters from "
				<< options.best_parameters_path
				<< " | saved best loss = "
				<< best_loss
				<< '\n';
		}

		MagnitudeSpectrumLoss loss_function(target.samples);

		std::cout << std::scientific
			<< std::setprecision(9);

		if (options.optimizer == OptimizerKind::Adam)
		{
			torch::optim::Adam synth_optimizer(
				model.parameters(),
				torch::optim::AdamOptions(
					options.synth_learning_rate));
			torch::optim::ReduceLROnPlateauScheduler
				synth_scheduler(
					synth_optimizer,
					torch::optim::ReduceLROnPlateauScheduler::min,
					static_cast<float>(
						options.synth_lr_reduction_factor),
					options.synth_lr_patience,
					options.synth_lr_threshold,
					torch::optim::ReduceLROnPlateauScheduler::rel,
					options.synth_lr_cooldown,
					{
						static_cast<float>(
							options.minimum_synth_learning_rate)
					},
					1.0e-12,
					false);
			double current_synth_learning_rate =
				options.synth_learning_rate;

			for (int iteration = 0;
				iteration < options.iterations;
				++iteration)
			{
				synth_optimizer.zero_grad();

				const torch::Tensor synthesized =
					model.forward(
						target.samples.numel(),
						target.sample_rate);
				const torch::Tensor loss =
					loss_function(synthesized);
				const double loss_value =
					loss.item<double>();
				if (!std::isfinite(loss_value))
				{
					throw std::runtime_error(
						"Loss became NaN or Inf at iteration " +
						std::to_string(iteration));
				}

				ObserveOptimizationCandidate(
					model,
					synthesized,
					loss_value,
					iteration,
					target.sample_rate,
					options,
					best_loss);

				loss.backward();
				torch::nn::utils::clip_grad_norm_(
					model.parameters(),
					options.maximum_gradient_norm);
				synth_optimizer.step();

				if (options.automatic_synth_learning_rate &&
					iteration >=
						options.synth_lr_warmup_iterations)
				{
					synth_scheduler.step(
						static_cast<float>(loss_value));

					const double updated_learning_rate =
						static_cast<torch::optim::AdamOptions&>(
							synth_optimizer
								.param_groups()
								.front()
								.options())
						.lr();
					if (updated_learning_rate !=
						current_synth_learning_rate)
					{
						std::cout
							<< "Iteration "
							<< iteration
							<< " | synth learning rate: "
							<< current_synth_learning_rate
							<< " -> "
							<< updated_learning_rate
							<< '\n';
						current_synth_learning_rate =
							updated_learning_rate;
					}
				}
			}
		}
		else
		{
			auto lbfgs_options =
				torch::optim::LBFGSOptions(
					options.lbfgs_learning_rate);
			lbfgs_options.max_iter(10);
			lbfgs_options.max_eval(20);
			lbfgs_options.history_size(200);
			lbfgs_options.line_search_fn("strong_wolfe");

			torch::optim::LBFGS optimizer(
				model.parameters(),
				lbfgs_options);

			for (int iteration = 0;
				iteration < options.iterations;
				++iteration)
			{
				auto closure = [&]() -> torch::Tensor
					{
						optimizer.zero_grad();
						const torch::Tensor synthesized =
							model.forward(
								target.samples.numel(),
								target.sample_rate);
						const torch::Tensor loss =
							loss_function(synthesized);
						const double loss_value =
							loss.item<double>();
						if (!std::isfinite(loss_value))
						{
							throw std::runtime_error(
								"Loss became NaN or Inf at iteration " +
								std::to_string(iteration));
						}

						ObserveOptimizationCandidate(
							model,
							synthesized,
							loss_value,
							iteration,
							target.sample_rate,
							options,
							best_loss);

						loss.backward();
						torch::nn::utils::clip_grad_norm_(
							model.parameters(),
							options.maximum_gradient_norm);
						return loss;
					};

				optimizer.step(closure);
			}
		}

		std::cout
			<< "Optimization finished. Best loss = "
			<< best_loss
			<< '\n';
		return best_loss;
	}

} // namespace compressor_ddsp
