#pragma once

#include <array>
#include <math.h>

constexpr static int NumNonlinearLayer = 25;
constexpr static int WaveformSize = 256;
constexpr static int BlockSize = 64;
constexpr static int NumWfSynths = 8;

struct Nonlinear
{
	std::array<float, NumNonlinearLayer> a;//待优化系数
	std::array<float, NumNonlinearLayer> b;//待优化系数
	std::array<float, NumNonlinearLayer> c;//待优化系数
	float g, dc;//待优化系数

	void Init()
	{
		dc = 1.0;
		g = 0.0;
		for (auto& v : a)v = 0;
		for (auto& v : b)v = 0;
		for (auto& v : c)v = 0;
	}

	float NLFunc(float x)
	{
		float y = dc + g * x;
		for (int i = 0; i < NumNonlinearLayer; ++i)
			y += a[i] * tanhf(b[i] * x + c[i]);
		return y;
	}
};

struct WaveformSynthParam
{
	const float pitch = 130.81 / 44100.0;//固定线性频率系数，不优化
	Nonlinear modf;//待优化
	Nonlinear moda;//待优化
	std::array<float, WaveformSize> waveform;//待优化系数

	void Init()
	{
		modf.Init();
		moda.Init();
		for (int i = 0; i < WaveformSize; ++i)
		{
			waveform[i] = sinf(2.0 * 3.14159265358 * i / WaveformSize);
		}
	}
	float ReadWaveform(float x)
	{
		x -= floorf(x);
		x *= WaveformSize;
		int idx1 = x;
		int idx2 = (idx1 + 1) % WaveformSize;
		float frac = x - idx1;
		float a1 = waveform[idx1];
		float a2 = waveform[idx2];
		return a1 + (a2 - a1) * frac;
	}
};

struct WaveformSynthRuntime
{
	float nlt = 0.0;
	float osct = 0.0;
	float dt = 0, next_dt = 0;
	float a = 0, next_a = 0;

	float k = 0.0, kv = 1.0 / BlockSize;

	void Prepare(WaveformSynthParam& param)
	{
		nlt = 1.0;
		osct = 0.0;
		dt = param.modf.NLFunc(0.0);
		next_dt = param.modf.NLFunc(1.0);
		a = param.moda.NLFunc(0.0);
		next_a = param.moda.NLFunc(1.0);
	}
	void ProcessBlock(WaveformSynthParam& param, float* out, int numSamples)
	{
		for (int i = 0; i < numSamples; ++i)
		{
			k += kv;
			if (k >= 1.0)
			{
				k -= 1.0;
				nlt += 1.0;
				dt = next_dt;
				a = next_a;
				next_dt = param.modf.NLFunc(nlt);
				next_a = param.moda.NLFunc(nlt);
			}
			float vdt = dt + (next_dt - dt) * k;
			float va = a + (next_a - a) * k;

			osct += vdt * param.pitch;
			float v = param.ReadWaveform(osct);
			out[i] += v * va;
		}
	}
};

struct WaveformSampleCompressorParam
{
	std::array<WaveformSynthParam, NumWfSynths> wfsp;//优化这个
	void Init()
	{
		for (int i = 0; i < NumWfSynths; ++i)
			wfsp[i].Init();
	}
};
struct WaveformSampleCompressorRuntime
{
	std::array<WaveformSynthRuntime, NumWfSynths> wfrt;
	void Prepare(WaveformSampleCompressorParam& param)
	{
		for (int i = 0; i < NumWfSynths; ++i)
			wfrt[i].Prepare(param.wfsp[i]);
	}
	void ProcessBlock(WaveformSampleCompressorParam& param, float* out, int numSamples)
	{
		for (int i = 0; i < numSamples; ++i)
			out[i] = 0;
		for (int i = 0; i < NumWfSynths; ++i)
			wfrt[i].ProcessBlock(param.wfsp[i], out, numSamples);
	}
};


#include "wavfile.h"
constexpr static int EvalWindowSize = 1024;
constexpr static int EvalHopSize = EvalWindowSize / 2;
struct EvalLoss
{
	int numSamples = 0;
	float sampleRate = 0.0;
	std::vector<float> global;
	std::vector<float> sample;

	int numMagBlocks = 0;
	std::vector<std::vector<float>> globalMagBlocks;
	std::vector<std::vector<float>> sampleMagBlocks;

	static void FFT(float* re, float* im, int numSamples, int inv)
	{
		int i, j, len;
		for (i = 1, j = 0; i < numSamples; ++i) {
			int bit = numSamples >> 1;
			while (j & bit) {
				j ^= bit;
				bit >>= 1;
			}
			j ^= bit;
			if (i < j) {
				float temp;
				temp = re[i];
				re[i] = re[j];
				re[j] = temp;
				temp = im[i];
				im[i] = im[j];
				im[j] = temp;
			}
		}
		for (len = 2; len <= numSamples; len <<= 1) {
			float angle = (inv ? 2.0f : -2.0f)
				* (float)M_PI / (float)len;
			float wLenRe = cosf(angle);
			float wLenIm = sinf(angle);
			for (i = 0; i < numSamples; i += len) {
				float wRe = 1.0f;
				float wIm = 0.0f;
				for (j = 0; j < len / 2; ++j) {
					int even = i + j;
					int odd = i + j + len / 2;
					float oddRe = re[odd] * wRe - im[odd] * wIm;
					float oddIm = re[odd] * wIm + im[odd] * wRe;
					float evenRe = re[even];
					float evenIm = im[even];
					re[even] = evenRe + oddRe;
					im[even] = evenIm + oddIm;
					re[odd] = evenRe - oddRe;
					im[odd] = evenIm - oddIm;
					float nextWRe = wRe * wLenRe - wIm * wLenIm;
					float nextWIm = wRe * wLenIm + wIm * wLenRe;
					wRe = nextWRe;
					wIm = nextWIm;
				}
			}
		}
		if (inv) {
			for (i = 0; i < numSamples; ++i) {
				re[i] /= (float)numSamples;
				im[i] /= (float)numSamples;
			}
		}
	}

	void ProcessMagBlocks(std::vector<float>& global, std::vector<std::vector<float>>& globalMagBlocks)
	{
		//初始化频谱桶大小
		globalMagBlocks.resize(numMagBlocks);
		for (int i = 0; i < numMagBlocks; ++i)
		{
			globalMagBlocks[i].resize(EvalWindowSize / 2);
		}

		std::array<float, EvalWindowSize> re;
		std::array<float, EvalWindowSize> im;
		std::array<float, EvalWindowSize> window;
		//实现窗函数
		for (int i = 0; i < EvalWindowSize; ++i)
		{
			window[i] = 0.5 - 0.5 * cosf((float)i * 2.0 * M_PI / EvalWindowSize);
		}
		//根据窗长和步长填充频谱数据
		for (int n = 0; n < numMagBlocks; ++n)
		{
			int startidx = n * EvalHopSize;
			for (int i = 0; i < EvalWindowSize; ++i)
			{
				float v = (i + startidx >= numSamples) ? 0 : global[i + startidx];
				re[i] = v * window[i];
				im[i] = 0;
			}
			FFT(re.data(), im.data(), EvalWindowSize, 0);
			for (int i = 0; i < EvalWindowSize / 2; ++i)
			{
				globalMagBlocks[n][i] = sqrtf(re[i] * re[i] + im[i] * im[i]);
			}
		}
	}
	void LoadGlobalSample(std::string wavpath)
	{
		WavReader wr;
		wr.OpenWAV(wavpath);
		numSamples = wr.GetNumSamples();
		sampleRate = wr.GetSampleRate();

		std::vector<float> bufl, bufr;
		bufl.resize(numSamples);
		bufr.resize(numSamples);
		wr.ReadBlock(bufl.data(), bufr.data(), numSamples);
		for (int i = 0; i < numSamples; ++i)
		{
			global[i] = bufl[i];//先只考虑左声道
		}
		numMagBlocks = numSamples / EvalHopSize + 1;
		ProcessMagBlocks(global, globalMagBlocks);//计算wav的频谱数据
	}

	WaveformSampleCompressorRuntime runtime;
	double DoEval(WaveformSampleCompressorParam& param)
	{
		runtime.Prepare(param);
		runtime.ProcessBlock(param, sample.data(), numSamples);
		ProcessMagBlocks(sample, sampleMagBlocks);//计算waveform合成器的频谱数据
		double loss = 0;
		for (int n = 0; n < numMagBlocks; ++n)
		{
			for (int i = 0; i < EvalWindowSize / 2; ++i)
			{
				float magA = globalMagBlocks[n][i];
				float magB = sampleMagBlocks[n][i];
				float v = magA - magB;
				v = v * v;
				loss += v * 0.01;
			}
		}
		return loss;
	}
};