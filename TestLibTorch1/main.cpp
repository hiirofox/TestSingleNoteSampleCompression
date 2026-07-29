#include "CompressorDDSP.h"

#include <exception>
#include <iostream>
#include <stdexcept>

int main()
{
    std::string inputWavPath = "D:/Projects/c++/TestLibTorch/C4.wav";
    try
    {
        compressor_ddsp::OptimizationOptions options;
        options.output_wav_path = "D:/Projects/c++/TestLibTorch/out.wav";
        options.best_parameters_path =
            "D:/Projects/c++/TestLibTorch/compressor_ddsp_best.txt";

        int program_choice = -1;
        std::cout
            << "render best param waveforms [0]\n"
            << "enter optimizer [1]\n"
            << ':';
        std::cin >> program_choice;
        if (!std::cin ||
            (program_choice != 0 &&
             program_choice != 1))
        {
            throw std::invalid_argument(
                "Program choice must be 0 or 1");
        }

        if (program_choice == 0)
        {
            compressor_ddsp::RenderBestSynthComponents(
                inputWavPath,
                options.best_parameters_path,
                "D:/Projects/c++/TestLibTorch/outputs",
                options.frequency_envelope_coefficient);
            return 0;
        }

        int basin_choice = -1;
        std::cout
            << "use last best param [0]\n"
            << "use init param [1]\n"
            << ':';
        std::cin >> basin_choice;
        if (!std::cin ||
            (basin_choice != 0 && basin_choice != 1))
        {
            throw std::invalid_argument(
                "Basin choice must be 0 or 1");
        }
        options.use_last_best_parameters =
            basin_choice == 0;

        int optimizer_choice = -1;
        std::cout
            << "use adam [0]\n"
            << "use lbfgs [1]\n"
            << ':';
        std::cin >> optimizer_choice;
        if (!std::cin ||
            (optimizer_choice != 0 &&
             optimizer_choice != 1))
        {
            throw std::invalid_argument(
                "Optimizer choice must be 0 or 1");
        }
        options.optimizer =
            optimizer_choice == 0
            ? compressor_ddsp::OptimizerKind::Adam
            : compressor_ddsp::OptimizerKind::LBFGS;

        std::cout << "iterations:";
        std::cin >> options.iterations;
        if (!std::cin || options.iterations <= 0)
        {
            throw std::invalid_argument(
                "Iterations must be positive");
        }

        compressor_ddsp::OptimizeSingleSample(
           inputWavPath,
            options);
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "Error: "
            << exception.what()
            << '\n';
        return 1;
    }

    return 0;
}
