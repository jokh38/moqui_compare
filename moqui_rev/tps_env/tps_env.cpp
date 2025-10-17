/*!
 * @file tps_env.cpp
 * @brief Main entry point for running a Monte Carlo simulation using the Treatment Planning System (TPS) environment.
 * @details This executable initializes the `mqi::tps_env` from an input file, runs the simulation, and reports the total execution time. It serves as a command-line interface for the Moqui simulation engine.
*/

#include <cassert>
#include <chrono>
#include <iostream>

#include <moqui/base/environments/mqi_tps_env.hpp>
#include <moqui/base/mqi_cli.hpp>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
__device__ cudaTextureObject_t phantom_texture_object = 0;
#endif

#include "gdcmAttribute.h"
#include "gdcmDataElement.h"
#include "gdcmDataSet.h"
#include "gdcmDict.h"
#include "gdcmDicts.h"
#include "gdcmGlobal.h"
#include "gdcmIPPSorter.h"
#include "gdcmImage.h"
#include "gdcmReader.h"
#include "gdcmScanner.h"
#include "gdcmSorter.h"
#include "gdcmStringFilter.h"
#include "gdcmTag.h"
#include "gdcmTesting.h"

typedef float phsp_t;

/*!
 * @brief The main function for the TPS environment executable.
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line arguments. The first argument is optionally the path to the input file.
 * @return 0 on successful execution, non-zero otherwise.
*/
int
main(int argc, char* argv[]) {
    ///< construct a treatment planning system environment
    auto        start = std::chrono::high_resolution_clock::now();
    std::string input_file;

    if (argc > 1) {
        input_file = argv[1];
    } else {
        input_file = "./tps_env/moqui_tps.in";
    }
    mqi::tps_env<phsp_t> myenv(input_file);
    myenv.initialize_and_run();

    auto                                      stop     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "Time taken by MC engine: " << duration.count() << " milli-seconds\n";
    return 0;
}
