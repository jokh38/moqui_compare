### Development Plan: Other Key Features & Improvements

*   **Features**: CMake build system, Log file-based beam source, Modular treatment machines, and general code quality.

*   **Status**: Mostly complete. These features represent significant software engineering improvements that are already in place.

*   **Development Plan**: The plan for these features is one of continuous improvement and extension, building upon the solid foundation that has been established.

    *   **1. Build System (CMake)**
        *   **Status**: Complete and functional.
        *   **Plan**: 
            *   **Maintenance**: Diligently update `CMakeLists.txt` as new source files, headers, or dependencies are added to the project.
            *   **CI/CD Integration**: Integrate the CMake build process with a Continuous Integration (CI) system (like GitHub Actions). This would automatically build and test the project on every commit, catching integration errors early.

    *   **2. Log File-Based Beam Source**
        *   **Status**: Implemented for CSV-based log files.
        *   **Plan**:
            *   **Extend Support**: Abstract the log file reading logic into a more general interface. Create new classes that implement this interface for different log file formats from other vendors (e.g., Varian, Elekta). This would greatly enhance the versatility of `moqui` for research and clinical validation.
            *   **Error Handling**: Improve the robustness of the CSV parser in `read_logfile_dir()` to handle potential errors like missing columns, incorrect data types, or empty files, providing clearer error messages to the user.

    *   **3. Modular Treatment Machines**
        *   **Status**: Architecture is in place, with examples like `mqi_treatment_machine_smc_gtr2.hpp`.
        *   **Plan**:
            *   **Expand Library**: As new research or clinical needs arise, add new treatment machine models by creating new header files in the `moqui/treatment_machines/` directory. Each new file would define a class for a specific machine, encapsulating its unique geometry (SAD, range shifter specs) and material properties.
            *   **Documentation**: For each new machine model added, create a corresponding markdown file documenting its specifications and any unique parameters.

    *   **4. Code Quality and Documentation**
        *   **Status**: Significantly improved in `moqui_rev`.
        *   **Plan**:
            *   **Enforce Standards**: Continue the excellent practice of using Doxygen-style comments (`/// \brief`). Consider adding a static analysis tool (like `clang-tidy`, for which a configuration file already exists) to the CI pipeline to automatically enforce coding standards.
            *   **Unit Testing**: While `moqui_rev` has a better structure, it lacks a formal unit testing framework. Integrating a framework like Google Test would be a major step forward. New features (like the Voxelized Aperture or new machine models) should be accompanied by unit tests to ensure their correctness and prevent future regressions.
