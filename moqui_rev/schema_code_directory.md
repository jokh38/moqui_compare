# Code Directory Schema

## Project Overview

This document describes the code organization and build structure for the moqui_rev project, which is a C++ Monte Carlo simulation engine for Treatment Planning Systems (TPS).

## Build System

### CMake Configuration

The project uses CMake as its build system with the following configuration:
- **Minimum CMake Version**: 3.20
- **C++ Standard**: C++17
- **Build Generator**: Ninja (recommended)
- **Testing Framework**: Google Test (GTest)

### Build Commands

```bash
# Configure the build
cmake -B build -G Ninja

# Build the project
cmake --build build

# Run tests
ctest --test-dir build

# Clean build
rm -rf build && cmake -B build -G Ninja
```

## Directory Structure

```
moqui_rev/
├── CMakeLists.txt                    # Main CMake configuration
├── tps_env/                          # TPS Environment executable
│   ├── CMakeLists.txt               # CMake config for TPS environment
│   └── tps_env.cpp                  # Main entry point for TPS simulation
├── src/                             # Source files
│   ├── main.cpp                     # Original main executable (basic)
│   └── library.cpp                  # Library implementation
├── moqui/                           # Moqui simulation engine
│   ├── base/                        # Core simulation components
│       │   ├── environments/        # Environment classes
│       │   │   ├── mqi_tps_env.hpp # TPS environment definition
│       │   │   ├── mqi_phantom_env.hpp
│       │   │   └── mqi_xenvironment.hpp
│       │   └── [other base modules]
│       ├── kernel_functions/        # CUDA/kernel functions
│       └── treatment_machines/      # Treatment machine models
├── include/moqui_rev/               # Public headers
│   └── library.hpp                  # Library interface
├── tests/                           # Test files
│   ├── CMakeLists.txt              # Test CMake configuration
│   └── test_example.cpp            # Example tests
└── build/                          # Build output directory
```

## Executable Targets

### 1. TPS Environment Main Program

**Location**: `tps_env/tps_env.cpp`
**Target Name**: `tps_env` (defined in tps_env/CMakeLists.txt)
**Purpose**: Main entry point for running Monte Carlo simulations using the Treatment Planning System environment

**Description**:
- Initializes `mqi::tps_env` from an input file
- Runs the simulation with timing measurement
- Serves as the primary command-line interface for the Moqui engine
- Default input file: `./tps_env/moqui_tps.in`
- Accepts command-line argument for custom input file path

**Key Features**:
- Monte Carlo simulation execution
- Performance timing and reporting
- DICOM support (via GDCM library)
- CUDA support for GPU acceleration

### 2. Basic Library Executable

**Location**: `src/main.cpp`
**Target Name**: `moqui_rev` (defined in root CMakeLists.txt)
**Purpose**: Basic executable that links with the moqui_rev_lib

## Current Build Configuration

### TPS Environment Compilation

**Important Note**: The current codebase should only build `tps_env/tps_env.cpp`. Other compilation issues are not relevant for the current build process.

The `tps_env/tps_env.cpp` file is compiled by CMake through the following process:

1. **Root CMakeLists.txt** sets up the project configuration
2. **tps_env/CMakeLists.txt** defines the `tps_env` executable target
3. **tps_env/moqui_tps.in** serves as the configuration file for the TPS environment
4. CMake compiles `tps_env.cpp` and links it with:
   - Main moqui program libraries (from moqui/)
   - Required dependencies (GDCM, CUDA runtime if available)
   - System libraries

### Build Files Used
- **CMakeLists.txt**: Main CMake configuration file
- **tps_env/moqui_tps.in**: Configuration file used by tps_env.cpp
- **tps_env/CMakeLists.txt**: TPS environment specific CMake configuration
- **tps_env/tps_env.cpp**: Only source file that needs to be compiled

### Configuration File
The `tps_env/moqui_tps.in` file contains the TPS environment configuration that:
- Defines simulation parameters
- Specifies input data paths
- Sets up treatment machine parameters
- Configures Monte Carlo simulation settings

### Dependencies

The TPS environment depends on:
- **Moqui Base Library**: Core simulation framework (moqui/base/)
- **GDCM Library**: DICOM medical imaging support (for reading CT, RTPLAN, RTSTRUCT files)
- **DCMTK Library**: DICOM RT Dose output support (for writing RT Dose files)
- **CUDA Runtime**: GPU acceleration support (when compiled with CUDA)
- **Standard C++ Libraries**: iostream, chrono, etc.

### DICOM Support Requirements

**For Basic DICOM Operations (Required):**
- **GDCM Library**: Must be installed for reading DICOM files (CT images, RT structures, RT plans)

**For DICOM RT Dose Output (Optional):**
- **DCMTK Library**: Optional library for writing DICOM RT Dose files
  - If not installed, the system will automatically fall back to MHD format
  - Install with: `sudo apt-get install libdcmtk-dev dcmtk` (Ubuntu/Debian)
  - Other platforms: `brew install dcmtk` (macOS), `sudo yum install dcmtk-devel` (RHEL/CentOS)

## Build Targets

### Executables
- `tps_env`: Main TPS simulation executable
- `moqui_rev`: Basic library executable

### Libraries
- `moqui_rev_lib`: Core library (from src/library.cpp)

### Testing
- `test_example`: Unit tests (built when enabling testing)

## Usage

### Running TPS Simulation

```bash
# Using default input file
./build/tps_env/tps_env

# Using custom input file
./build/tps_env/tps_env /path/to/input/file.in
```

The program will:
1. Load the input configuration
2. Initialize the TPS environment
3. Run the Monte Carlo simulation
4. Report execution time in milliseconds

## Development Notes

- The main simulation logic is implemented in the `mqi::tps_env` class
- The executable in `tps_env/` serves as the primary interface
- The original `src/main.cpp` provides a basic example executable
- All core simulation components are located in `moqui/base/`
- CUDA kernels and GPU-related code are in `moqui/kernel_functions/`
