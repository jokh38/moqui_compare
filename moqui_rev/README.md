# moqui_rev

A C++ Monte Carlo simulation engine for Treatment Planning Systems (TPS).

## Dependencies

### Required Dependencies
- **GDCM Library**: For reading DICOM medical imaging files
  - Ubuntu/Debian: `sudo apt-get install libgdcm-dev`
  - CentOS/RHEL: `sudo yum install gdcm-devel`
  - macOS: `brew install gdcm`

### Optional Dependencies
- **DCMTK Library**: For DICOM RT Dose output support
  - Ubuntu/Debian: `sudo apt-get install libdcmtk-dev dcmtk`
  - CentOS/RHEL: `sudo yum install dcmtk-devel`
  - macOS: `brew install dcmtk`
  - If not installed, DICOM output will fall back to MHD format

- **CUDA Toolkit**: For GPU acceleration support (optional)
  - Download from NVIDIA's website or use package manager

## Building

```bash
# Configure
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# Test
ctest --test-dir build --output-on-failure
```

## Usage

### Running TPS Simulation

```bash
# Using default input file
./build/tps_env/tps_env

# Using custom input file
./build/tps_env/tps_env /path/to/input/file.in
```

### Output Formats

The system supports multiple output formats:
- `OutputFormat dcm`: DICOM RT Dose (requires DCMTK, falls back to MHD if unavailable)
- `OutputFormat mhd`: MetaImage format (recommended if DCMTK unavailable)
- `OutputFormat mha`: MetaImage format (single file)
- `OutputFormat raw`: Raw binary format

## DICOM Support

- **Reading**: Full DICOM support for CT, RTPLAN, RTSTRUCT files (via GDCM)
- **Writing**: DICOM RT Dose output (via DCMTK, optional)
- **Fallback**: Automatic fallback to MHD format when DCMTK is unavailable

## Development

```bash
# Install pre-commit hooks
pre-commit install

# Format code
clang-format -i src/*.cpp include/**/*.hpp
```

## Author

Your Name <your.email@example.com>
