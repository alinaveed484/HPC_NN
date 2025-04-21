# Project Name

This repository contains multiple versions of a project, each located in its respective folder within the `src` directory. The versions include `v1`, `v2`, `v3`, `v3.1`, `v4`, and `v5`. Additionally, there is a `data` folder for storing relevant data files.

## Repository Structure

- **src/**
  - `v1/` - Version 1 of the project
  - `v2/` - Version 2 of the project
  - `v3/` - Version 3 of the project
  - `v3.1/` - Version 3.1 of the project
  - `v4/` - Version 4 of the project
  - `v5/` - Version 5 of the project
- **data/** - Folder for storing data files

Each version folder contains a `Makefile` that allows you to build and run the project by simply running the `make` command.

## Prerequisites

- A compatible C/C++ compiler (e.g., `gcc`, `g++`) for all versions.
- CUDA toolkit for versions `v2`, `v3`, `v3.1`, and `v4`.
- OpenACC-compatible compiler (e.g., PGI/NVIDIA HPC SDK) for version `v5`.

**Note**: CUDA and OpenACC dependencies are not included in this repository. Users must install and configure these dependencies on their own system before building and running the respective versions.

## Setup and Running

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Navigate to a Version Folder**:

   ```bash
   cd src/vX  # Replace X with the desired version (e.g., v1, v2, etc.)
   ```

3. **Build and Run**:

   ```bash
   make
   ```

   The `make` command will compile and execute the project for the selected version.

4. **Data Folder**:

   - Place any required data files in the `data` folder.
   - Ensure the data files are correctly referenced in the project code if needed.

## Version-Specific Notes

- **v1**: No additional dependencies required beyond a standard C/C++ compiler.
- **v2, v3, v3.1, v4**: Require CUDA toolkit installed and properly configured.
- **v5**: Requires both CUDA and OpenACC support.

## Troubleshooting

- Ensure all dependencies (CUDA, OpenACC, compilers) are correctly installed and added to your system's PATH.
- Check the `Makefile` in each version folder for specific compilation flags or requirements.
- If you encounter issues, verify that the `data` folder contains the necessary files and that they are accessible.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements for the project.
