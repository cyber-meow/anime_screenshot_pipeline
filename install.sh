#!/usr/bin/env bash

# Define python command
python_cmd="python"

# Pretty print
delimiter="################################################################"

printf "\n%s\n" "${delimiter}"
printf "\e[1m\e[32mPython Environment Setup Script\n\e[0m"
printf "\n%s\n" "${delimiter}"


# Update Git submodules
printf "\e[1m\e[34mUpdating Git submodules...\e[0m\n"
git submodule update --init --recursive
if [ $? -ne 0 ]; then
    printf "\e[1m\e[31mERROR: Failed to update Git submodules.\e[0m\n"
    exit 1
fi


# Check if the script is being run as root
if [[ $(id -u) -eq 0 ]]; then
    printf "\e[1m\e[31mERROR: This script should not be run as root.\e[0m\n"
    exit 1
fi


# Prompt user for environment setup choice
echo "Select environment setup:"
echo "1) venv"
echo "2) conda"
echo "3) existing environment"
read -p "Enter choice [1-3]: " env_choice


# Function to setup using venv
setup_venv() {
    if ! ${python_cmd} -m venv --help > /dev/null 2>&1; then
        printf "\e[1m\e[31mERROR: venv is not installed or not available.\e[0m\n"
        exit 1
    fi
    
    # Create venv environment
    ${python_cmd} -m venv venv
    if [ ! -d "venv" ]; then
        printf "\e[1m\e[31mERROR: Failed to create venv environment.\e[0m\n"
        exit 1
    fi
    
    # Activate venv environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m\n"
        exit 1
    fi
}


# Function to setup using conda
setup_conda() {
    if ! command -v conda > /dev/null; then
        printf "\e[1m\e[31mERROR: conda is not installed.\e[0m\n"
        exit 1
    fi

    # Path to the Conda initialization script
    CONDA_INIT_SCRIPT="$HOME/miniconda3/etc/profile.d/conda.sh"
    
    # Check if the Conda initialization script exists
    if [ ! -f "$CONDA_INIT_SCRIPT" ]; then
        printf "\e[1m\e[31mERROR: Conda initialization script not found at %s.\e[0m\n" "$CONDA_INIT_SCRIPT"
        exit 1
    fi

    # Initialize Conda for script
    source "$CONDA_INIT_SCRIPT"

    # Create conda environment
    conda create --name anime2sd python=3.10 -y
    if [ $? -ne 0 ]; then
        printf "\e[1m\e[31mERROR: Failed to create conda environment.\e[0m\n"
        exit 1
    fi

    # Activate conda environment
    conda activate anime2sd
    if [ $? -ne 0 ]; then
        printf "\e[1m\e[31mERROR: Failed to activate conda environment.\e[0m\n"
        exit 1
    fi
}


# Choose environment setup
case $env_choice in
    1)
        setup_venv
        ;;
    2)
        setup_conda
        ;;
    3)
        if [[ -z "${VIRTUAL_ENV}" && -z "${CONDA_DEFAULT_ENV}" ]]; then
            printf "\e[1m\e[31mERROR: No existing Python environment is activated.\e[0m\n"
            exit 1
        fi
        ;;
    *)
        printf "\e[1m\e[31mInvalid choice. Exiting.\e[0m\n"
        exit 1
        ;;
esac

printf "\e[1m\e[32mEnvironment setup complete.\e[0m\n"
printf "\n%s\n" "${delimiter}"

# Run the install.py script and check for errors
if ! ${python_cmd} install.py; then
    printf "\e[1m\e[31mERROR: Installation failed. Please check the error messages above.\e[0m\n"
    exit 1
fi

printf "\e[1m\e[32mInstallation complete.\e[0m\n"
printf "\n%s\n" "${delimiter}"


# Add notices at the end of the script
if [ $? -eq 0 ]; then
    printf "\e[1m\e[33mNOTICE:\e[0m If you want to run frame extraction (stage 1 of the 'screenshots' pipeline), "
    printf "please make sure ffmpeg is installed and can be run from the command line.\n"
    printf "\e[1m\e[33mOn Ubuntu, run:\e[0m sudo apt update && sudo apt install ffmpeg\n\n"

    printf "\e[1m\e[33mNOTICE:\e[0m If you want to use onnxruntime on GPU, "
    printf "please make sure that CUDA 11.8 toolkit is installed and can be found on LD_LIBRARY_PATH\n"
    printf "\e[1m\e[33mFor installation, go to:\e[0m https://developer.nvidia.com/cuda-11-8-0-download-archive\n"
fi
printf "\n%s\n" "${delimiter}"


# Provide instructions based on the chosen environment setup
case $env_choice in
    1)
        printf "\e[1m\e[32mTo activate the venv environment, run:\e[0m source venv/bin/activate\n"
        printf "\n%s\n" "${delimiter}"
        ;;
    2)
        printf "\e[1m\e[32mTo activate the conda environment, run:\e[0m conda activate anime2sd\n"
        printf "\n%s\n" "${delimiter}"
        ;;
    3)
        # No additional instructions needed for existing environment
        ;;
esac
