@echo off
setlocal

set "python_cmd=python"
set "delimiter=****************************************************************"

echo %delimiter%
echo Python Environment Setup Script
echo %delimiter%

REM Check if the script is being run as administrator
net session >nul 2>&1
if %errorlevel% == 0 (
    echo ERROR: This script should not be run as administrator.
    exit /b
)

REM Update Git submodules
echo Updating Git submodules...
git submodule update --init --recursive
if not %errorlevel% == 0 (
    echo ERROR: Failed to update Git submodules.
    exit /b
)

echo.
echo Select environment setup:
echo 1) venv
echo 2) conda
echo 3) existing environment
set /p env_choice="Enter choice [1-3]: "

REM Function to setup using venv
:setup_venv
%python_cmd% -m venv --help >nul 2>&1
if not %errorlevel% == 0 (
    echo ERROR: venv is not installed or not available.
    exit /b
)
%python_cmd% -m venv venv
if not exist "venv" (
    echo ERROR: Failed to create venv environment.
    exit /b
)
call venv\Scripts\activate.bat
goto end

REM Function to setup using conda
:setup_conda
where conda >nul 2>&1
if not %errorlevel% == 0 (
    echo ERROR: conda is not installed.
    exit /b
)

REM Conda environment setup is tricky in batch and might not work as expected
REM You might need to adjust this part based on your Conda setup
call conda create --name anime2sd python=3.10 -y
if not %errorlevel% == 0 (
    echo ERROR: Failed to create conda environment.
    exit /b
)
call conda activate anime2sd
if not %errorlevel% == 0 (
    echo ERROR: Failed to activate conda environment.
    exit /b
)
goto end

REM Choose environment setup
if "%env_choice%"=="1" goto setup_venv
if "%env_choice%"=="2" goto setup_conda
if "%env_choice%"=="3" (
    REM Check for existing environment
    if defined VIRTUAL_ENV goto end
    if defined CONDA_DEFAULT_ENV goto end
    echo ERROR: No existing Python environment is activated.
    exit /b
) else (
    echo Invalid choice. Exiting.
    exit /b
)

:end
echo %delimiter%
echo Environment setup complete.
echo %delimiter%

REM Run the install.py script and check for errors
%python_cmd% install.py
if not %errorlevel% == 0 (
    echo ERROR: Installation failed. Please check the error messages above.
    exit /b
)

REM Add notices at the end of the script
echo %delimiter%
echo NOTICE: If you want to run frame extraction (stage 1 of the 'screenshots' pipeline), please make sure FFmpeg is installed and can be run from the command line.
echo On Windows, you can install FFmpeg using Chocolatey: https://chocolatey.org/install
echo Then, run 'choco install ffmpeg'
echo Or download it directly from https://ffmpeg.org/download.html
echo.
echo NOTICE: If you want to use onnxruntime on GPU, please make sure that CUDA 11.8 toolkit is installed and can be found on PATH
echo For installation, go to: https://developer.nvidia.com/cuda-11-8-0-download-archive
echo %delimiter% 

REM Provide instructions based on the chosen environment setup
if "%env_choice%"=="1" (
    echo To activate the venv environment, run: call venv\Scripts\activate.bat
) else if "%env_choice%"=="2" (
    echo To activate the conda environment, run: conda activate anime2sd
)
echo %delimiter%

endlocal