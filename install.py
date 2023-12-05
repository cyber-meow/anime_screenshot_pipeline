import subprocess
import os
import sys


def run(command, desc=None):
    if desc is not None:
        print(desc)

    # Join the command list into a single string if it's a list
    if isinstance(command, list):
        command = " ".join(command)

    process = subprocess.run(command, shell=True, capture_output=False)
    if process.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error code: {process.returncode}")
        sys.exit(1)


def install_package(package, command):
    if not is_installed(package):
        run(command, f"Installing {package}")


def is_installed(package):
    try:
        subprocess.run(
            f"{sys.executable} -m pip show {package}",
            shell=True,
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def prepare_environment():
    # Install PyTorch
    install_package(
        "torch",
        (
            f"{sys.executable} -m pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu121"
        ),
    )

    # Install other requirements from requirements.txt
    requirements_path = os.path.join(os.getcwd(), "requirements.txt")
    if os.path.exists(requirements_path):
        run(
            f"{sys.executable} -m pip install -r {requirements_path}",
            "Installing requirements from requirements.txt",
        )

    # Install waifuc package
    waifuc_path = os.path.join(os.getcwd(), "waifuc")
    if os.path.exists(waifuc_path):
        os.chdir(waifuc_path)
        run(
            f"{sys.executable} -m pip install .",
            "Installing waifuc package",
        )


if __name__ == "__main__":
    prepare_environment()
