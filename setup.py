from setuptools import setup, find_packages


# List of requirements
# This could be retrieved from requirements.txt
requirements = []


# Package (minimal) configuration
setup(
    name="anime2sd",
    version="0.0.1",
    description="pipeline for anime all in one sd model",
    package_dir={"": "."},
    packages=find_packages(),  # __init__.py folders search
    install_requires=requirements
)
