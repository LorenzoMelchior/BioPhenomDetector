from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="BioPhenomDetector",
    version="0.1",
    packages=find_packages(where="src"),
    install_requires=read_requirements(),
    python_requires=">=3.11",
    author="Lorenzo Melchior",
    description="A python module to analyze biophysical phenomena",
    url="https://github.com/LorenzoMelchior/BioPhenomDetector.git",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
)
