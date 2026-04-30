from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kelp-detection-sentinel2",
    version="2.0.0",
    author="KelpMap Project",
    description="Binary kelp detection from Sentinel-2 imagery using lightweight INT8 CNN models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aquab1t/kelp-detection-sentinel2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "rasterio>=1.2.0",
        "scipy>=1.6.0",
        "joblib>=1.0.0",
        "tqdm>=4.60.0",
        "tflite-runtime>=2.7.0",
    ],
    include_package_data=True,
    package_data={
        "": ["*.tflite", "*.joblib"],
    },
)
