from setuptools import setup, find_packages

setup(
    name="tlustynn",
    version="0.1.0",
    author="",
    author_email="",
    description="Physics-Informed Neural Network (PINN) for TLUSTY stellar atmosphere prediction",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/tlusty-nn",
    packages=find_packages(),
    package_data={
        "tlustynn": ["checkpoints/*"],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
