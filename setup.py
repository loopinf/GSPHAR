from setuptools import setup, find_packages

setup(
    name="gsphar",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "tqdm",
    ],
    author="Original Authors",
    author_email="example@example.com",
    description="Graph Signal Processing for Heterogeneous Autoregressive (GSPHAR) model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/GSPHAR",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
