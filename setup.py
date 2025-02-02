from setuptools import setup, find_packages

setup(
    name="supply_chain",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "gymnasium>=0.28.1",
        "pymoo>=0.5.0",
        "scikit-fuzzy>=0.4.2",
    ],
)
