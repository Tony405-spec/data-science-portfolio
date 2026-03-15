from setuptools import find_packages, setup

setup(
    name="data_science_portfolio",
    version="0.1.0",
    description="Simple practice workflow to simulate MLOps habits.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
