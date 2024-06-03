from setuptools import setup, find_packages

setup(
    name="LCIA",
    version="0.1.0",
    author="Aagam Shah",
    install_requires=["opencv-python", "ipykernel"],
    packages=find_packages(),
    description="Master package for analysis of cross sectional images of LPBF melt tracks.",
)
