from setuptools import setup, find_packages

setup(
    name="LCIA",
    version="1.0.0",
    author="Aagam Shah",
    install_requires=[
        "pipreqs",
        "numpy<=1.26.0",
        "opencv-python==4.9.0.80",
        "ipykernel",
        "matplotlib",
        "build",
        "scikit-image",
        "pymouse",
        "pynput",
        "pytesseract",
        "scikit_learn",
        "tensorflow",
    ],
    packages=find_packages(),
    description="Master package for analysis of cross sectional images of LPBF melt tracks.",
)
