from setuptools import find_packages, setup

setup(
    name="tuw-w2023-ml-exercise3",
    version="0.0.1",
    description="Image colorization",
    author="Tibor Čuš",
    author_email="cus.tibor@outlook.com",
    license="MIT",
    install_requires=[
        "torch==2.2.0",
        "torchvision==0.17.0",
        "scikit-image==0.22.0"

    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)