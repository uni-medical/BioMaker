from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="meddle",
    version="0.1.0",
    author="Haoyu Wang",
    author_email="small_dark@sina.com",
    description="Agentic Medical Deep Learning Engineer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uni-medical/MedDLE",
    packages=find_packages(),
    package_data={
        "meddle": [
            "../requirements.txt",
            "utils/config.yaml",
            "utils/viz_templates/*",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "meddle = meddle.run:run",
        ],
    },
)
