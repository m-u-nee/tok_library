from setuptools import setup, find_packages

setup(
    name="tok_library",  # Your project name
    version="0.1.0",  # Version following semantic versioning
    author="Mattia",  # Your name
    author_email="mnee@pleias.fr",  # Your email
    description="A collection of pleias scripts for tokenizing and decoding.",  # Short description
    long_description=open("README.md", "r").read(),  # Long description from the README.md file
    long_description_content_type="text/markdown",  # Specify that the README is in markdown format
    url="https://github.com/m-u-nee/tok_library",  # Your project's URL (GitHub or other repository)
    packages=find_packages(where='src'),  # Automatically find Python packages in the src/ directory
    package_dir={"": "src"},  # Define the source directory for your packages
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    install_requires=[  # Dependencies for running your project
        "transformers",  # Hugging Face's Transformers library for tokenization
        "datasets",  # Datasets library for loading text data
        "pyyaml",  # For YAML configuration parsing
        "pandas",  # For handling Parquet and CSV files
        "pytest"  # For running unit tests
    ],
    extras_require={  # Optional dependencies for development
        "dev": [
            "pytest",
        ],
    },
    classifiers=[  # Classify the project
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version required
    entry_points={  # Command-line scripts
        "console_scripts": [
            "preprocess_data=tokenization_pipeline.preprocess_data:main",  # Preprocessing script
            "inspect_binary=tokenization_pipeline.inspect_binary:main",  # Inspection script
        ],
    },
)