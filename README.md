# Setup Guide

This guide will help you set up and run the project locally on your machine.
(The evidence is lcoated in /Evidencia.png)

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Installation Steps

1. Clone the repository
```bash
git clone <repository-url>
cd <repository-name>
```

2. Activate virtual environment
```bash
# Activate virtual environment
# On Linux/macOS:
source env/bin/activate
# On Windows:
.\env\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## Run programaTumor.py (Python3.11 only)
programaTumor.py Just runs over tumor/bin/active virtualenv and need python3.11
to activate do the following:
```bash
source /tumor/bun/activate
cd src
python programaTumor.py
```

## Run programa6.py
Navigate to the src directory and run any script:
```bash
cd src
python programa6.py  # or any other script you want to run
```

## Deactivating Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:
```bash
deactivate
```

## Project Structure
```
.
├── README.md
├── requirements.txt
└── src/
    └── programa6.py
```

## Troubleshooting

If you encounter any issues with the virtual environment:
1. Make sure you're in the root directory of the project
2. Ensure Python is correctly installed and added to your system PATH
3. Try removing the `env` directory and creating it again

## Contributing

Feel free to submit issues and pull requests for improvements.
