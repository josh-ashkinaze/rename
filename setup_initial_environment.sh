#!/bin/bash

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install litellm requests beautifulsoup4 seaborn matplotlib scikit-learn pandas openai sentence-transformers scipy statsmodels chardet ftfy python-dotenv

# Output the installed packages to a requirements.txt file
pip freeze > requirements.txt

echo "Setup complete and requirements saved."
