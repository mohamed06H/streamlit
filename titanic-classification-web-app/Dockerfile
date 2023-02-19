# Use a lightweight Python image as base
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory to the working directory
COPY . .

# Run the train.py script to generate the model
RUN python model/train.py

# Expose the port that the Streamlit app will run on
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "app.py"]