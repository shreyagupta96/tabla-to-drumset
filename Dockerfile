# Use Miniconda as base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create the environment
RUN conda env create -f environment.yml

# Make sure environment is activated:
SHELL ["conda", "run", "-n", "ML312", "/bin/bash", "-c"]

# Set default environment
ENV PATH /opt/conda/envs/ML312/bin:$PATH

# Copy your project files (optional)
COPY . .

# Default command
CMD ["python", "api_2.py"]

