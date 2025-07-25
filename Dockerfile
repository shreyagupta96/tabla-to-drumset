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

# Copy all project files
COPY . .

# Create static directory and copy frontend files
RUN mkdir -p ./static ./static/tabla ./static/drums

COPY index.html styles.css script.js config.js ./static/
COPY tabla/ ./static/tabla/
COPY drums/ ./static/drums/

# Set permissions
RUN chmod -R 755 ./static


# Create non-root user for security
# RUN adduser --disabled-password --gecos '' appuser
# RUN chown -R appuser:appuser /app
# USER appuser

# Expose port 5010
EXPOSE 5010

# Start the backend server (which also serves static files)
CMD ["python", "api.py"]