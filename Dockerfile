# Multi-stage Docker build for frontend + backend
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./backend/

# Create static directory for frontend files
RUN mkdir -p ./static

# Copy frontend static files
COPY index.html ./static/
COPY styles.css ./static/
COPY script.js ./static/
COPY config.js ./static/

# Copy audio directories (create them if they don't exist)
COPY tabla/ ./static/tabla/ 2>/dev/null || mkdir -p ./static/tabla/
COPY drums/ ./static/drums/ 2>/dev/null || mkdir -p ./static/drums/

# Create placeholder files for audio directories if they're empty
RUN touch ./static/tabla/.gitkeep ./static/drums/.gitkeep

# Set proper permissions
RUN chmod -R 755 ./static

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port 5010
EXPOSE 5010

# Start the backend server (which also serves static files)
CMD ["python", "backend/app.py"]