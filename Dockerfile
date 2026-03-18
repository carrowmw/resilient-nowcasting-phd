# Use a lightweight Python base image
FROM python:3.11-slim

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Claude Code using the native binary installer
RUN curl -fsSL https://claude.ai/install.sh | bash

# Set up the working directory inside the container
WORKDIR /workspace

# Copy your ML requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to open a bash shell
CMD ["/bin/bash"]
