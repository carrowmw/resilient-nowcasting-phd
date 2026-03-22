# Use a lightweight Python base image
FROM python:3.11-slim

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies (Adding curl and gnupg for Node.js)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    unzip \
    gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Install Node.js (Required for Gemini)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Install the Agents
# Install Gemini via npm
RUN npm install -g @google/gemini-cli
# Install Claude via native binary
RUN curl -fsSL https://claude.ai/install.sh | bash

# 4. Set up the working directory inside the container
WORKDIR /workspace

# 5. Copy your ML requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to open a bash shell
CMD ["/bin/bash"]