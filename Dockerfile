# Use stable base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Set timeout for slow internet
ENV PIP_DEFAULT_TIMEOUT=1000

# Copy project
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch separately (faster + reliable)
RUN pip install torch==1.13.1 torchvision==0.14.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]