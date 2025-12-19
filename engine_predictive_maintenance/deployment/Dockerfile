# Base image
FROM python:3.9-slim

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /home/user/app

# Copy requirements first (better caching)
COPY deployment/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY deployment/app.py ./app.py

# Switch to non-root user
USER user

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
