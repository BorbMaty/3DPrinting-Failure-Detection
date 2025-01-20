# Base python image
FROM python:3.9-slim

# Create the working directory in the container
WORKDIR /app

# install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# run main.py
CMD ["python", "main.py"]
