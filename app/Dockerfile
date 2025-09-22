# Base python image
FROM python:3.11-slim

# Create the working directory in the container
WORKDIR /app

# install the dependencies
COPY requirements.txt .
RUN apt-get update && apt-get upgrade -y && apt-get clean && \
	pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# run main.py
CMD ["python", "teacher.py"]
