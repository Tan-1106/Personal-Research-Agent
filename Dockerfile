# Use lightweight Python 3.11 image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code into the container
COPY . .

# Command to run the bot
CMD ["python", "-m", "src.utils.telegram_bot"]
