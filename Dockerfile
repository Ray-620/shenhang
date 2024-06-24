# Use an official Python runtime as a parent image
FROM python:3.9.7-slim

ADD . /app
# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.org/simple/

# Make port 3556 available to the world outside this container
EXPOSE 3556

# Run app.py when the container launches
CMD ["python", "app.py"]
