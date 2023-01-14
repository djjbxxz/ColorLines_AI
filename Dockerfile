# Using official ubuntu image as a parent image
FROM ubuntu:22.04

# Setting the working directory to /app
WORKDIR /colorline

# Copy the current directory contents into the container at /app
COPY . /colorline

# Getting the updates for Ubuntu and installing python into our environment
RUN apt-get -y update  && apt-get install -y python

RUN apt-get update \  
       && apt-get install -y --no-install-recommends \  
       apt-utils \  
       build-essential \   
       cmake 


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Run app.py when the container launches
# CMD ["python", "app.py"]