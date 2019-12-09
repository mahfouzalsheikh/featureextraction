# Pull base image
FROM tensorflow/tensorflow:latest-gpu-py3
LABEL maintainer "Focus21 Inc."

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /code

# Install dependencies
COPY Pipfile Pipfile.lock /code/

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y gdal-bin
RUN apt-get install libgdal-dev -y
RUN apt-get install python3-gdal -y
RUN apt-get install -y libsm6 libxext6 libxrender1 libfontconfig1
RUN pip install --upgrade pip
RUN rm -rf /var/lib/apt/lists/*
RUN pip install pipenv
RUN pipenv install --system

# Copy project
COPY . /code/