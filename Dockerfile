FROM python:3.7

RUN useradd --create-home appuser
USER appuser

# set default environment variables
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

ENV PORT=8888

RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata \
        libopencv-dev \
        build-essential \
        libssl-dev \
        libpq-dev \
        libcurl4-gnutls-dev \
        libexpat1-dev \
        python3-setuptools \
        python3-pip \
        python3-dev \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app
ADD . /app/

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

