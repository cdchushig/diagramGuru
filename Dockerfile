FROM python:3.6-onbuild

WORKDIR /var/www/html

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies
RUN apt-get update && apt-get upgrade -y && apt-get autoclean

# install psycopg2 dependencies
RUN apt-get install -y \
    postgresql-dev gcc python3-dev musl-dev git \
    g++ gcc libxml2-dev libxslt-dev \
    libxslt \
    python-psycopg2 \
    libpq-dev \
    python-opencv

COPY ./requirements.txt /var/www/html

RUN  pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .