#Lightweight Python
FROM python:3.7-slim

#Copy local files to the Container Image
WORKDIR /Literature-Search-Engine
ADD requirements.txt /Literature-Search-Engine/requirements.txt
ADD . /Literature-Search-Engine

#Install Dependencies
RUN pip install -r /Literature-Search-Engine/requirements.txt

#Run the flask service on container startup
CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 wsgi:app