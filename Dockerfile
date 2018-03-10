
FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y python \
    python-pip \
    python-tk \
    sox && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH /app

ADD . .
RUN pip install --upgrade --default-timeout=100 -r requirements.txt  
RUN pip install --upgrade matplotlib

EXPOSE 9002

CMD ["python", "-u", "./prediction/server/server.py"] 
