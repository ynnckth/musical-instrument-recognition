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

RUN pip install --upgrade pip==20.0.2
RUN pip install --upgrade --default-timeout=100 -r requirements.txt

EXPOSE 9002

CMD ["python", "-u", "./bastdm5/server.py"]
