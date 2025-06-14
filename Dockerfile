FROM python:3.11-slim

RUN apt-get update && apt-get install -y git curl build-essential vim && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip

RUN git clone https://github.com/wd7512/seu-injection-framework /workspace

WORKDIR /workspace

COPY Research/data/ /workspace/Research/data/

RUN pip install -r pytorch_requirements.txt
RUN pip install -r requirements.txt
