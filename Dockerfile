FROM python:3.11-slim

RUN apt-get update && apt-get install -y git curl && \
    pip install --upgrade pip

RUN apt install build-essential

# Clone your repo first
RUN git clone https://github.com/wd7512/seu-injection-framework /workspace

WORKDIR /workspace

# Install PyTorch and other requirements
RUN pip install -r pytorch_requirements.txt
RUN pip install -r requirements.txt
