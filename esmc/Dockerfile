FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt --break-system-packages
COPY src .
