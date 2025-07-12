# 使用 CUDA 11.8 + cuDNN8 映像，Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 安裝 Python 與必要工具
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git wget curl unzip libglib2.0-0 libgl1 libgomp1 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# 將 python3 指向 python
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# 需求檔（GPU 版本）
COPY . ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python3 -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 複製程式碼
COPY . .

# 設定啟動語法，假設主程式為 app.py，提供 HTTP /predict
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "app:app"]

