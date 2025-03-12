# CUDA12.6+cuDNN8 
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04
ENV ANACONDA /opt/anaconda
ENV PATH $ANACONDA/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libopencv-dev \
    axel \
    zip \
    unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/list/* 

RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh -P /tmp && \
    bash /tmp/Anaconda3-2024.06-1-Linux-x86_64.sh -b -p $ANACONDA && \
    rm -rf /tmp/Anaconda3-2024.06-1-Linux-x86_64.sh

RUN conda install -y -c anaconda pip
RUN conda install -y -c anaconda yaml
# TensorBoard を追加でインストール
RUN pip install tensorboard tensorboardX

# デフォルトの作業ディレクトリを設定
WORKDIR /workspace

# コンテナ起動時に Conda 環境を有効にする
CMD ["/bin/bash"]