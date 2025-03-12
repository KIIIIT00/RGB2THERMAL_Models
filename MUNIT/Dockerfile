FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
ENV TZ=Asia/Tokyo
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone
    
# Set anaconda path
ENV ANACONDA /opt/anaconda
ENV PATH $ANACONDA/bin:$PATH

RUN apt-get update && \
 apt-get install -y --no-install-recommends \
         wget \
         libopencv-dev \
         python3-opencv \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         axel \
         zip \
         unzip && \
         apt-get clean && \
         rm -rf /var/lib/apt/list/*

# Install the latest Anaconda
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -P /tmp
RUN bash /tmp/Anaconda3-5.0.1-Linux-x86_64.sh -b -p $ANACONDA
RUN rm /tmp/Anaconda3-5.0.1-Linux-x86_64.sh -rf

# Install PyTorch for CUDA 12.4
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install pip and other dependencies
RUN conda install -y -c anaconda pip
RUN conda install -y -c anaconda yaml
RUN pip install tensorboard tensorboardX
