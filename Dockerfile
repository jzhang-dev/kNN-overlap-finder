FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    build-essential \
    bowtie2 \
    bwa \
    cmake \
    curl \
    git \
    minimap2 \
        nano \
    python3 \
    python3-pip \
    r-base \
    samtools \
    screen \
    unzip \
    vim \
    wget \
    libncurses5-dev \
    libncursesw5-dev \
    libbz2-dev \
    zlib1g-dev \
    liblzma-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libtbb-dev \
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    libhdf5-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*
    
# Install Conda
ENV DEBIAN_FRONTEND=noninteractive
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Create a non-root user
RUN useradd -m miaocj
USER miaocj
