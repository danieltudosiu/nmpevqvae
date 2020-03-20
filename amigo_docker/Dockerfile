FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
LABEL maintainer="thomas.varsavsky.15@ucl.ac.uk"

RUN apt-get update

RUN apt-get install -y \
      gcc-4.8 \
      g++-4.8 \
      curl \
      vim  \
      wget \
      make \
      htop \
      nano \
      tmux \
      libssl-dev \
      modemmanager \
      libpam-systemd:amd64 \
      policykit-1 \
      network-manager \
      build-essential \
      linux-headers-generic \
      module-init-tools \
      libbz2-dev \
      sqlite \
      libsqlite3-dev \
      libturbojpeg \
      zlib1g-dev \
      libpng-dev \
      libxml2-dev \
      libjpeg-dev \
      libx11-dev \
      checkinstall \
      libxext-dev \
      git

RUN wget https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tar.xz \
 && tar xf Python-3.6.4.tar.xz \
 && rm Python-3.6.4.tar.xz \
 && cd Python-3.6.4 \
 && apt-get install -y libssl-dev \
 && ./configure --enable-shared && make -j20 && make install \
 && cd ../ \
 && pip3 install --upgrade pip

RUN apt-get install -y sudo

RUN apt-get install -y libsm6 libxrender1 libfontconfig1

RUN pip3 install numpy \
 && pip3 install pillow \
 && pip3 install scipy \
 && pip3 install cython \
 && pip3 install scikit-image \
 && pip3 install typed_ast \
 && pip3 install seaborn \
 && pip3 install pandas==0.22.0 \
 && pip3 install sklearn \
 && pip3 install jupyter \
 && pip3 install simpleitk \
 && pip3 install nibabel \
 && pip3 install pymongo \
 && pip3 install opencv-python==4.1.1.26 \
 && pip3 install sacred==0.7.4 \
 && pip3 install tqdm \
 && pip3 install py3nvml \
 && pip3 install ninja==1.9.0.post1 \
 && pip3 install torch==1.2.0 \
 && pip3 install torchvision==0.4.0 \
 && pip3 install tensorflow-gpu==1.15 \
 && pip3 install packaging \
 && pip3 install blinker \
 && pip3 install pyyaml \
 && pip3 install fslpy


WORKDIR /root

EXPOSE 8888
ARG USER_NAME
ARG USER_ID
RUN adduser --uid ${USER_ID} --gecos '' ${USER_NAME}
RUN adduser ${USER_NAME} sudo
RUN passwd -de ${USER_NAME}

CMD ["/bin/bash"]
