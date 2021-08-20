# FROM python:3.6
# LABEL maintainer="matheus.sasso17@gmail.com"

# # Install the C compiler tools
# RUN apt-get update -y && \
#   apt-get install build-essential -y && \
#   pip install --upgrade pip

# # Install libspatialindex
# WORKDIR /tmp
# RUN wget http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.5.tar.gz && \
#   tar -xvzf spatialindex-src-1.8.5.tar.gz && \
#   cd spatialindex-src-1.8.5 && \
#   ./configure && \
#   make && \
#   make install && \
#   cd - && \
#   rm -rf spatialindex-src-1.8.5* && \
#   ldconfig

# # Install rtree and geopandas
# RUN pip install rtree geopandas

FROM ubuntu:18.04

LABEL maintainer="msasso@cpqd.com.br"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    vim \
    chromium-browser \
    curl \
    libssl-dev \
    git \
    mercurial \
    pepperflashplugin-nonfree \
    libffi-dev \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app/

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

#installing requirements
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

#Exposing Ports
EXPOSE 8080
EXPOSE 6006

USER root

# Entrypoint
ENTRYPOINT ["/bin/bash"]
CMD []

