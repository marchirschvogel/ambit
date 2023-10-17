# Ambit-compatible dolfinx Docker image (19 Aug 2023)
FROM ghcr.io/fenics/dolfinx/dolfinx@sha256:1f374e90d5e918a71a4bdba994bf434cdaf84fedc47aa11ac33295864180fb76

LABEL org.opencontainers.image.source=https://github.com/marchirschvogel/ambit
LABEL org.opencontainers.image.version=latest
LABEL org.opencontainers.image.description="ambit"
LABEL org.opencontainers.image.licenses=MIT

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

# install pip and git with apt
RUN apt-get update && apt-get install -y python3-pip git

# upgrade pip and setuptools
RUN python3 -m pip install pip setuptools --upgrade

# set the working directory to install docker dependencies
WORKDIR /tmp

# install Python module
RUN git clone https://github.com/marchirschvogel/ambit.git
RUN python3 -m pip install git+https://github.com/marchirschvogel/ambit.git --no-cache-dir

# remove contents of the temporary directory to minimize the size of the image
RUN rm -rf /tmp
