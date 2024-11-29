# Ambit-compatible dolfinx Docker image (29 Nov 2024)
FROM ghcr.io/fenics/dolfinx/dolfinx@sha256:6bed905ff94a1810707ec6d70f34adb52d2157d22fa5505b40760a126cc31f44

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
RUN pip3 -v install --no-cache-dir pytest

# remove contents of the temporary directory to minimize the size of the image
RUN rm -rf /tmp
