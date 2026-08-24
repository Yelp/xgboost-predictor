FROM --platform=linux/amd64 amazoncorretto:17

RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime

RUN dnf install -y \
    findutils \
    git \
    make \
    unzip \
    zip \
    && dnf clean all

WORKDIR /workspace
