FROM tiagopeixoto/graph-tool

RUN pacman -S gcc \
		python-pip \
		grep \
		diffutils \
		gawk \
		cmake \
		make \
		--noconfirm --needed

RUN pip install --upgrade pip

RUN pacman -S cython --noconfirm --needed

# Copy requirements
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --ignore-installed

# Entrypoint from home
COPY . /app/
WORKDIR /app/

# DEVEL
ENTRYPOINT /bin/bash
