# base image
FROM continuumio/miniconda3

# install commands for the system dependencies
RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
    ca-certificates gcc libffi-dev wget unzip git openssh-client gnupg curl \
    python3-dev python3-setuptools

WORKDIR /app

# create the environment for Conda-based application
COPY environment.yaml .
RUN conda env create -f environment.yaml

# make run commands use the new environment
# override the default shell with SHELL command
SHELL ["conda", "run", "-n", "plantations3", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure boto3 is installed:"
RUN python -c "import boto3"

# separately install the C++ version of GDAL
# add-apt-repository is not in base image, so first install software properties
RUN apt-get update && apt-get install -y software-properties-common &&\
    add-apt-repository ppa:ubuntugis/ppa &&\
 	apt-get -y install gdal-bin &&\
 	apt-get -y install libgdal-dev &&\
 	export CPLUS_INCLUDE_PATH=/usr/include/gdal &&\
 	export C_INCLUDE_PATH=/usr/include/gdal &&\
 	pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') --global-option=build_ext --global-option="-I/usr/include/gdal"

# copy over the appropriate scripts/data
COPY src/ /app/src/
COPY data/ashanti.csv /app/data/ashanti.csv
COPY data/urbanmask.tif /app/data/urbanmask.tif
COPY models/cat_model_v11.pkl /app/models/cat_model_v11.pkl

EXPOSE 8080

# list of commands - removed args
ENTRYPOINT ["conda", "run","--no-capture-output", "-n", "plantations3", "python3", "src/plantation_classifier.py"]