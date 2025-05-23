# base image
FROM continuumio/miniconda3

# install commands for the system dependencies
RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
    ca-certificates gcc libffi-dev wget unzip git openssh-client gnupg curl \
    python3-dev python3-setuptools 

WORKDIR /app

# create the environment for Conda-based application
COPY prod_environment.yaml .
RUN conda env create -f prod_environment.yaml

# make run commands use the new environment
# override the default shell with SHELL command
SHELL ["conda", "run", "-n", "plantations5", "/bin/bash", "-c"]

# separately install the C++ version of GDAL
# add-apt-repository is not in base image, so first install software properties
RUN apt-get update && apt-get install -y software-properties-common &&\
    add-apt-repository ppa:ubuntugis/ppa &&\
 	apt-get -y install gdal-bin &&\
 	apt-get -y install libgdal-dev &&\
 	export CPLUS_INCLUDE_PATH=/usr/include/gdal &&\
 	export C_INCLUDE_PATH=/usr/include/gdal &&\
 	pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') --global-option=build_ext --global-option="-I/usr/include/gdal"

# production reqs - make sure to check dockerignore
COPY src/ /app/src/
COPY config.yaml /app/config.yaml 
COPY params.yaml /app/params.yaml 
COPY src/utils/validate_io.py /app/src/utils/validate_io.py
COPY src/utils/mosaic.py /app/src/utils/mosaic.py
COPY src/production.py /app/src/production.py
COPY data/train/selected_features.json /app/data/train/selected_features.json
COPY models/model.joblib /app/models/model.joblib

# texture reqs
# COPY src/utils/validate_io.py /app/src/validate_io.py
# COPY src/features/texture_analysis.py /app/src/texture_analysis.py
# COPY config.yaml /app/config.yaml

EXPOSE 8080

# list of commands - supply args with docker run?
ENTRYPOINT ["conda", "run","--no-capture-output", "-n", "plantations5", "python3", "src/production.py"]