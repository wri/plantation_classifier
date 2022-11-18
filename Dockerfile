# base image
FROM continuumio/miniconda3

# install commands for the system dependencies
RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
    ca-certificates gcc libffi-dev wget unzip git openssh-client gnupg curl \
    python3-dev python3-setuptools

WORKDIR /src

# create the environment (this is a Conda-based application)
COPY environment.yaml .
RUN conda env create -f environment.yaml

# make run commands use the new environment
# override default shell with SHELL command
SHELL ["conda", "run", "-n", "plantenv", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure boto3 is installed:"
RUN python -c "import boto3"

# separately install the C++ version of GDAL
RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update &&\
 	apt-get -y install gdal-bin &&\
 	apt-get -y install libgdal-dev &&\
 	export CPLUS_INCLUDE_PATH=/usr/include/gdal &&\
 	export C_INCLUDE_PATH=/usr/include/gdal &&\
 	pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') --global-option=build_ext --global-option="-I/usr/include/gdal"

# copy over the appropriate scripts
COPY src/* src/

EXPOSE 8080

# list of commands - confirm arguments are correct here
ENTRYPOINT ["conda", "run","--no-capture-output", "-n", "plantenv", "python3", "plantation_classifier.py" "tile_idx" "country" "model"]



# FROM tensorflow/tensorflow:1.15.5-gpu-py3

# delete old keys and fetch new ones in order to get base image set up
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list 
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# # install commands for the system dependencies
# RUN apt-get update -y && apt-get install --no-install-recommends -y -q \
#     ca-certificates gcc libffi-dev wget unzip git openssh-client gnupg curl \
#     python3-dev python3-setuptools

# # Create directory structure
# RUN mkdir src
# WORKDIR /src
# COPY requirements.txt requirements.txt

# # John's suggestion for conda:
# RUN conda install --force-reinstall -y -q --name $ENV_NAME -c conda-forge --file requirements.txt

# # copy and install system dependencies
# # RUN pip install --upgrade pip
# # COPY requirements.txt requirements.txt
# # RUN pip install -r requirements.txt

# # copy over the appropriate scripts
# COPY src/* src/

# # not sure if this is required for gdal if I have it in reqs.txt?
# RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update &&\
#  	apt-get -y install gdal-bin &&\
#  	apt-get -y install libgdal-dev &&\
#  	export CPLUS_INCLUDE_PATH=/usr/include/gdal &&\
#  	export C_INCLUDE_PATH=/usr/include/gdal &&\
#  	pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') --global-option=build_ext --global-option="-I/usr/include/gdal"

# EXPOSE 8080

# # list of commands
# CMD python3 plantation_classifier.py --tile_idx --country --model

# # # docker build -t plantations:1.0 (rename)

