# Docker file for the image_classification plugin app

FROM fnndsc/ubuntu-python3:latest
MAINTAINER fnndsc "dev@babymri.org"

ENV APPROOT="/usr/src/image_classification"  VERSION="0.1"
COPY ["image_classification", "${APPROOT}"]
COPY ["requirements.txt", "${APPROOT}"]

WORKDIR $APPROOT

RUN pip install -r requirements.txt

CMD ["image_classification.py", "--json"]