FROM continuumio/anaconda3:2019.03

WORKDIR /workdir
COPY ./submit/requirements.txt /workdir/
RUN pip install -r requirements.txt
