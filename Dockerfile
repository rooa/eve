FROM jayanthkoushik/spy3
MAINTAINER Jayanth Koushik <jkoushik@cs.cmu.edu>

ARG DEBIAN_FRONTEND=noninteractive

USER root
RUN pip install git+git://github.com/Theano/Theano.git
RUN pip install keras
RUN pip install --upgrade pandas
RUN pip install tqdm

USER docker
RUN mkdir /home/docker/.theano
RUN echo '\
[global]\n\
device = cpu\n\
floatX = float32\n\
\n\
[blas]\n\
ldflags = -L/usr/lib -lopenblas\n\
\n\
[gcc]\n\
cxxflags = -O3\n\
\n\
[nvcc]\n\
fastmath = True' > /home/docker/.theano/config

RUN mkdir /home/docker/.keras
RUN echo '\
{\n\
    "image_data_format": "channels_first",\n\
    "epsilon": 1e-07,\n\
    "floatx": "float32",\n\
    "backend": "theano"\n\
}' > /home/docker/.keras/keras.json
