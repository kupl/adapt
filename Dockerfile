FROM tensorflow/tensorflow:2.2.0-gpu

# Install adapt.
WORKDIR /src
COPY . .
RUN rm -r docker
RUN pip install . --no-deps --no-cache-dir
RUN pip install imageio --no-cache-dir

# Configure docker bash.
COPY ./docker/bash.bashrc /etc/bash.bashrc

# Set entry point.
WORKDIR /workdir
CMD bash
