FROM tensorflow/tensorflow:2.2.0-gpu-jupyter

# Install adapt.
WORKDIR /src
COPY . .
RUN rm -r docker
RUN pip install . --no-deps --no-cache-dir
RUN pip install imageio --no-cache-dir

# Configure docker bash.
COPY ./docker/bash.bashrc /etc/bash.bashrc

# Set entry point.
WORKDIR /workspace
EXPOSE 8888
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/workspace --ip 0.0.0.0 --no-browser --allow-root"]
