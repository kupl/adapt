# Tutorial
ADAPT offers some tutorials about how to use it.
The offered tutorials are based on jupyter notebook.

## Jupyter Notebook with Docker
Our docker image offers a pre-installed jupyter notebook.
You can run a docker based jupyter notebook with the following commands.
```bash
$ cd /path/to/adapt/tutorial
$ docker run -v $(pwd):/workspace -u $(id -u):$(id -g) -p 8888:8888 adapt
```

## Jupyter Notebook with virtual environment
You can see the installation of jupyter notebook at [here](../README.md#Jupyter-notebook).
After, the following command will launch a jupyter notebook server.
```bash
(venv) $ jupyter notebook
```

## Contents
The following table shows the contents of this folder.

|                  File                  | Description                               |
|:--------------------------------------:|:------------------------------------------|
| [test_lenet5.ipynb](test_lenet5.ipynb) | Test LeNet-5 that trained for MNIST.      |
| [test_vgg19.ipynb](test_vgg19.ipynb)   | Test VGG-19 provided by Tensorflow/Keras. |
