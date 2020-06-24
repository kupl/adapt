# ADAPT
ADAPT is the open source white-box testing framework for deep neural networks, which first introduced
in [Effective White-Box Testing for Deep Neural Networks with Adaptive Neuron-Selection Strategy](http://prl.korea.ac.kr/~pronto/home/papers/issta20.pdf).

## Download
Download this repository using git.
```bash
$ git clone https://github.com/kupl/adapt.git
$ cd adapt
```
The following commands in this document will be executed inside the ```adapt``` folder.

## Docker
ADAPT offers a docker image with pre-installed ADAPT.
If you want to install docker, please see [here](https://docs.docker.com/get-docker/).
Following command will create an image.
```bash
$ docker build . --tag adapt
```
To open a bash using the docker image, use the following command.
```bash
$ docker run -it -v $(pwd)/tutorial:/workspace -u $(id -u):$(id -g) adapt bash
```
If no commands are pass to docker, jupyter notebook server will be launched. The command will open a server at [http://localhost:8888/](http://localhost:8888).
```bash
$ docker run -v $(pwd)/tutorial:/workspace -u $(id -u):$(id -g) -p 8888:8888 adapt
```

## Local Installation
ADAPT uses [Tensorflow 2.0](https://www.tensorflow.org/). To install Tensorflow 2.0, you need a ```pip>=19.0```.
The following commands will create a virtual environment and update ```pip``` with the Ubuntu machine.
If you successfully create a virtual environment, your shell will be prefixed by ```(venv)```.
*Updating system ```pip``` using ```pip``` can cause some [problems](https://github.com/pypa/pip/issues/5599), so using virtual enviroment is **highly recommended**.
More detailed instructions can be found in [here](https://www.tensorflow.org/install/pip).*
``` bash
$ sudo apt update
$ sudo apt install python3-dev python3-pip python3-venv
$ python3 -m venv --system-site-packages ./venv
$ source ./venv/bin/activate
(venv) $ pip install --upgrade pip
```
Install ADAPT with the following command. ADAPT will automatically install all dependancies, including ```tensorflow```.
```bash
(venv) $ pip install .
```
To deactivate the virtual environment, type the following command. Then, ```(venv)``` in front of your shell will disappear.
```
(venv) $ deactivate
```

### Jupyter notebook
The following command will install jupyter notebook in your virtual environment.
```bash
(venv) $ pip install notebook
```
To launch jupyter notebook server, type the following command.
```bash
(venv) $ jupyter notebook
```
If your machine have jupyter notebook installed in system-wide, you can install a kernel with the following commands.
```bash
(venv) $ pip install -I ipykernel
(venv) $ python -m ipykernel install --user --name ADAPT
```
Now, you can see the ```ADAPT``` entry when you create a new notebook.

## Tutorial
ADAPT offeres some tutorials at [tutorial](tutorial).

## Issues
We are welcome any issues. Please, leave them in the Issue tab.