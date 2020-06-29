from setuptools import find_packages
from setuptools import setup

setup_params = dict(
  name='adapt',
  version='1.0',
  description='Adapt is a white-box testing framework for deep neural networks',
  url='http://prl.korea.ac.kr',
  download_url='https://github.com/kupl/ADAPT',
  author='Software Analysis Laboratory, Korea University',
  license='MIT',
  packages=find_packages(exclude=['docker', 'tutorial', 'venv']),
  setup_requires=[], 
  install_requires=['tensorflow>=2.0.0', 'imageio'], 
  dependency_links=[],
)

if __name__ == '__main__':
  setup(**setup_params)
