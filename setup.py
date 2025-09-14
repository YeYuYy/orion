# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['orion',
 'orion.backend',
 'orion.backend.lattigo',
 'orion.backend.python',
 'orion.core',
 'orion.models',
 'orion.nn']

package_data = \
{'': ['*'], 'orion.backend': ['heaan/*', 'openfhe/*']}

install_requires = \
['PyYAML>=6.0',
 'certifi>=2024.2.2',
 'h5py>=3.5.0',
 'matplotlib>=3.1.0',
 'numpy>=1.21.0',
 'scipy>=1.7.0,<=1.14.1',
 'torch>=2.2.0',
 'torchvision>=0.17.0',
 'tqdm>=4.30.0']

setup_kwargs = {
    'name': 'orion-fhe',
    'version': '1.0.2',
    'description': 'A Fully Homomorphic Encryption Framework for Deep Learning',
    'long_description': "# Orion\n\n## Installation\nWe tested our implementation on `Ubuntu 22.04.5 LTS`. First, install the required dependencies:\n\n```\nsudo apt update && sudo apt install -y \\\n    build-essential git wget curl ca-certificates \\\n    python3 python3-pip python3-venv \\\n    unzip pkg-config libgmp-dev libssl-dev\n```\n\nInstall Go (for Lattigo backend):\n\n```\ncd /tmp\nwget https://go.dev/dl/go1.22.3.linux-amd64.tar.gz\nsudo tar -C /usr/local -xzf go1.22.3.linux-amd64.tar.gz\necho 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc\nsource ~/.bashrc\ngo version # go version go1.22.3 linux/amd64\n```\n\n### Install Orion\n\n```\ngit clone https://github.com/baahl-nyu/orion.git\ncd orion/\npip install -e .\n```\n\n### Run the examples!\n\n```\ncd examples/\npython3 run_lola.py\n```\n",
    'author': 'Austin Ebel',
    'author_email': 'abe5240@nyu.edu',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<3.13',
}
from tools.build_lattigo import *
build(setup_kwargs)

setup(**setup_kwargs)
