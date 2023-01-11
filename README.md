# nonlocal-boxes

## Content

The repository is structured as follows. We only describe the most important files for a new user.
```bash
./
|-- nonlocal-boxes: Core of package. 
|  |-- toto.py    : Something
|-- ipynb: Contains Python notebooks which demonstrate how the code works
|  |-- Eval.ipynb: Fast evaluation of the function to optimize
|  |-- Eval2.ipynb: Fast evaluation of the function to optimize (V2)
|-- tests: Unit tests
|-- README.md: This file
```

## Installation

1. Create new virtual environment

```bash
$ python3 -m venv .venv_boxes
```

(Do
sudo apt install python3-venv
if needed)

3. Activate virtual environment

```bash
$ source .venv_boxes/bin/activate
```

4. Upgrade pip, wheel and setuptools 

```bash
$ pip install --upgrade pip
$ pip install --upgrade setuptools
$ pip install wheel
```

5. Install the `nonLocalBoxes` package (later).

```bash
python setup.py develop
```

6. (Optional) In order to use Jupyter with this virtual environment .venv
```bash
pip install ipykernel
python -m ipykernel install --user --name=.venv_boxes
```
(see https://janakiev.com/blog/jupyter-virtual-envs/ for details)

7. (Not needed if step 5 is used) Packages
```bash
pip install numpy matplotlib scipy torch
```

## Configuration
Nothing to do

## Credits
Later