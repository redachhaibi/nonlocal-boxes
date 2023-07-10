# nonlocal-boxes

## Content

The repository is structured as follows. We only describe the most important files for a new user.
```bash
./
|-- nonlocal-boxes/: Core of package. 
|  |-- `__init__.py`: init file.
|  |-- 'evaluate.py': package of the function to evaluate, using PyTorch (and 'evaluate_with_numpy.py' for the same with NumPy).
|  |-- 'utils.py': package of the constants, using PyTorch (and 'utils_with_numpy.py' for the same with NumPy)
|-- ipynb/: Contains Python notebooks which demonstrate how the code works
|  |-- `BG15.ipynb`: implementation of [BG15]
|  |-- `Check-triangle-PR-P0-P1.ipynb`: Numerically check the known result in the triangle {PR, P0, P1}.
|  |-- `Eval.ipynb`: Fast evaluation of the function to optimize.
|  |-- `Eval2.ipynb`: Fast evaluation of the function to optimize (V2).
|  |-- `Eval2 - notations.pdf`: Explains the notations of `Eval2.ipynb`.
|  |-- `Eval3.ipynb`: Fast evaluation of the function to optimize (V3): now W is a matrix.
|  |-- `Eval3 - notations.pdf`: Explains the notations of `Eval3.ipynb`.
|  |-- `Eval4.ipynb`: Tests with the package containing the function to evaluate.
|  |-- `Gradient-Descent.ipynb`: Basic gradient descent.
|  |-- `Gradient-Descent-2.ipynb`: Clean version of `Gradient-Descent.ipynb`.
|  |-- `Gradient-Descent-3.ipynb`: Same as `Gradient-Descent-2.ipynb`, but with tests with different boxes. -> FIND NEW COLLAPSING WIRING (+ heavy ball + line search)
|  |-- `Pytorch-tests.ipynb`: Tests with PyTorch.
|  |-- `sdf2pointCloud_2D.ipynb`: Reda s example of gradient descent.
|  |-- `Test-Wiring.ipynb`: Tests if a new wiring is new and check which triangle is collapsed.
|-- tests/: Unit tests
|-- `README.md`: This file
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

5. Install the `nonLocalBoxes` package.

```bash
$ python setup.py develop
```

6. (Optional) In order to use Jupyter with this virtual environment .venv
```bash
$ pip install ipykernel
$ python -m ipykernel install --user --name=.venv_boxes
```
(see https://janakiev.com/blog/jupyter-virtual-envs/ for details)

7. (Not needed if step 5 is used) Packages
```bash
$ pip install numpy matplotlib scipy torch panel jupyter_bokeh
```

## Configuration
Nothing to do

## Credits
Later
