# **SigmaEpsilon** - High-Performance Computational Mechanics in Python

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dewloosh/sigmaepsilon/main?labpath=examples%2Flpp.ipynb?urlpath=lab)
[![CircleCI](https://circleci.com/gh/dewloosh/sigmaepsilon.svg?style=shield)](https://circleci.com/gh/dewloosh/sigmaepsilon) 
[![Documentation Status](https://readthedocs.org/projects/sigmaepsilon/badge/?version=latest)](https://sigmaepsilon.readthedocs.io/en/latest/?badge=latest) 
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/sigmaepsilon.svg)](https://pypi.org/project/sigmaepsilon) 


> **Warning**
> This package is under active development and in an **alpha stage**. Come back later, or star the repo to make sure you don’t miss the first stable release!

## Highlights

Head over to the Quick Examples page in the docs to explore our gallery of examples showcasing what SigmaEpsilon can do! Want to test-drive SigmaEpsilon? All of the examples from the gallery are live on MyBinder for you to test drive without installing anything locally: Launch on Binder.

### Overview

* A `math` submodule including general-purpose solutions for problems in **Linear Algebra**, **Linear and Nonlinear Optimization** and **Graph Theory**.

* A `mesh` submodule to handle compound polygonal meshes, with support for transformations, mesh analysis, simple mesh generation, plotting etc. It includes a domain specific version of the framework presented in `sighmaepsilon.math` to handle vector spaces, fine tuned for Euclidean geometry. 

* A `solid` submodule to analyze and optimize solid structures of all kinds with the **Finite Element Method**. The implementations so far only cover linear behaviour, but with practically no limits on the complexity of the shape and topology of the domain under investigation.

## **Installation**
This is optional, but we suggest you to create a dedicated virtual enviroment at all times to avoid conflicts with your other projects. Create a folder, open a command shell in that folder and use the following command

```console
>>> python -m venv venv_name
```

Once the enviroment is created, activate it via typing

```console
>>> .\venv_name\Scripts\activate
```

`sigmaepsilon` can be installed (either in a virtual enviroment or globally) from PyPI using `pip` on Python >= 3.6:

```console
>>> pip install sigmaepsilon
```

## **Documentation**

Refer to the [docs](https://sigmaepsilon.readthedocs.io/en/latest/) for further details on installation and usage.

## **Testing**

To run all tests, open up a console in the root directory of the project and type the following

```console
>>> python -m unittest
```

## **Dependencies**

must have 
  * `Numba`, `NumPy`, `SciPy`, `SymPy`, `awkward`

stringly suggested
  * `PyVista`, `Plotly`, `matplotlib`, `sectionproperties`

optional 
  * `networkx`

## **License**

This package is licensed under the MIT license.