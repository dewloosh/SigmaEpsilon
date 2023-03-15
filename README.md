# **SigmaEpsilon** - High-Performance Computational Solid Mechanics in Python

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dewloosh/SigmaEpsilon/main?labpath=notebooksnotebooks%2Flpp.ipynb?urlpath=lab)
[![CircleCI](https://circleci.com/gh/dewloosh/SigmaEpsilon.svg?style=shield)](https://circleci.com/gh/dewloosh/SigmaEpsilon) 
[![Documentation Status](https://readthedocs.org/projects/sigmaepsilon/badge/?version=latest)](https://sigmaepsilon.readthedocs.io/en/latest/?badge=latest) 
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/sigmaepsilon.svg)](https://pypi.org/project/sigmaepsilon)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Warning**
> This package is under active development and in an **alpha stage**. Come back later, or star the repo to make sure you don’t miss the first stable release!

## Highlights

Head over to the Quick Examples page in the docs to explore our gallery of examples showcasing what SigmaEpsilon can do! Want to test-drive SigmaEpsilon? All of the examples from the gallery are live on MyBinder for you to test drive without installing anything locally: Launch on Binder.

### Overview

* A `solid` submodule to analyze and optimize solid structures of all kinds with the **Finite Element Method**. The implementations so far only cover linear behaviour, but with practically no limits on the complexity of the shape and topology of the domain under investigation.

## Installation

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

We use Numba's JIT compiler to speed up heavy computations, and it relies on the C++ redistributable package. It is likely already installed on your system, but if it is not, you can download it from Microsoft's website under "Other Tools, Frameworks, and Redistributables".

must have

* `Numba`, `NumPy`, `SciPy`, `SymPy`, `awkward`

strongly suggested

* `PyVista`, `Plotly`, `matplotlib`, `sectionproperties`

optional

* `networkx`

## **License**

SigmaEpsilon is Copyright(C) 2022: Bence Balogh

All rights reserved.

This program is dual-licensed as follows:

(1) You may use SigmaEpsilon as free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

In this case the program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License at http://www.gnu.org/licenses/gpl.txt or in the LICENSE file of this repositiry for more details.

(2) You may use SigmaEpsilon as part of a commercial software. In this case a proper agreement must be reached with the Authors based on a proper licensing contract.