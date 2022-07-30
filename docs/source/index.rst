=====================================================================
**SigmaEpsilon** - High-Performance Computational Mechanics in Python
=====================================================================

Features
--------

* Batteries to handle linear algebra in Euclidean Space. The class architecture 
  allows for an arbitrary level of nested frames embedded into one another, 
  as well as the transformation of any number of vectors between any two of these 
  frames. 

* It can be used as a configurator for plots produced with `PyVista`.

* A set of classes and routines to handle jagged topologies. 

* A facility for plotting over flat traingulations in 2d using `matplotlib`.

* A small set of recipes for mesh generation. It includes a grid generator that
  generates all kinds of grids suitable for the **Finite Element Method**, as it
  is capable of using a wide range of linear and nonlinear base classes. 
  
* A set of utility routines for various operations on meshes. Voxelization, 
  extrusion, tiling, etc.


Requirements
------------

The implementations in this module are created on top of 

* | `NumPy`, `SciPy` and `Numba` to speed up computationally sensitive parts,

* | `SymPy` for symbolic operations and some vector algebra,

* | the `awkward` library for its high performance data structures, gpu support
  | and general `Numba` compliance.


Gallery
-------

.. nbgallery::
    :name: rst-gallery
    :glob:
    :reversed:

    _notebooks/*
    
        
Contents
--------

.. toctree::
    :maxdepth: 2
   
    user_guide

API
---

.. toctree::
    :maxdepth: 3
   
    api
   
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`