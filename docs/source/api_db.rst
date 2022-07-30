=====================
Composition of a Mesh
=====================

The classes in this module are the backbone of the data model behind every model.
A model consists of a `DeepDict` instance responble to orchestrate data related
to sets of points and cells in a compund polygonal mesh, similar to how kuberetes
orchestrates containers.

Data Model
----------
TODO

------------
Mesh Classes
------------

Mesh classes are all subclasses of :class:`dewloosh.core.DeepDict`, which means
that in the first place a mesh is a nested dictionary, with a self-replicating
default factory. Every dictionary inside a sturcure like this can hold on to either
points or cells, or both. They also can store data related to the points or the cells,
which can be later referenced, similar how `plotly` channels ids to `pandas` dataframes.

The `PolyData` class is the base class of all the other mesh classes. In principle, it
should be able to handle all kinds of inputs and gives more control in overall.
The subclasses like `TriMesh` and `LineMesh` simplify mesh composition by using 
algorithms specific to the cell type, which makes calculations go faster. 
These topology-specific classes have presets, which results in a fewer number of 
arguments in overall, resulting in more readable code. 

.. autoclass:: dewloosh.mesh.PolyData
    :members:

Line Meshes
===========

.. autoclass:: dewloosh.mesh.linedata.LineData
    :members:

Triangulations
==============

.. autoclass:: dewloosh.mesh.tri.trimesh.TriMesh
    :members:


Tetrahedralizations
===================

.. autoclass:: dewloosh.mesh.tet.tetmesh.TetMesh
    :members: 


Grids
=====

.. autoclass:: dewloosh.mesh.rgrid.Grid
    :members: 


------------
Data Classes
------------

.. autoclass:: dewloosh.mesh.pointdata.PointData
    :members:

.. autoclass:: dewloosh.mesh.celldata.CellData
    :members:  