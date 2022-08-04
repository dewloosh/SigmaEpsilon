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

Mesh classes are all subclasses of :class:`sigmaepsilon.core.DeepDict`, which means
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

.. autoclass:: sigmaepsilon.mesh.PolyData
    :members:

Line Meshes
===========

.. autoclass:: sigmaepsilon.mesh.linedata.LineData
    :members:

Triangulations
==============

.. autoclass:: sigmaepsilon.mesh.tri.trimesh.TriMesh
    :members:


Tetrahedralizations
===================

.. autoclass:: sigmaepsilon.mesh.tet.tetmesh.TetMesh
    :members: 


Grids
=====

.. autoclass:: sigmaepsilon.mesh.grid.Grid
    :members: 


------------
Data Classes
------------

.. autoclass:: sigmaepsilon.mesh.pointdata.PointData
    :members:

.. autoclass:: sigmaepsilon.mesh.celldata.CellData
    :members:  