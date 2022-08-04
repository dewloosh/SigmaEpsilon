=================================
Linear Algebra in Euclidean Space
=================================

These classes are meant to provide a usable representation of a vectorspace
for applications related to computational geometry in the domain of Euclidean geometry.

For a quick recap on the math involved here, go to 
:doc:`/api/math/api_vs`, where you find a brief summary and a downloadable pdf 
file with a more complete explanation.

.. note::
    Currently the implementation only covers vectors with orthonormal base vectors.

Points
======

The module provides two classes to handle points in 3d space. The `Point`_ class
is able to handle the transformations of one or several points.

.. _Point:

.. autoclass:: sigmaepsilon.mesh.space.point.Point
    :members:

.. autoclass:: sigmaepsilon.mesh.space.PointCloud
    :members: 

Cartesian Reference Frames
==========================

.. autoclass:: sigmaepsilon.mesh.space.CartesianFrame
    :members:
    :inherited-members:  orient, orient_new, show, dcm, axes, root, eye