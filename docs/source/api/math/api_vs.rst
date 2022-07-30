==============
Linear Algebra
==============

These classes are meant to provide a usable representation of a vectorspace.
At the moment, the capabilities are limited to vectors and their transformations 
in coordinate frames with orthonormal basis vectors, but allows for arbitrary 
interrelations of them.

The Data Model
==============

Every vector and frame class is a subclass of ``numpy.ndarray``. The way of inheritance
is the mixture of both approaches suggested in `NumPy`'s docs. A data class is inherited
by directly subclassing ``numpy.ndarray``, and is responsible to store coordinates in the
actual coordinate frame of the vector. This is the representation of a vector in a frame.
The class for vectors is implemented the other way arounds, and its instances 
actually wrap an instance of the first class. The purpose of the second class is to
handle frames and transformations on the stored array. This creates a frontend-backend
architecture, where a frontend can be served by several kinds of backends. An example for
this is the ``TopologyArray`` class, which operates on either a ``numpy.ndarray``
or an ``awkward.Array`` (depending on the arguments at object creation), yet the
frontend provides a unified interface.

The Direction Cosine Matrix
===========================

The notion of the *Direction Cosine Matrix* (DCM) is meant to unify the direction of 
relative transformation between two frames.

.. note::
   Click :download:`here <../../_static/linalgR3.pdf>` to read the extended version of this
   brief intro as a pdf document.

If a vector :math:`\mathbf{v}` is given in frames :math:`\mathbf{A}` and :math:`\mathbf{B}` as

.. math::
   :nowrap:
   
   \begin{equation}
   \mathbf{v} = \alpha_1 \mathbf{a}_{1} + \alpha_2 \mathbf{a}_{2} = \beta_1 \mathbf{b}_{1} + \beta_2 \mathbf{b}_{2},
   \end{equation}
   
then the matrix :math:`^{A}\mathbf{R}^{B}` is called *the DCM from A to B*. It transforms the components as

.. math::
   :nowrap:

   \begin{equation}
   \left[
   \boldsymbol{\beta}
   \right]
   = 
   \left[
   ^{A}\mathbf{R}^{B}
   \right]
   \left[ \boldsymbol{\alpha}\right],
   \quad
   \left[
   \boldsymbol{\alpha}
   \right]
   = 
   \left[
   ^{A}\mathbf{R}^{B}
   \right]^{-1}
   \left[ \boldsymbol{\beta}\right]
   =
   \left[
   ^{B}\mathbf{R}^{A}
   \right]
   \left[ \boldsymbol{\beta}\right]
   \end{equation}

and the base vectors as

.. math::
   :nowrap:

   \begin{equation}
   \left[
   \mathbf{b}_i
   \right]
   = 
   \left[
   ^{B}\mathbf{R}^{A}
   \right]
   \left[ \mathbf{a}_i \right], \quad (i=1,2,3,...).
   \end{equation}

Vectors
=======

.. autoclass:: sigmaepsilon.math.linalg.vector.VectorBase
    :members:

.. autoclass:: sigmaepsilon.math.linalg.Vector
    :members:

Frames
======

.. autoclass:: sigmaepsilon.math.linalg.ReferenceFrame
    :members:

Arrays
======

.. autoclass:: sigmaepsilon.math.linalg.array.ArrayBase
    :members:

.. autoclass:: sigmaepsilon.math.linalg.array.Array
    :members:

.. autoclass:: sigmaepsilon.math.linalg.sparse.JaggedArray
    :members:

.. autoclass:: sigmaepsilon.math.linalg.sparse.csr_matrix
    :members: