from typing import Any, Iterable

import numpy as np
from numpy import ndarray
from sectionproperties.analysis.section import Section

from polymesh.config import __hasplotly__, __hasmatplotlib__

if __hasplotly__:
    from dewloosh.plotly import plot_lines_3d

from .mesh import FemMesh
from .cells import B2, B3


class LineMesh(FemMesh):
    """
    A data class dedicated to 1d cells. It handles sections and other line
    related information, plotting, etc.

    See Also
    --------
    :class:`sigmaepsilon.fem.mesh.FemMesh`
    """

    _cell_classes_ = {
        2: B2,
        3: B3,
    }

    def plot(self, *args, scalars=None, backend="plotly", scalar_labels=None, **kwargs):
        """
        Plots the line elements using one of the supported backends.

        Parameters
        ----------
        scalars : numpy.ndarray, Optional
            Data to plot. Default is None.

        backend : str, Optional
            The backend to use for plotting. Available options are 'plotly' and 'vtk'.
            Default is 'plotly'.

        scalar_labels : Iterable, Optional
            Labels of the scalars in 'scalars'. Only if Plotly is selected as the backend.
            Defaeult is None.

        Returns
        -------
        Any
            A PyVista or a Plotly object.

        """
        if backend == "vtk":
            return self.pvplot(
                *args, scalars=scalars, scalar_labels=scalar_labels, **kwargs
            )
        elif backend == "plotly":
            assert __hasplotly__
            coords = self.coords()
            topo = self.topology()
            return plot_lines_3d(coords, topo)
        elif backend == "mpl":
            raise NotImplementedError
            assert __hasmatplotlib__
        else:
            msg = "No implementation for backend '{}'".format(backend)
            raise NotImplementedError(msg)

    def plot_dof_solution(
        self,
        *args,
        backend: str = "plotly",
        case: int = 0,
        component: int = 0,
        labels: Iterable = None,
        **kwargs
    ) -> Any:
        """
        Plots degrees of freedom solution using 'vtk' or 'plotly'.

        Parameters
        ----------
        scalars : numpy.ndarray, Optional
            Data to plot. Default is None.
        case : int, Optional
            The index of the load case. Default is 0.
        component : int, Optional
            The index of the DOF component. Default is 0.
        labels : Iterable, Optional
            Labels of the DOFs. Only if Plotly is selected as the backend.
            Defaeult is None.
        **kwargs : dict, Optional
            Keyqord arguments forwarded to :func:`pvplot`.

        Returns
        -------
        Any
            A figure object or None, depending on the selected backend.
        """
        if backend == "vtk":
            scalars = self.nodal_dof_solution()[:, component, case]
            return self.pvplot(*args, scalars=scalars, **kwargs)
        elif backend == "plotly":
            scalar_labels = labels if labels is not None else ["U", "V", "W"]
            coords = self.coords()
            topo = self.topology()
            dofsol = self.nodal_dof_solution()[:, :3, case]
            return plot_lines_3d(
                coords, topo, scalars=dofsol, scalar_labels=scalar_labels
            )
        else:
            msg = "No implementation for backend '{}'".format(backend)
            raise NotImplementedError(msg)

    def _init_config_(self):
        super()._init_config_()
        key = self.__class__._pv_config_key_
        self.config[key]["color"] = "k"
        self.config[key]["line_width"] = 10
        self.config[key]["render_lines_as_tubes"] = True


class BernoulliFrame(LineMesh):
    """
    A subclass of :class:`LineMesh` to handle input and output
    of 1d meshes.

    Note
    ----
    This in experimental stage.

    """

    NDOFN = 6

    _cell_classes_ = {
        2: B2,
        3: B3,
    }
