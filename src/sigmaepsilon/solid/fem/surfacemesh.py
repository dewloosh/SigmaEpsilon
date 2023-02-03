from polymesh.config import __hasplotly__, __hasmatplotlib__

from .mesh import FemMesh

from .cells import Q4_M, Q9_M


class SurfaceMesh(FemMesh):
    """
    A data class dedicated to 2d cells. It handles sections and other line
    related information, presets for plotting, etc.

    """

    _cell_classes_ = {
        4: Q4_M,
        9: Q9_M,
    }

    def __init__(self, *args, section=None, **kwargs):
        if section is not None:
            pass
        self._section = section
        super().__init__(*args, **kwargs)

    @property
    def section(self):
        if self._section is not None:
            return self._section
        else:
            if self.is_root():
                return self._section
            else:
                return self.parent.section
