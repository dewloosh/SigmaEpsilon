from abc import abstractmethod


class MaterialLike:
    """
    Base class for materials.
    """
    @abstractmethod
    def elastic_stiffness_matrix(self):
        raise NotImplementedError


class SectionLike(MaterialLike):
    """
    Base class for beam sections.
    """
    ...
