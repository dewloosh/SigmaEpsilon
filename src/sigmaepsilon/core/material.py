from abc import abstractmethod


class MaterialLike:
    @abstractmethod
    def elastic_stiffness_matrix(self):
        raise NotImplementedError


class SectionLike(MaterialLike):
    ...
