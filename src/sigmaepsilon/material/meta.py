from abc import abstractmethod


class MaterialModel:
    @abstractmethod
    def elastic_stiffness_matrix(self):
        raise NotImplementedError
