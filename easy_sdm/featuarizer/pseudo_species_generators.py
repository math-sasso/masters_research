from abc import ABC, abstractmethod
class BasePseudoSpeciesGenerator(ABC):
    def __init__(self) -> None:
        pass
    @abstractmethod
    def generate(self):
        pass

class RandomPseudoSpeciesGenerator(BasePseudoSpeciesGenerator):
    def __init__(self) -> None:
        raise NotImplementedError("This class is not yet implemented")

    def generate(self):
        pass

class RSEPPseudoSpeciesGenerator(BasePseudoSpeciesGenerator):
    def __init__(self) -> None:
        raise NotImplementedError("This class is not yet implemented")

    def generate(self):
        pass

