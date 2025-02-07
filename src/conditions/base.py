from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel


class Condition(BaseModel, ABC):
    condition_id: str
    likelihood: float

    def is_active(self) -> bool:
        return np.random.choice([True, False], p=[self.likelihood, 1 - self.likelihood])
    
    @abstractmethod
    def activate(self):
        pass


class StaticValueCondition(Condition):
    value: float

    def activate(self) -> float:
        return self.value
    

class MultipleValuesCondition(Condition):
    values: list[float]
    
    def activate(self) -> float:
        return np.random.choice(self.values)