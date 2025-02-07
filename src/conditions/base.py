from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from pydantic import BaseModel


class Condition(BaseModel, ABC):
    condition_id: int
    likelihood: float
    mode: Literal["before", "after"]

    def is_active(self) -> bool:
        return np.random.choice([True, False], p=[self.likelihood, 1 - self.likelihood])
    
    @abstractmethod
    def activate(self):
        pass