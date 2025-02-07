from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional, Union
import numpy as np


class Distribution(BaseModel):
    distribution_type: Literal["normal", "longtail", "uniform"]
    lower_bound: int = Field(..., ge=0)
    upper_bound: int = Field(..., ge=0)
    mean: Optional[float] = None  # Used for normal/longtail
    std_dev: Optional[float] = None  # Used for normal/longtail
    int_or_float: Literal["int", "float"] = "int"

    @model_validator(mode="after")
    def set_mean_std(self) -> "Distribution":
        # Auto-calculate mean and std_dev if not provided
        if self.mean is None or self.std_dev is None:
            if self.distribution_type == "normal":
                self.mean = (self.lower_bound + self.upper_bound) / 2
                self.std_dev = (self.upper_bound - self.lower_bound) / 4  # 95% within bounds
            elif self.distribution_type == "longtail":
                self.mean = np.log((self.lower_bound + self.upper_bound) / 2)
                self.std_dev = np.log(1 + (self.upper_bound - self.lower_bound) / 4)
            elif self.distribution_type == "uniform":
                self.mean = None  # Uniform doesn't use mean/std_dev
                self.std_dev = None
            else:
                raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
        return self

    def sample(self) -> Union[int, float]:
        while True:
            if self.distribution_type == "normal":
                value = np.random.normal(self.mean, self.std_dev)
            elif self.distribution_type == "longtail":
                value = np.random.lognormal(self.mean, self.std_dev)
            elif self.distribution_type == "uniform":
                value = np.random.uniform(self.lower_bound, self.upper_bound)
            else:
                raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
            
            if self.lower_bound <= value <= self.upper_bound:
                break  # Accept only values within bounds
        if self.int_or_float == "int":
            value = int(value)
        return value
    
    @model_validator(mode="before")
    @classmethod
    def check_bounds(cls, v) -> dict:
        if v["lower_bound"] >= v["upper_bound"]:
            raise ValueError(f"Lower bound must be less than upper bound")
        return v