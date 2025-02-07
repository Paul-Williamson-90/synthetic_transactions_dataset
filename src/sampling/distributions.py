from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional
import numpy as np


class Distribution(BaseModel):
    distribution_type: Literal["normal", "longtail", "uniform"]
    lower_bound: int = Field(..., ge=0)
    upper_bound: int = Field(..., ge=0)
    mean: Optional[float] = None  # Used for normal/longtail
    std_dev: Optional[float] = None  # Used for normal/longtail
    int_or_float: Literal["int", "float"] = "int"

    def __init__(self, **data):
        super().__init__(**data)
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

    def sample(self):
        if self.distribution_type == "normal":
            samples = np.random.normal(self.mean, self.std_dev, 1)
        elif self.distribution_type == "longtail":
            samples = np.random.lognormal(self.mean, self.std_dev, 1)
        elif self.distribution_type == "uniform":
            samples = np.random.uniform(self.lower_bound, self.upper_bound, 1)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
        
        samples = np.clip(samples, self.lower_bound, self.upper_bound)  # Enforce bounds
        if self.int_or_float == "int":
            samples = int(samples)
        return samples[0]
    
    @model_validator(mode="before")
    @classmethod
    def check_bounds(cls, v):
        if v["lower_bound"] >= v["upper_bound"]:
            raise ValueError(f"Lower bound must be less than upper bound")
        return v