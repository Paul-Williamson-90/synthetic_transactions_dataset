import numpy as np
from pydantic import BaseModel, model_validator

from src.items.base import Item
from src.sampling.distributions import Distribution


class ItemCategory(BaseModel):
    service_category_id: int
    likelihood: float
    quantity_distribution: Distribution
    items: list[Item]

    @model_validator(mode="before")
    @classmethod
    def quantity_distribution_check(cls, v) -> dict:
        quantity_distribution: Distribution = v["quantity_distribution"]
        if quantity_distribution.upper_bound > len(v["items"]):
            raise ValueError("Quantity distribution upper bound must be less than or equal to the number of items")
        return v

    def sample_items(self) -> list[dict]:
        items: list[Item] = np.random.choice(
            self.items, 
            size=self.quantity_distribution.sample(), 
            replace=False
        )
        return [item.sample() for item in items]
