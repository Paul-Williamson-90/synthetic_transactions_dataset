import numpy as np
from typing import Optional
from pydantic import BaseModel, model_validator

from src.items.base import Item
from src.sampling.distributions import Distribution
from src.conditions.base import Condition


class ItemCategory(BaseModel):
    service_category_id: int
    likelihood: float
    quantity_distribution: Distribution
    items: list[Item]
    probability_condition: Optional[Condition] = None
    price_condition: Optional[Condition] = None

    @model_validator(mode="before")
    @classmethod
    def quantity_distribution_check(cls, v) -> dict:
        quantity_distribution: Distribution = v["quantity_distribution"]
        if quantity_distribution.upper_bound > len(v["items"]):
            raise ValueError("Quantity distribution upper bound must be less than or equal to the number of items")
        return v

    def sample_items(self) -> dict:
        items: list[Item] = []
        active_conditions: list[int] = []
        likelihood = self.likelihood

        if self.probability_condition.is_active():
            likelihood = self.probability_condition.likelihood
            active_conditions.append(self.probability_condition.condition_id)

        if np.random.choice([True, False], p=[likelihood, 1 - likelihood]):
            items: list[Item] = np.random.choice(
                self.items, 
                size=self.quantity_distribution.sample(), 
                replace=False
            )

        multiplier = 1
        if self.price_condition.is_active():
            multiplier = self.price_condition.activate()

        return {
            "items": [item.sample(multiplier) for item in items],
            "conditions": active_conditions
        }