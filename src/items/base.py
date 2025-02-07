from typing import Optional
import numpy as np
from pydantic import BaseModel, model_validator

from src.sampling.distributions import Distribution


class Item(BaseModel):

    service_id: int
    price: float
    likelihood: float
    quantity_distribution: Distribution
    variant_distribution: Optional[Distribution] = None
    rounded: bool = True

    @model_validator(mode="after")
    def check_price(self) -> "Item":
        if self.rounded:
            self.price = round(self.price, 2)
        return self

    def sample_quantity(self) -> int:
        return self.quantity_distribution.sample()
    
    def price_modification(self, factor: float):
        self.price *= factor
        if self.rounded:
            self.price = round(self.price, 2)

    def sample(self, multiplier: float = 1.0) -> dict:
        return {
            "service_id": self.service_id,
            "price": self.price * multiplier,
            "quantity": self.sample_quantity(),
            "variant": self.variant_distribution.sample() if self.variant_distribution else None
        }
    

def create_item(
        service_id: int,
        lower_price_bound: float = np.random.normal(50, 10),
        upper_price_bound: float = np.random.normal(50, 10) + np.random.normal(50, 10),
        price_distribution_type: str = "uniform",
        int_or_float_price: str = np.random.choice(["int", "float"]),
        likelihood_range: tuple[float, float] = (0, 1),
        quantity_distribution_types: list[str] = ["uniform", "normal", "longtail"],
        quantity_upper_bound: tuple[int, int] = (1, 20),
        quantity_int_or_float: str = "int",
        variant_distribution_type: Optional[str] = None,
        variant_upper_bound: Optional[int] = None,
        variant_lower_bound: Optional[int] = None,
    ) -> Item:

    if not (variant_distribution_type or variant_upper_bound or variant_lower_bound):
        variant_distribution = None
    else:
        variant_distribution = Distribution(
            distribution_type=variant_distribution_type,
            lower_bound=variant_lower_bound,
            upper_bound=variant_upper_bound,
            int_or_float="int"
        )
    
    price = Distribution(
        distribution_type=price_distribution_type,
        lower_bound=lower_price_bound,
        upper_bound=upper_price_bound,
        int_or_float=int_or_float_price,
    ).sample()
    
    return Item(
        service_id=service_id,
        price=price,
        likelihood=np.random.uniform(likelihood_range[0], likelihood_range[1]),
        quantity_distribution=Distribution(
            distribution_type=np.random.choice(quantity_distribution_types),
            lower_bound=1,
            upper_bound=np.random.randint(quantity_upper_bound[0], quantity_upper_bound[1]),
            int_or_float=quantity_int_or_float
        ),
        variant_distribution=variant_distribution
    )