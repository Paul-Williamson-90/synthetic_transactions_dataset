from typing import Optional
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

    def sample(self) -> dict:
        return {
            "service_id": self.service_id,
            "price": self.price,
            "quantity": self.sample_quantity(),
            "variant": self.variant_distribution.sample() if self.variant_distribution else None
        }