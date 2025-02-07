from pydantic import BaseModel, model_validator

from src.sampling.distributions import Distribution


class BaseItem(BaseModel):

    service_type_id: int
    price: float
    likelihood: float
    quantity_distribution: Distribution
    rounded: bool = True

    @model_validator(mode="after")
    def check_price(self) -> "BaseItem":
        if self.rounded:
            self.price = round(self.price, 2)
        return self

    def sample_quantity(self) -> int:
        return self.quantity_distribution.sample()
    
    def price_modification(self, factor: float):
        self.price *= factor
        if self.rounded:
            self.price = round(self.price, 2)