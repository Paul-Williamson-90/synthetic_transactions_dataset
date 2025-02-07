from pydantic import BaseModel

from src.sampling.distributions import Distribution


class BaseItem(BaseModel):

    service_type_id: int
    price: float
    likelihood: float
    quantity_distribution: Distribution

    def sample_quantity(self):
        return self.quantity_distribution.sample()
    
    def price_modification(self, factor: float):
        self.price *= factor