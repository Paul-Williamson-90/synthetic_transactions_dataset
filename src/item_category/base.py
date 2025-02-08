import warnings
import numpy as np
from typing import Optional
from pydantic import BaseModel, model_validator

from src.items.base import Item, create_item
from src.sampling.distributions import Distribution
from src.conditions.base import Condition, StaticValueCondition, MultipleValuesCondition


class ItemCategoryInclusionCondition(Condition):
    item_category: "ItemCategory"
    no_conditions: bool = False
    
    def activate(self) -> int:
        return self.item_category.sample_items(no_conditions=self.no_conditions)


class ItemCategory(BaseModel):
    item_category_id: int
    likelihood: float
    quantity_distribution: Distribution
    items: list[Item]
    probability_condition: Optional[Condition] = None
    price_condition: Optional[Condition] = None
    joint_item_category_condition: Optional[ItemCategoryInclusionCondition] = None

    @model_validator(mode="before")
    @classmethod
    def quantity_distribution_check(cls, v) -> dict:
        quantity_distribution: Distribution = v["quantity_distribution"]
        if quantity_distribution.upper_bound > len(v["items"]):
            raise ValueError("Quantity distribution upper bound must be less than or equal to the number of items")
        return v

    def sample_items(self, no_conditions: bool = False) -> dict:
        items: list[Item] = []
        active_conditions: list[int] = []
        likelihood = self.likelihood

        if self.probability_condition and self.probability_condition.is_active() and not no_conditions:
            likelihood = self.probability_condition.likelihood
            active_conditions.append(self.probability_condition.condition_id)

        if np.random.choice([True, False], p=[likelihood, 1 - likelihood]):
            items: list[Item] = np.random.choice(
                self.items, 
                size=self.quantity_distribution.sample(), 
                replace=False
            )

        multiplier = 1
        if self.price_condition and self.price_condition.is_active() and not no_conditions:
            multiplier = self.price_condition.activate()

        additional_items = []
        if self.joint_item_category_condition and self.joint_item_category_condition.is_active() and not no_conditions:
            active_conditions.append(self.joint_item_category_condition.condition_id)
            additional_items = self.joint_item_category_condition.activate()

        return {
            "item_category_id": self.item_category_id,
            "items": [item.sample(multiplier) for item in items] + additional_items,
            "conditions": active_conditions
        }
    
    def modify_prices(self, factor: float):
        for item in self.items:
            item.price_modification(factor)
    

class ItemCategorySelectionPool:
    
    def __init__(
            self,
            item_category_id: int,
            likelihood_upper_bound: float,
            likelihood_lower_bound: float,
            category_quantity_distribution: Distribution,
            n_services: int,
            lower_price_bound: float = np.random.normal(50, 10),
            upper_price_bound: float = np.random.normal(50, 10) + np.random.normal(50, 10),
            price_distribution_type: str = "uniform",
            int_or_float_price: str = np.random.choice(["int", "float"]),
            likelihood_range: tuple[float, float] = (0, 1),
            quantity_distribution_types: list[str] = ["uniform", "normal", "longtail"],
            quantity_upper_bound: tuple[int, int] = (1, 20),
            quantity_int_or_float: str = "int",
            variant_distribution_type: str = "uniform",
            variant_upper_bound: int = 5,
            variant_lower_bound: int = 1,
        ):
        """
        Instantiate an ItemCategorySelectionPool object.

        Args:
            item_category_id (int): The unique identifier for the item category.
            likelihood_upper_bound (float): The upper bound of the likelihood of selecting the category in an order.
            likelihood_lower_bound (float): The lower bound of the likelihood of selecting the category in an order.
            category_quantity_distribution (Distribution): The distribution of the number of items to be selected from the category.
            n_services (int): The number of services to be created in the category.
            lower_price_bound (float, optional): The lower bound of the price of the items in the category. Defaults to np.random.normal(50, 10).
            upper_price_bound (float, optional): The upper bound of the price of the items in the category. Defaults to np.random.normal(50, 10) + np.random.normal(50, 10).
            price_distribution_type (str, optional): The distribution of the prices of the items in the category. Defaults to "uniform".
            int_or_float_price (str, optional): The type of the price values. Defaults to np.random.choice(["int", "float"]).
            likelihood_range (tuple[float, float], optional): The range of likelihood values for items. Defaults to (0, 1).
            quantity_distribution_types (list[str], optional): The types of quantity distributions. Defaults to ["uniform", "normal", "longtail"].
            quantity_upper_bound (tuple[int, int], optional): The upper bound of the quantity distribution. Defaults to (1, 20).
            quantity_int_or_float (str, optional): The type of the quantity values. Defaults to "int".
            variant_distribution_type (str, optional): The distribution of the number of variants per item. Defaults to "uniform".
            variant_upper_bound (int, optional): The upper bound of the number of variants per item. Defaults to 5.
            variant_lower_bound (int, optional): The lower bound of the number of variants per item. Defaults to 1.
        """
        self.item_category_id = item_category_id
        self.likelihood_upper_bound = likelihood_upper_bound
        self.likelihood_lower_bound = likelihood_lower_bound
        self.category_quantity_distribution = category_quantity_distribution
        
        self.probability_condition = None
        if np.random.choice([True, False]):
            self.probability_condition = np.random.choice([
                StaticValueCondition(
                    condition_id=f"{item_category_id}_probability",
                    likelihood=np.random.uniform(0, 1),
                    value=np.random.uniform(0, 1)
                )
            ])

        self.price_condition = None
        if np.random.choice([True, False]):
            self.price_condition = np.random.choice([
                StaticValueCondition(
                    condition_id=f"{item_category_id}_price",
                    likelihood=np.random.uniform(0, 1),
                    value=np.random.uniform(0, 2)
                ),
                MultipleValuesCondition(
                    condition_id=f"{item_category_id}_price",
                    likelihood=np.random.uniform(0, 1),
                    values=np.random.uniform(0, 2, np.random.randint(1, 5)).tolist()
                )
            ])

        self.items = [
            create_item(
                f"{self.item_category_id}_{service_id}",
                lower_price_bound,
                upper_price_bound,
                price_distribution_type,
                int_or_float_price,
                likelihood_range,
                quantity_distribution_types,
                quantity_upper_bound,
                quantity_int_or_float,
                variant_distribution_type,
                variant_upper_bound,
                variant_lower_bound
            ) for service_id in range(1, n_services + 1)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def sample_items(self, n_samples: int) -> list[Item]:
        if n_samples > len(self):
            warnings.warn(f"Number of samples requested is greater than the number of items in the category. Returning all items.")
        n_samples = min(n_samples, len(self))
        likelihood_mean = (self.likelihood_upper_bound + self.likelihood_lower_bound) / 2
        likelihood_std_dev = (self.likelihood_upper_bound - self.likelihood_lower_bound) / 4
        return ItemCategory(
            item_category_id=self.item_category_id,
            likelihood=np.clip(np.random.normal(likelihood_mean, likelihood_std_dev), 0, 1),
            quantity_distribution=self.category_quantity_distribution,
            probability_condition=self.probability_condition,
            price_condition=self.price_condition,
            items=np.random.choice(self.items, n_samples).tolist()
        )
