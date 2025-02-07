import numpy as np

from src.item_category.base import ItemCategory


class OrderProfile:
    def __init__(
            self, 
            customer_id: int, 
            item_categories: list[ItemCategory],
        ):
        self.customer_id = customer_id
        self.item_categories = item_categories
        for item_category in self.item_categories:
            item_category.modify_prices(1 + (((np.random.rand() * 2) - 1) / 2))

    def sample(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "item_categories": [item_category.sample_items() for item_category in self.item_categories]
        }
    
    def modify_all_prices(self, factor: float):
        for item_category in self.item_categories:
            item_category.modify_prices(factor)

    def modify_prices(self, item_category_id: int, factor: float):
        for item_category in self.item_categories:
            if item_category.item_category_id == item_category_id:
                item_category.modify_prices(factor)
                break

    def modify_prices_random(self, factor: float):
        item_category: ItemCategory = np.random.choice(self.item_categories)
        item_category.modify_prices(factor)