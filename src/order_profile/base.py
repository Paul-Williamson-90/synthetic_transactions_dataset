import numpy as np
import pandas as pd

from src.item_category.base import ItemCategory


class OrderProfile:
    def __init__(
            self, 
            customer_id: int, 
            item_categories: list[ItemCategory],
            increase_every: int = 50
        ):
        self.customer_id = customer_id
        self.item_categories = item_categories
        for item_category in self.item_categories:
            item_category.modify_prices(1 + (((np.random.rand() * 2) - 1) / 2))
        self.last_price_increase = 0
        self.increase_every = increase_every
        self.new_customer = True
        self.order_frequency = np.random.randint(1, 5)

    def increase_viable(self) -> bool:
        if self.last_price_increase >= self.increase_every:
            return True
        else:
            self.last_price_increase += 1
            return False
    
    def reset_increase(self):
        self.last_price_increase = 0

    def _convert_to_order_df(self, order: dict) -> pd.DataFrame:
        customer_id = order["customer_id"]
        item_categories = order["item_categories"]

        order_df = pd.DataFrame()
        for item_category in item_categories:
            item_category_id = item_category["item_category_id"]
            item_category_conditions = item_category["conditions"]
            item_category_items = pd.DataFrame(item_category["items"])
            item_category_items["item_category_id"] = item_category_id
            item_category_items["conditions"] = [item_category_conditions for _ in range(len(item_category_items))]
            item_category_items["customer_id"] = customer_id
            item_category_items["new_customer"] = self.new_customer
            order_df = pd.concat([order_df, item_category_items])

        return order_df

    def sample(self) -> pd.DataFrame:
        item_categories = [item_category.sample_items() for item_category in self.item_categories]
        item_categories = [x for x in item_categories if x["items"]]
        output = self._convert_to_order_df({
            "customer_id": self.customer_id,
            "item_categories": item_categories
        })
        self.new_customer = False
        return output
    
    def modify_all_prices(self, factor: float):
        for item_category in self.item_categories:
            item_category.modify_prices(factor)

    def modify_prices(self, item_category_id: int, factor: float):
        for item_category in self.item_categories:
            if item_category.item_category_id == item_category_id:
                item_category.modify_prices(factor)
                break

    def modify_prices_random(self, factor: float, n: int):
        item_category: list[ItemCategory] = np.random.choice(self.item_categories, min(n, len(self.item_categories)), replace=False)
        for item in item_category:
            item.modify_prices(factor)