import pandas as pd
import numpy as np

from src.item_category.base import ItemCategorySelectionPool
from src.order_profile.base import OrderProfile
from src.sampling.distributions import Distribution


DEFAULT_SELECTION_POOLS = {
    "turn_rates": ItemCategorySelectionPool(
        item_category_id=1,
        likelihood_lower_bound=0.9,
        likelihood_upper_bound=1,
        category_quantity_distribution=Distribution(
            distribution_type="normal",
            lower_bound=1,
            upper_bound=1,
            int_or_float="int"
        ),
        n_services=10,
        lower_price_bound=1000,
        upper_price_bound=5000,
        price_distribution_type="longtail",
        int_or_float_price="float",
        likelihood_range=(1, 1),
        quantity_distribution_types=["normal"],
        quantity_upper_bound=(1, 2),
        quantity_int_or_float="int",
        variant_distribution_type=None,
        variant_upper_bound=None,
        variant_lower_bound=None,
    ),
    "labour_rates": ItemCategorySelectionPool(
        item_category_id=2,
        likelihood_lower_bound=0.2,
        likelihood_upper_bound=0.5,
        category_quantity_distribution=Distribution(
            distribution_type="normal",
            lower_bound=0,
            upper_bound=3,
            int_or_float="int"
        ),
        n_services=15,
        lower_price_bound=30,
        upper_price_bound=200,
        price_distribution_type="longtail",
        int_or_float_price="float",
        likelihood_range=(0.1, 0.6),
        quantity_distribution_types=["longtail"],
        quantity_upper_bound=(1, 5),
        quantity_int_or_float="int",
        variant_distribution_type="longtail",
        variant_upper_bound=1,
        variant_lower_bound=0,
    ),
    "plane_services": ItemCategorySelectionPool(
        item_category_id=3,
        likelihood_lower_bound=0.2,
        likelihood_upper_bound=0.5,
        category_quantity_distribution=Distribution(
            distribution_type="normal",
            lower_bound=0,
            upper_bound=2,
            int_or_float="int"
        ),
        n_services=15,
        lower_price_bound=200,
        upper_price_bound=500,
        price_distribution_type="longtail",
        int_or_float_price="float",
        likelihood_range=(0.1, 1),
        quantity_distribution_types=["longtail"],
        quantity_upper_bound=(1, 2),
        quantity_int_or_float="int",
        variant_distribution_type=None,
        variant_upper_bound=None,
        variant_lower_bound=None,
    ),
    "additional_services": ItemCategorySelectionPool(
        item_category_id=3,
        likelihood_lower_bound=0.4,
        likelihood_upper_bound=0.7,
        category_quantity_distribution=Distribution(
            distribution_type="normal",
            lower_bound=0,
            upper_bound=2,
            int_or_float="int"
        ),
        n_services=15,
        lower_price_bound=100,
        upper_price_bound=500,
        price_distribution_type="longtail",
        int_or_float_price="float",
        likelihood_range=(0.1, 1),
        quantity_distribution_types=["longtail"],
        quantity_upper_bound=(1, 2),
        quantity_int_or_float="int",
        variant_distribution_type=None,
        variant_upper_bound=None,
        variant_lower_bound=None,
    ),
    "other": ItemCategorySelectionPool(
        item_category_id=4,
        likelihood_lower_bound=0.01,
        likelihood_upper_bound=0.1,
        category_quantity_distribution=Distribution(
            distribution_type="normal",
            lower_bound=0,
            upper_bound=2,
            int_or_float="int"
        ),
        n_services=15,
        lower_price_bound=1,
        upper_price_bound=500,
        price_distribution_type="uniform",
        int_or_float_price="float",
        likelihood_range=(0.1, 1),
        quantity_distribution_types=["longtail"],
        quantity_upper_bound=(1, 2),
        quantity_int_or_float="int",
        variant_distribution_type=None,
        variant_upper_bound=None,
        variant_lower_bound=None,
    ),
}

class Universe:

    def __init__(
            self,
            n_customers: int = 50,
            item_category_selection_pools: dict[str, ItemCategorySelectionPool] = DEFAULT_SELECTION_POOLS,
            n_item_sample_bounds: tuple[int, int] = (3, 5),
    ):
        self.n_item_sample_bounds = n_item_sample_bounds
        self.item_category_selection_pools = item_category_selection_pools
        self.profiles = [
            OrderProfile(
                customer_id=customer_id,
                item_categories=[
                    item_category_selection_pools[key]
                    .sample_items(n_samples=np.random.randint(n_item_sample_bounds[0], n_item_sample_bounds[1])) 
                    for key in item_category_selection_pools.keys()
                ],
            ) for customer_id in range(1, n_customers + 1)
        ]
        self._cycle = 0

    def add_customer(self):
        self.profiles.append(
            OrderProfile(
                customer_id=len(self.profiles) + 1,
                item_categories=[
                    self.item_category_selection_pools[key]
                    .sample_items(n_samples=np.random.randint(self.n_item_sample_bounds[0], self.n_item_sample_bounds[1]))
                    for key in self.item_category_selection_pools.keys()
                ],
            )
        )

    def generate_orders(
            self, 
            n_cycles: int = 10, 
            rounds_per_cycle: int = 50,
            amendment_probability: float = 0.8,
            ammendment_scale: float = 10.0,
            new_customer_probability: float = 0.05,
        ) -> pd.DataFrame:
        orders: list[pd.DataFrame] = []

        start_cycle = self._cycle
        for cycle in range(start_cycle, start_cycle + n_cycles + 1):
            for r in range(1, rounds_per_cycle + 1):
                for profile in self.profiles:
                    increased = False
                    if profile.increase_viable():
                        if np.random.choice([True, False], p=[amendment_probability, 1 - amendment_probability]):
                            profile.modify_prices_random(
                                factor=round(((np.random.rand() * 2) - 1) / ammendment_scale, 2),
                                n=np.random.randint(1, 5)
                            )
                            increased = True

                    order = profile.sample()
                    order["contract_ammendment"] = increased
                    order["round"] = f"{cycle + start_cycle}_{r}"
                    orders.append(order)
            self._cycle += 1
            if np.random.choice([True, False], p=[new_customer_probability, 1 - new_customer_probability]):
                self.add_customer()

        orders_df = pd.concat(orders, ignore_index=True)
        return orders_df