import pandas as pd

from src.universe.base import Universe
from src.corruptors.base import Corruptor

def generate_corrupted_dataset(
        n_customers: int = 20, 
        years: int = 3
    ) -> pd.DataFrame:
    universe = Universe(n_customers=n_customers, rounds_per_cycle=365)
    corruptor = Corruptor()
    output = universe.generate_orders(n_cycles=years)
    output = corruptor.process(output)
    return output