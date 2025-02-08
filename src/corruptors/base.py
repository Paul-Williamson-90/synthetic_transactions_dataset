import numpy as np
import pandas as pd

from src.constants import FIXED_COLS


class Corruptor:

    _cols: list[str] = FIXED_COLS

    def __init__(
            self,
            random_price_change_prob: float = 0.03,
            condition_not_implemented_prob: float = 0.03,
            cust_probability: float = 0.5,
            days_shift: int = 20,
            max_occurrences: int = 3,
            missing_charges_prob: float = 0.01
    ):
        self.random_price_change_prob = random_price_change_prob
        self.condition_not_implemented_prob = condition_not_implemented_prob
        self.cust_probability = cust_probability
        self.days_shift = days_shift
        self.max_occurrences = max_occurrences
        self.missing_charges_prob = missing_charges_prob

    def _incorrect_price(self, df: pd.DataFrame) -> pd.DataFrame:
        active_rows = np.random.choice([True, False], size=df.shape[0], p=[self.random_price_change_prob, 1-self.random_price_change_prob])
        prices = [price * (1 + np.random.rand()) if active else price for price, active in zip(df["price"], active_rows)]
        df["price"] = prices
        df["flag_random_price_change"] = active_rows
        return df
    
    def _billing_logic_error(self, df: pd.DataFrame) -> pd.DataFrame:
        condition_cols = df.filter(like="condition_").columns
        df["temp"] = df["final_price"]
        for cond in condition_cols:
            active_indexes = df[df[cond] == 1].index
            selected = np.random.choice(active_indexes, int(len(active_indexes) * self.condition_not_implemented_prob), replace=False)
            df.loc[selected, "final_price"] = df.loc[selected, "price"] * df.loc[selected, "quantity"]
        df["final_price"] = df["final_price"]
        df["flag_condition_not_implemented"] = df["temp"] != df["final_price"]
        df.drop(columns="temp", inplace=True)
        return df
    
    def _missing_information(self, df: pd.DataFrame) -> pd.DataFrame:
        contract_ammendments = (
            df[["customer_id", "date", "contract_ammendment"]]
            .drop_duplicates()
            .set_index(["customer_id", "date"])
            .sort_index()
            .unstack("customer_id")
        ).copy()
        custs = np.random.choice(contract_ammendments.columns, int(contract_ammendments.shape[1] * self.cust_probability), replace=False)

        for cust in custs:
            ammended = contract_ammendments[cust][contract_ammendments[cust]==True]
            ammended = [(k,v) for k,v in ammended.to_dict().items()]

            n_select = np.random.randint(1, min(self.max_occurrences + 1, len(ammended)))

            to_change = np.random.choice(range(len(ammended)), n_select, replace=False)

            for i in range(n_select):
                contract_ammendments.loc[ammended[to_change[i]][0], cust] = False
                date_offset = np.random.randint(1, self.days_shift)
                if ammended[to_change[i]][0] + pd.DateOffset(days=date_offset) < contract_ammendments.index[-1]:
                    contract_ammendments.loc[ammended[to_change[i]][0] + pd.DateOffset(days=date_offset), cust] = True

        contract_ammendments = contract_ammendments.stack("customer_id").reset_index(drop=False)
        df["old_contract_ammendment"] = df["contract_ammendment"]
        df = df.drop(columns=["contract_ammendment"]).merge(contract_ammendments, on=["customer_id", "date"], how="left")
        df["flag_price_change_no_ammendment"] = np.where(
            (df["old_contract_ammendment"] == True) 
            & (df["contract_ammendment"] == False), 
            True, False
        )
        df.drop(columns=["old_contract_ammendment"], inplace=True)
        return df
    
    def _missing_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = df.sample(frac=self.missing_charges_prob, replace=False)
        missing_flag = missing[["date", "customer_id"]].drop_duplicates()
        missing_flag["flag_missing_charges"] = True
        df = df.drop(missing_flag.index, axis=0, errors="ignore")
        df = df.merge(missing_flag, on=["date", "customer_id"], how="left")
        df["flag_missing_charges"] = df["flag_missing_charges"].fillna(False)
        return df
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._incorrect_price(df)
        df = self._billing_logic_error(df)
        df = self._missing_information(df)
        df = self._missing_charges(df)
        df["flag_discrepancy"] = df.filter(like="flag_").sum(axis=1).astype(bool)

        df = df[
            self._cols 
            + [x for x in df.columns if x not in self._cols and x not in df.filter(like="flag_").columns]
            + [x for x in df.filter(like="flag_").columns]
        ]
        return df