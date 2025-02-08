# Synthetic Transaction Dataset Generator

## Summary
- This synthetic dataset generator is intended for designing methodologies for anomaly detection algorithms in transactional data.
- The data is inspired from a client project I have worked on, seeking to identify revenue leakage in the client's financial process.

## Setup
Create a virtual env and install the dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## About the dataset
- date: the date the service/product took place
- customer_id: unique identifier for a customer
- order_number: the order number the service is attributed to
- item_category_id: high level product category
- service_id: low level product category
- variant: variant of the product
- price: single quantity rate
- quantity: how many were purchased
- final_price: the final price of the product considering quantity and any active conditions that impact price
- new_customer: a flag indicating if it is the first order for the customer
- contract_ammendment: a flag indicating a new contract/ammendment has been created with the customer
- condition_{n}_probability: a condition that makes the nth item_category_id probability of occurrence change
- condition_{n}_price: a condition that makes the price of nth item_category_id products change
- flag_random_price_change: flag indicating the price of the product is incorrect (target label)
- flag_condition_not_implemented: flag indicating a condition is active but not implemented (target label)
- flag_price_change_no_ammendment: flag indicating prices for the customer have changed without a new contract/ammendment (delayed) (target label)
- flag_missing_charges: flag indicating there is a missing item at an order level (target label)
- flag_discrepancy: flag indicating one or more of the previous flags is active (target label)

## Suggestions for usage
- Each of the flag_{name} features can be used for training and/or evaluating an anomaly detection algorithm (dependent on supervised vs unsupversied model approach). 
- It is unlikely a single algorithm would work for all flags, and you will likely need to transform the data into various perspectives in order to focus in on some of the flags.