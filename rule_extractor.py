import re


messages = [
    "Please check order ORD-58329 for me",
    "My order ORD-92104 has not arrived",
    "I need help with ORD-77120",
    "There is no order number in this message",
]

for message in messages:
    order_ids = re.findall(r"ORD-\d+", message)

    print("\nMessage:", message)
    print("Order IDs:", order_ids)