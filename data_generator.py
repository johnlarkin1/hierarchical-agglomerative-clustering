import pandas as pd
import random
import string


NUMBER_OF_RECORDS = 200


def random_string(length=10):
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def modify_retailer_name(name):
    """Slightly modify the retailer name to simulate dirty data."""
    operations = [
        lambda s: s
        + random_string(random.randint(1, 4)),  # Add random letters at the end
        lambda s: random.choice(string.ascii_lowercase) * 2
        + s,  # Add a letter at the beginning
        lambda s: "aa".join(
            random.choice([c.upper(), c.lower()]) for c in s
        ),  # Mixed capitalization
        lambda s: s + str(random.randint(10, 99)),  # Add numbers at the end
        lambda s: "".join(
            c + random.choice(["x", "", " "]) for c in s
        ),  # weird spacing
    ]
    # Apply a random modification operation to the name
    modified_name = random.choice(operations)(name)
    return modified_name


# Sample data
retailer_names = [
    "Best Buy",
    "Target",
    "Walmart",
    "Costco",
    "Home Depot",
    "Trader Joes",
    "Petco",
    "Whole Foods",
    "Walgreens",
    "CVS",
]
counties = ["County" + str(i) for i in range(1, 151)]
data = []

for i in range(NUMBER_OF_RECORDS):  # Generating 10 records
    retailer_id = random.randint(100, 999)
    retailer_name = random.choice(retailer_names)
    modified_retailer_name = modify_retailer_name(retailer_name)
    data.append(
        [
            i + 1,  # store_record_id
            retailer_id,  # retailer_id
            retailer_name,  # retailer_nm
            random.randint(1, 100),  # store_id
            random_string(15),  # store_address_1
            random_string(15),  # store_address_2
            random_string(10),  # store_city
            random.choice(["NY", "CA", "TX", "FL", "IL", "OH", "PA"]),  # store_state
            "".join(random.choices(string.digits, k=5)),  # store_zip_code
            random.choice(counties),
            random.randint(10000, 99999),  # vpid
            modified_retailer_name,  # retailer_nm_modified
        ]
    )

# Convert to DataFrame
df = pd.DataFrame(
    data,
    columns=[
        "store_record_id",
        "retailer_id",
        "retailer_nm",
        "store_id",
        "store_address_1",
        "store_address_2",
        "store_city",
        "store_state",
        "store_zip_code",
        "store_county",
        "vpid",
        "retailer_nm_modified",
    ],
)
df.to_csv("data.csv", index=False)
