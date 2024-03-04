import pandas as pd
import random
import string

# Input
SHOULD_GENERATE_EASY = False


if SHOULD_GENERATE_EASY:
    NUMBER_OF_RECORDS = 100
    DEST_FILE = "data_easy.csv"
else:
    NUMBER_OF_RECORDS = 400
    DEST_FILE = "data_complex.csv"


def random_string(length=10):
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def typo_generator(s):
    """Introduce a typographical error into the string."""
    pos = random.randint(0, len(s) - 1)
    return s[:pos] + random.choice(string.ascii_lowercase) + s[pos + 1 :]


def abbreviation(s):
    """Shorten the name as an abbreviation."""
    parts = s.split()
    if len(parts) > 1:
        return "".join(part[0].upper() for part in parts)
    else:
        return s[: len(s) // 2]


def modify_retailer_name(name):
    """Slightly modify the retailer name to simulate dirty data."""
    operations = [
        lambda s: s.replace(" ", ""),  # Remove spaces
        lambda s: s.lower(),  # Convert to lowercase
        lambda s: s.upper(),  # Convert to uppercase
        lambda s: " ".join(s.split()[: len(s.split()) // 2]),  # Keep first part
        lambda s: "".join(random.choice([c.upper(), c.lower()]) for c in s),  # Mixed capitalization
        lambda s: s + str(random.randint(10, 99)),  # Add numbers at the end
    ]
    if not SHOULD_GENERATE_EASY:
        operations.extend(
            [
                lambda s: s
                + random.choice([" Inc", " LLC", " Corp", " Co", " Ltd", " Corp."]),  # Add suffix
                lambda s: "".join(
                    c + random.choice(["-", "", " "]) for c in s
                ),  # Extra or missing characters
                lambda s: s + random_string(random.randint(1, 4)),  # Add random letters at the end
                lambda s: typo_generator(s),  # Introduce a typographical error
                lambda s: abbreviation(s),  # Use abbreviation
            ]
        )
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
    "Whole Foods",
    "Walgreens",
    "CVS",
    "Petco",
    "DataDog",
    "Farmer's Dog",
    "Chewy",
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
df.to_csv(DEST_FILE, index=False)
