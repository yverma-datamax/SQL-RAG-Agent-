"""
seed_db.py — Creates and populates the sample e-commerce SQLite database.
Run once before opening the notebook: python src/seed_db.py
"""

import sqlite3
import random
from datetime import datetime, timedelta

DB_PATH = "../data/ecommerce.db"

PRODUCTS = [
    ("Wireless Headphones", "Electronics", 89.99, 120),
    ("Running Shoes", "Apparel", 64.99, 200),
    ("Coffee Maker", "Kitchen", 49.99, 85),
    ("Yoga Mat", "Sports", 29.99, 310),
    ("Mechanical Keyboard", "Electronics", 119.99, 60),
    ("Water Bottle", "Sports", 19.99, 400),
    ("Desk Lamp", "Home", 34.99, 150),
    ("Notebook Set", "Stationery", 12.99, 500),
    ("Bluetooth Speaker", "Electronics", 59.99, 95),
    ("Protein Powder", "Health", 44.99, 180),
]

CUSTOMERS = [
    ("Alice Martin", "alice@email.com", "West"),
    ("Bob Chen", "bob@email.com", "East"),
    ("Clara Diaz", "clara@email.com", "South"),
    ("David Kim", "david@email.com", "North"),
    ("Eva Torres", "eva@email.com", "West"),
    ("Frank Liu", "frank@email.com", "East"),
    ("Grace Park", "grace@email.com", "South"),
    ("Hiro Tanaka", "hiro@email.com", "North"),
    ("Isla Brown", "isla@email.com", "West"),
    ("Jake Wilson", "jake@email.com", "East"),
]

STATUSES = ["completed", "completed", "completed", "pending", "refunded"]


def seed():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS products;

        CREATE TABLE products (
            product_id   INTEGER PRIMARY KEY,
            name         TEXT,
            category     TEXT,
            price        REAL,
            stock        INTEGER
        );

        CREATE TABLE customers (
            customer_id  INTEGER PRIMARY KEY,
            name         TEXT,
            email        TEXT,
            region       TEXT,
            joined_date  TEXT
        );

        CREATE TABLE orders (
            order_id     INTEGER PRIMARY KEY,
            customer_id  INTEGER REFERENCES customers(customer_id),
            product_id   INTEGER REFERENCES products(product_id),
            quantity     INTEGER,
            total_amount REAL,
            order_date   TEXT,
            status       TEXT
        );
    """)

    cur.executemany(
        "INSERT INTO products VALUES (NULL,?,?,?,?)",
        PRODUCTS
    )

    base_date = datetime(2023, 1, 1)
    for i, (name, email, region) in enumerate(CUSTOMERS):
        joined = (base_date + timedelta(days=random.randint(0, 60))).strftime("%Y-%m-%d")
        cur.execute("INSERT INTO customers VALUES (NULL,?,?,?,?)", (name, email, region, joined))

    random.seed(42)
    for _ in range(200):
        cid = random.randint(1, len(CUSTOMERS))
        pid = random.randint(1, len(PRODUCTS))
        qty = random.randint(1, 4)
        price = PRODUCTS[pid - 1][2]
        total = round(price * qty, 2)
        days_offset = random.randint(0, 364)
        order_date = (base_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
        status = random.choice(STATUSES)
        cur.execute(
            "INSERT INTO orders VALUES (NULL,?,?,?,?,?,?)",
            (cid, pid, qty, total, order_date, status)
        )

    conn.commit()
    conn.close()
    print(f"✅ Database seeded at {DB_PATH}")
    print(f"   → {len(PRODUCTS)} products, {len(CUSTOMERS)} customers, 200 orders")


if __name__ == "__main__":
    seed()
