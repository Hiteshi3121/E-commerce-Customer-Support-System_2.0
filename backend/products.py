# backend/products.py
"""
Product Catalog  ─  NovaCart AI
════════════════════════════════════════════════════════════
Manages the products table in the same orders.db used by the rest of the system.

Design principles:
  • Single source of truth — same SQLite DB, no extra files or services
  • Fuzzy match via LLM entity extraction + SQL LIKE/LOWER — handles typos and
    natural phrasing ("buy some flowers" → product: "Fresh Flowers Bouquet")
  • product_id is a foreign key stored in the orders table for full traceability
  • Categories: Electronics | Kitchen & Appliances | Clothing & Fashion |
                Books & Stationery | Food & Grocery

Public API:
  init_products_db()           → creates table + seeds catalog (idempotent)
  search_product(name: str)    → returns best-match Product dict or None
  get_product_by_id(pid: str)  → direct lookup by product_id
  format_product_card(product) → returns emoji-rich display string for chat
"""

from __future__ import annotations
import difflib
from backend.db import get_connection


# ─────────────────────────────────────────────────────────────────────
# CATALOG  —  5 categories × ~8–10 products each
# Fields: product_id, name, category, price (INR), stock, description
# ─────────────────────────────────────────────────────────────────────

PRODUCT_CATALOG: list[dict] = [

    # ── ELECTRONICS ──────────────────────────────────────────────
    {"product_id": "ELEC-001", "name": "Wireless Bluetooth Headphones",
     "category": "Electronics", "price": 2499,  "stock": 150,
     "description": "Over-ear noise-cancelling headphones with 30hr battery life"},

    {"product_id": "ELEC-002", "name": "Smart LED TV 43 Inch",
     "category": "Electronics", "price": 28999, "stock": 40,
     "description": "4K UHD Smart TV with built-in Wi-Fi and Android OS"},

    {"product_id": "ELEC-003", "name": "Laptop 15 Inch",
     "category": "Electronics", "price": 52999, "stock": 30,
     "description": "Core i5, 16GB RAM, 512GB SSD — ideal for work and study"},

    {"product_id": "ELEC-004", "name": "Smartphone 128GB",
     "category": "Electronics", "price": 17999, "stock": 200,
     "description": "Android smartphone, 6.5\" AMOLED, 50MP camera, 5000mAh"},

    {"product_id": "ELEC-005", "name": "Wireless Charging Pad",
     "category": "Electronics", "price": 999,   "stock": 300,
     "description": "15W fast wireless charger compatible with all Qi devices"},

    {"product_id": "ELEC-006", "name": "USB-C Hub 7-in-1",
     "category": "Electronics", "price": 1799,  "stock": 180,
     "description": "HDMI, USB 3.0 ×3, SD card, PD charging in one compact hub"},

    {"product_id": "ELEC-007", "name": "Mechanical Keyboard",
     "category": "Electronics", "price": 3499,  "stock": 90,
     "description": "TKL layout, blue switches, RGB backlit, USB-C cable"},

    {"product_id": "ELEC-008", "name": "Wireless Mouse",
     "category": "Electronics", "price": 1299,  "stock": 250,
     "description": "Ergonomic, 2.4GHz receiver, 12-month battery life"},

    {"product_id": "ELEC-009", "name": "Portable Bluetooth Speaker",
     "category": "Electronics", "price": 2199,  "stock": 120,
     "description": "IPX7 waterproof, 360° sound, 24hr playtime, USB-C"},

    {"product_id": "ELEC-010", "name": "Smart Watch",
     "category": "Electronics", "price": 4999,  "stock": 100,
     "description": "Health tracking, GPS, AMOLED display, 7-day battery"},

    {"product_id": "ELEC-011", "name": "Wireless Earbuds",
     "category": "Electronics", "price": 1799, "stock": 180,
     "description": "True wireless, 24hr total playtime, IPX4 water resistant, touch controls"},

    # ── KITCHEN & APPLIANCES ──────────────────────────────────────
    {"product_id": "KITCH-001", "name": "Electric Pressure Cooker 5L",
     "category": "Kitchen & Appliances", "price": 3299, "stock": 80,
     "description": "8-in-1 multi-cooker: pressure cook, slow cook, sauté, steam"},

    {"product_id": "KITCH-002", "name": "Air Fryer 4L",
     "category": "Kitchen & Appliances", "price": 4499, "stock": 65,
     "description": "Rapid hot air circulation, 1500W, digital touchscreen controls"},

    {"product_id": "KITCH-003", "name": "Mixer Grinder 750W",
     "category": "Kitchen & Appliances", "price": 2199, "stock": 110,
     "description": "3 stainless steel jars, 4-speed control with pulse function"},

    {"product_id": "KITCH-004", "name": "Non-Stick Cookware Set",
     "category": "Kitchen & Appliances", "price": 1899, "stock": 75,
     "description": "5-piece set: frying pan, saucepan, kadai, lid — PFOA free"},

    {"product_id": "KITCH-005", "name": "Electric Kettle 1.5L",
     "category": "Kitchen & Appliances", "price": 899,  "stock": 200,
     "description": "1500W rapid boil, auto shut-off, stainless steel body"},

    {"product_id": "KITCH-006", "name": "Coffee Maker",
     "category": "Kitchen & Appliances", "price": 3499, "stock": 55,
     "description": "Drip coffee maker, 12-cup capacity, programmable timer"},

    {"product_id": "KITCH-007", "name": "Induction Cooktop",
     "category": "Kitchen & Appliances", "price": 2799, "stock": 70,
     "description": "2000W, 7 power levels, child safety lock, auto shut-off"},

    {"product_id": "KITCH-008", "name": "Refrigerator 260L",
     "category": "Kitchen & Appliances", "price": 22999, "stock": 25,
     "description": "Double door, frost-free, 3-star energy rating"},

    # ── CLOTHING & FASHION ────────────────────────────────────────
    {"product_id": "CLTH-001", "name": "Men's Casual T-Shirt",
     "category": "Clothing & Fashion", "price": 399,  "stock": 500,
     "description": "100% cotton, crew neck, available in S/M/L/XL, multiple colours"},

    {"product_id": "CLTH-002", "name": "Women's Kurti",
     "category": "Clothing & Fashion", "price": 699,  "stock": 400,
     "description": "Printed rayon kurti, A-line fit, sizes XS–3XL"},

    {"product_id": "CLTH-003", "name": "Men's Formal Shirt",
     "category": "Clothing & Fashion", "price": 999,  "stock": 300,
     "description": "Slim fit, wrinkle-resistant cotton blend, office-ready"},

    {"product_id": "CLTH-004", "name": "Women's Jeans",
     "category": "Clothing & Fashion", "price": 1299, "stock": 350,
     "description": "Mid-rise slim fit, stretch denim, waist 26–36"},

    {"product_id": "CLTH-005", "name": "Running Shoes",
     "category": "Clothing & Fashion", "price": 2499, "stock": 200,
     "description": "Lightweight mesh upper, cushioned sole, sizes 6–11"},

    {"product_id": "CLTH-006", "name": "Leather Wallet",
     "category": "Clothing & Fashion", "price": 799,  "stock": 250,
     "description": "Genuine leather, 8 card slots, RFID blocking"},

    {"product_id": "CLTH-007", "name": "Winter Jacket",
     "category": "Clothing & Fashion", "price": 2999, "stock": 120,
     "description": "Water-resistant outer shell, fleece lining, zipper pockets"},

    {"product_id": "CLTH-008", "name": "Sunglasses UV400",
     "category": "Clothing & Fashion", "price": 599,  "stock": 300,
     "description": "Polarised UV400 lenses, lightweight frame, unisex design"},

    # ── BOOKS & STATIONERY ────────────────────────────────────────
    {"product_id": "BOOK-001", "name": "Python Programming Book",
     "category": "Books & Stationery", "price": 549,  "stock": 200,
     "description": "Comprehensive Python guide from beginner to advanced, 2024 edition"},

    {"product_id": "BOOK-002", "name": "Data Structures & Algorithms Book",
     "category": "Books & Stationery", "price": 699,  "stock": 150,
     "description": "Interview-focused DSA with 300+ problems and solutions"},

    {"product_id": "BOOK-003", "name": "A4 Notebook 200 Pages",
     "category": "Books & Stationery", "price": 149,  "stock": 1000,
     "description": "Hard cover, ruled pages, acid-free paper, spiral bound"},

    {"product_id": "BOOK-004", "name": "Ballpoint Pen Set 10-Pack",
     "category": "Books & Stationery", "price": 99,   "stock": 2000,
     "description": "Smooth writing, 0.7mm tip, assorted colours, ergonomic grip"},

    {"product_id": "BOOK-005", "name": "Sticky Notes Pack",
     "category": "Books & Stationery", "price": 129,  "stock": 1500,
     "description": "450 sheets, 3 sizes, 6 colours, strong adhesive"},

    {"product_id": "BOOK-006", "name": "Highlighter Set 6-Pack",
     "category": "Books & Stationery", "price": 199,  "stock": 800,
     "description": "Fluorescent colours, chisel tip, smear-free on most papers"},

    {"product_id": "BOOK-007", "name": "Scientific Calculator",
     "category": "Books & Stationery", "price": 799,  "stock": 300,
     "description": "417 functions, solar + battery, dot matrix display"},

    {"product_id": "BOOK-008", "name": "Self-Help Book — Atomic Habits",
     "category": "Books & Stationery", "price": 399,  "stock": 250,
     "description": "Bestselling guide on building good habits and breaking bad ones"},

    # ── FOOD & GROCERY ────────────────────────────────────────────
    {"product_id": "FOOD-001", "name": "Basmati Rice 5kg",
     "category": "Food & Grocery", "price": 499,  "stock": 400,
     "description": "Premium long-grain aged basmati, low GI, naturally fragrant"},

    {"product_id": "FOOD-002", "name": "Cold Pressed Coconut Oil 1L",
     "category": "Food & Grocery", "price": 449,  "stock": 300,
     "description": "100% natural virgin coconut oil, unrefined, edible grade"},

    {"product_id": "FOOD-003", "name": "Organic Honey 500g",
     "category": "Food & Grocery", "price": 349,  "stock": 500,
     "description": "Raw unfiltered forest honey, no added sugar, lab tested"},

    {"product_id": "FOOD-004", "name": "Almonds 500g",
     "category": "Food & Grocery", "price": 399,  "stock": 600,
     "description": "Premium California almonds, natural, rich in Vitamin E"},

    {"product_id": "FOOD-005", "name": "Green Tea 100 Bags",
     "category": "Food & Grocery", "price": 299,  "stock": 700,
     "description": "Tulsi green tea, antioxidant-rich, individually wrapped bags"},

    {"product_id": "FOOD-006", "name": "Dark Chocolate Bar 70%",
     "category": "Food & Grocery", "price": 199,  "stock": 800,
     "description": "70% cacao, no artificial flavours, single origin"},

    {"product_id": "FOOD-007", "name": "Mixed Nuts Trail Mix 400g",
     "category": "Food & Grocery", "price": 349,  "stock": 500,
     "description": "Cashews, walnuts, pistachios, raisins — roasted and lightly salted"},

    {"product_id": "FOOD-008", "name": "Protein Powder Whey 1kg",
     "category": "Food & Grocery", "price": 1299, "stock": 150,
     "description": "25g protein per serving, chocolate flavour, no artificial colours"},

    {"product_id": "FOOD-009", "name": "Fresh Flowers Bouquet",
     "category": "Food & Grocery", "price": 299,  "stock": 200,
     "description": "Seasonal mixed flower bouquet, same-day delivery available"},

    {"product_id": "FOOD-010", "name": "Pizza (Margherita)",
     "category": "Food & Grocery", "price": 349,  "stock": 100,
     "description": "10-inch fresh stone-baked Margherita pizza, serves 2"},
]


# ─────────────────────────────────────────────────────────────────────
# DB INIT  —  creates table + seeds catalog (idempotent)
# ─────────────────────────────────────────────────────────────────────

def init_products_db() -> None:
    """
    Called once at startup (from main.py, after init_db()).
    Creates the products table and seeds the catalog if empty.
    Safe to call multiple times — no duplicate inserts.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id  TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            category    TEXT NOT NULL,
            price       INTEGER NOT NULL,
            stock       INTEGER NOT NULL DEFAULT 0,
            description TEXT
        )
    """)

    # Idempotent seed — only insert rows that don't already exist
    cursor.executemany(
        """
        INSERT OR IGNORE INTO products (product_id, name, category, price, stock, description)
        VALUES (:product_id, :name, :category, :price, :stock, :description)
        """,
        PRODUCT_CATALOG,
    )

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────
# LOOKUP FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def get_product_by_id(product_id: str) -> dict | None:
    """Direct lookup by product_id. Used when order already has product_id."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT product_id, name, category, price, stock, description "
        "FROM products WHERE product_id = ?",
        (product_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_dict(row)


def search_product(name: str) -> dict | None:
    """
    Fuzzy search for a product by name.

    Strategy (fastest → most permissive):
      1. Exact match (case-insensitive)
      2. Starts-with match
      3. LIKE %keyword% on each word in the query (longest keyword first)
      4. difflib sequence match against all product names (handles typos)

    Returns the best-matching Product dict, or None if no match found.
    Out-of-stock products are deprioritised but still returned if nothing else matches.
    """
    if not name or not name.strip():
        return None

    query   = name.strip().lower()
    conn    = get_connection()
    cursor  = conn.cursor()

    # ── 1. Exact match ────────────────────────────────────────────
    cursor.execute(
        "SELECT product_id, name, category, price, stock, description "
        "FROM products WHERE LOWER(name) = ?",
        (query,)
    )
    row = cursor.fetchone()
    if row:
        conn.close()
        return _row_to_dict(row)

    # ── 2. Starts-with match ──────────────────────────────────────
    cursor.execute(
        "SELECT product_id, name, category, price, stock, description "
        "FROM products WHERE LOWER(name) LIKE ? ORDER BY stock DESC LIMIT 1",
        (f"{query}%",)
    )
    row = cursor.fetchone()
    if row:
        conn.close()
        return _row_to_dict(row)

    # ── 3a. Full-phrase LIKE match ────────────────────────────────
    # "wireless earbuds" → tries `%wireless earbuds%` BEFORE splitting into words.
    # This prevents "wireless" alone from matching "Wireless Charging Pad" when
    # the user actually meant "wireless earbuds" (a different product entirely).
    if len(query.split()) > 1:
        cursor.execute(
            "SELECT product_id, name, category, price, stock, description "
            "FROM products WHERE LOWER(name) LIKE ? ORDER BY stock DESC LIMIT 1",
            (f"%{query}%",)
        )
        row = cursor.fetchone()
        if row:
            conn.close()
            return _row_to_dict(row)

    # ── 3b. Keyword LIKE match — try significant words only ───────
    # Only runs if the full-phrase match above failed.
    # Filters to the LAST word only for multi-word queries (the noun/product word)
    # to avoid matching the adjective "wireless" across unrelated products.
    words_all = [w for w in query.split() if len(w) >= 3]
    # For 2+ word queries, try the last word (usually the noun) first,
    # then fall back to other words.
    words_sorted = (
        [words_all[-1]] + sorted(words_all[:-1], key=len, reverse=True)
        if len(words_all) > 1 else words_all
    )
    for word in words_sorted:
        # Try the word as-is, then try singular form (pens→pen, books→book)
        candidates = [word]
        if word.endswith("s") and len(word) > 3:
            candidates.append(word[:-1])   # e.g. "pens" → "pen"
        if word.endswith("es") and len(word) > 4:
            candidates.append(word[:-2])   # e.g. "watches" → "watch"
        for candidate in candidates:
            cursor.execute(
                "SELECT product_id, name, category, price, stock, description "
                "FROM products WHERE LOWER(name) LIKE ? ORDER BY stock DESC LIMIT 1",
                (f"%{candidate}%",)
            )
            row = cursor.fetchone()
            if row:
                conn.close()
                return _row_to_dict(row)

    # ── 4. difflib fuzzy match — handles typos / alternate phrasing ─
    cursor.execute(
        "SELECT product_id, name, category, price, stock, description FROM products"
    )
    all_rows = cursor.fetchall()
    conn.close()

    if not all_rows:
        return None

    all_names = [r[1].lower() for r in all_rows]
    matches   = difflib.get_close_matches(query, all_names, n=1, cutoff=0.52)
    if matches:
        matched_name = matches[0]
        for row in all_rows:
            if row[1].lower() == matched_name:
                return _row_to_dict(row)

    return None   # no match found


def check_stock(product_id: str, quantity: int) -> tuple[bool, int]:
    """
    Checks whether sufficient stock is available for an order.

    Returns (is_sufficient: bool, available_stock: int).
    Call this BEFORE committing an order — never let an order through
    when the requested quantity exceeds available stock.
    """
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT stock FROM products WHERE product_id = ?", (product_id,))
    row = cursor.fetchone()
    conn.close()
    available = row[0] if row else 0
    return available >= quantity, available


def update_product_stock(product_id: str, quantity: int) -> bool:
    """
    Decrements the stock for a product when an order is placed.
    Uses MAX(0, stock - quantity) to prevent stock going negative.
    Returns True if the product row was found and updated.
    """
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE products SET stock = MAX(0, stock - ?) WHERE product_id = ?",
        (quantity, product_id)
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def _row_to_dict(row: tuple) -> dict:
    return {
        "product_id":  row[0],
        "name":        row[1],
        "category":    row[2],
        "price":       row[3],
        "stock":       row[4],
        "description": row[5],
    }


# ─────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────

def format_product_card(product: dict, quantity: int = 1) -> str:
    """
    Returns a formatted product summary card for display in the chat.
    Shown to user BEFORE payment confirmation.
    """
    unit_price  = product["price"]
    total_price = unit_price * quantity
    category    = product["category"]

    CATEGORY_EMOJI = {
        "Electronics":          "📱",
        "Kitchen & Appliances": "🍳",
        "Clothing & Fashion":   "👗",
        "Books & Stationery":   "📚",
        "Food & Grocery":       "🛒",
    }
    emoji = CATEGORY_EMOJI.get(category, "🛍️")

    # Compute estimated delivery (5–7 business days from today)
    from datetime import date, timedelta
    today       = date.today()
    eta_start   = today + timedelta(days=5)
    eta_end     = today + timedelta(days=7)
    eta_str     = f"{eta_start.strftime('%d %b')} – {eta_end.strftime('%d %b %Y')}"

    lines = [
        f"🛍️ **Order Summary**\n",
        f"{emoji} **{product['name']}**",
        f"   📂 Category   : {category}",
        f"   📝 Description: {product['description']}",
        f"   💰 Price/unit : ₹{unit_price:,}",
        f"   🔢 Quantity   : {quantity}",
        f"   💳 Total      : ₹{total_price:,}",
        f"   📦 In Stock   : {product['stock']} units available",
        f"   🚚 Est. Delivery: {eta_str}",
        "",
        "**How would you like to pay?**",
        "   1️⃣  Cash on Delivery (COD)",
        "   2️⃣  UPI",
        "   3️⃣  Credit / Debit Card",
        "   4️⃣  Net Banking",
        "",
        "*Reply with the payment method name or number (e.g. COD or 1)*",
    ]
    return "\n".join(lines)


PAYMENT_ALIASES: dict[str, str] = {
    # COD
    "1": "Cash on Delivery (COD)", "cod": "Cash on Delivery (COD)",
    "cash": "Cash on Delivery (COD)", "cash on delivery": "Cash on Delivery (COD)",
    # UPI
    "2": "UPI", "upi": "UPI", "gpay": "UPI", "phonepe": "UPI",
    "paytm": "UPI", "google pay": "UPI",
    # Card
    "3": "Credit / Debit Card", "card": "Credit / Debit Card",
    "credit card": "Credit / Debit Card", "debit card": "Credit / Debit Card",
    "credit": "Credit / Debit Card", "debit": "Credit / Debit Card",
    # Net Banking
    "4": "Net Banking", "net banking": "Net Banking",
    "netbanking": "Net Banking", "bank transfer": "Net Banking",
    "internet banking": "Net Banking",
}


def resolve_payment_method(user_input: str) -> str | None:
    """
    Maps user's free-form payment input to one of the 4 canonical options.
    Returns the canonical string or None if unrecognised.
    """
    cleaned = user_input.strip().lower()
    return PAYMENT_ALIASES.get(cleaned)


# ─────────────────────────────────────────────────────────────────────
# CATALOG DISPLAY  —  called by the router for "show me products" intent
# Reads from the live products table (not the hardcoded list) so any
# future admin additions are reflected immediately.
# ─────────────────────────────────────────────────────────────────────

def format_catalog_response() -> str:
    """
    Returns a neatly formatted product catalog grouped by category,
    read directly from the products table in the DB.
    """
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, price, description, category "
        "FROM products ORDER BY category, price"
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "Our product catalog is being updated. Please check back shortly!"

    CATEGORY_EMOJI = {
        "Electronics":          "📱",
        "Kitchen & Appliances": "🍳",
        "Clothing & Fashion":   "👗",
        "Books & Stationery":   "📚",
        "Food & Grocery":       "🛒",
    }

    # Group by category
    from collections import defaultdict
    grouped: dict[str, list] = defaultdict(list)
    for name, price, desc, cat in rows:
        grouped[cat].append((name, price, desc))

    lines = ["🛒 **NovaCart Product Catalog**\n"]
    for category, products in grouped.items():
        emoji = CATEGORY_EMOJI.get(category, "🛍️")
        lines.append(f"\n{emoji} **{category}**")
        for name, price, desc in products:
            lines.append(f"   • **{name}** — ₹{price:,}")
            lines.append(f"     _{desc}_")

    lines.append(
        "\n---\n"
        "To order any item, just say: *Order 1 Wireless Bluetooth Headphones*\n"
        "Or ask me anything about a product!"
    )
    return "\n".join(lines)