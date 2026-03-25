# backend/db.py
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME  = os.path.join(BASE_DIR, "orders.db")


# ─────────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────────

def init_users_db():
    conn   = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id  TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()


def init_db():
    conn   = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id            INTEGER   PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT,
            order_id      TEXT      UNIQUE,
            product_id    TEXT,
            product_name  TEXT,
            status        TEXT,
            return_reason TEXT,
            return_date   TIMESTAMP,
            order_date    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("PRAGMA table_info(orders)")
    columns = [col[1] for col in cursor.fetchall()]
    if "quantity" not in columns:
        cursor.execute("ALTER TABLE orders ADD COLUMN quantity INTEGER DEFAULT 1")
    if "payment_mode" not in columns:
        cursor.execute("ALTER TABLE orders ADD COLUMN payment_mode TEXT DEFAULT 'COD'")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS support_tickets (
            id                  INTEGER   PRIMARY KEY AUTOINCREMENT,
            ticket_num          TEXT,
            user_id             TEXT,
            order_id            TEXT,
            issue               TEXT,
            status              TEXT,
            ticket_created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS returns (
            id                  INTEGER   PRIMARY KEY AUTOINCREMENT,
            user_id             TEXT,
            order_id            TEXT,
            reason              TEXT,
            status              TEXT,
            return_created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    init_users_db()


# ─────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


# ─────────────────────────────────────────────────────────────────────
# POSITIONAL QUERIES (used by router for "first/last/nth" references)
# ─────────────────────────────────────────────────────────────────────

def get_user_orders(user_id: str) -> list:
    """All orders sorted oldest-first. index 0 = first, index -1 = last."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT order_id, product_name, quantity, status, order_date
        FROM   orders WHERE user_id = ? ORDER BY order_date ASC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"order_id": r[0], "product_name": r[1],
             "quantity": r[2], "status": r[3], "order_date": r[4]} for r in rows]


def get_user_tickets(user_id: str) -> list:
    """All tickets sorted oldest-first."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ticket_num, order_id, issue, status, ticket_created_date
        FROM   support_tickets WHERE user_id = ? ORDER BY ticket_created_date ASC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"ticket_num": r[0], "order_id": r[1],
             "issue": r[2], "status": r[3], "created_date": r[4]} for r in rows]


def get_user_returns(user_id: str) -> list:
    """All returns sorted oldest-first."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT order_id, reason, status, return_created_date
        FROM   returns WHERE user_id = ? ORDER BY return_created_date ASC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"order_id": r[0], "reason": r[1],
             "status": r[2], "created_date": r[3]} for r in rows]


# ─────────────────────────────────────────────────────────────────────
# ANALYTICS QUERIES
# Pure SQL — no LLM involved. Returns structured data for direct display.
#
# Used when user asks:
#   "which is my most ordered product?"
#   "how many tickets have I raised?"
#   "how many returns have I made?"
#   "give me a summary of my activity"
# ─────────────────────────────────────────────────────────────────────

def get_user_analytics(user_id: str) -> dict:
    """
    Returns a complete analytics snapshot for a user.
    All queries run in a single DB connection for efficiency.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    # ── Total orders ──────────────────────────────────────────────
    cursor.execute(
        "SELECT COUNT(*) FROM orders WHERE user_id = ?", (user_id,)
    )
    total_orders = cursor.fetchone()[0]

    # ── Most ordered product (by total quantity) ──────────────────
    cursor.execute("""
        SELECT product_name, SUM(quantity) as total_qty
        FROM   orders
        WHERE  user_id = ?
        GROUP  BY product_name
        ORDER  BY total_qty DESC
        LIMIT  1
    """, (user_id,))
    row = cursor.fetchone()
    most_ordered_product  = row[0] if row else None
    most_ordered_quantity = row[1] if row else 0

    # ── Top 3 most ordered products ───────────────────────────────
    cursor.execute("""
        SELECT product_name, SUM(quantity) as total_qty
        FROM   orders
        WHERE  user_id = ?
        GROUP  BY product_name
        ORDER  BY total_qty DESC
        LIMIT  3
    """, (user_id,))
    top_products = [{"product": r[0], "quantity": r[1]} for r in cursor.fetchall()]

    # ── Total spend (quantity-based, no price in schema) ──────────
    cursor.execute(
        "SELECT SUM(quantity) FROM orders WHERE user_id = ?", (user_id,)
    )
    total_items_ordered = cursor.fetchone()[0] or 0

    # ── Orders by status ──────────────────────────────────────────
    cursor.execute("""
        SELECT status, COUNT(*) FROM orders
        WHERE  user_id = ?
        GROUP  BY status
    """, (user_id,))
    orders_by_status = {r[0]: r[1] for r in cursor.fetchall()}

    # ── Last order ────────────────────────────────────────────────
    cursor.execute("""
        SELECT order_id, product_name, quantity, status, order_date
        FROM   orders
        WHERE  user_id = ?
        ORDER  BY order_date DESC
        LIMIT  1
    """, (user_id,))
    row = cursor.fetchone()
    last_order = {
        "order_id":     row[0],
        "product_name": row[1],
        "quantity":     row[2],
        "status":       row[3],
        "order_date":   (row[4] or "")[:10],
    } if row else None

    # ── Total tickets ─────────────────────────────────────────────
    cursor.execute(
        "SELECT COUNT(*) FROM support_tickets WHERE user_id = ?", (user_id,)
    )
    total_tickets = cursor.fetchone()[0]

    # ── Tickets by status ─────────────────────────────────────────
    cursor.execute("""
        SELECT status, COUNT(*) FROM support_tickets
        WHERE  user_id = ?
        GROUP  BY status
    """, (user_id,))
    tickets_by_status = {r[0]: r[1] for r in cursor.fetchall()}

    # ── Total returns ─────────────────────────────────────────────
    cursor.execute(
        "SELECT COUNT(*) FROM returns WHERE user_id = ?", (user_id,)
    )
    total_returns = cursor.fetchone()[0]

    # ── Returns by status ─────────────────────────────────────────
    cursor.execute("""
        SELECT status, COUNT(*) FROM returns
        WHERE  user_id = ?
        GROUP  BY status
    """, (user_id,))
    returns_by_status = {r[0]: r[1] for r in cursor.fetchall()}

    # ── Return rate ───────────────────────────────────────────────
    return_rate = round((total_returns / total_orders * 100), 1) if total_orders > 0 else 0

    conn.close()

    return {
        "total_orders":          total_orders,
        "total_items_ordered":   total_items_ordered,
        "most_ordered_product":  most_ordered_product,
        "most_ordered_quantity": most_ordered_quantity,
        "top_products":          top_products,
        "orders_by_status":      orders_by_status,
        "last_order":            last_order,
        "total_tickets":         total_tickets,
        "tickets_by_status":     tickets_by_status,
        "total_returns":         total_returns,
        "returns_by_status":     returns_by_status,
        "return_rate":           return_rate,
        "_user_id":              user_id,   # passed through for date-based sub-queries
    }


def format_analytics_response(analytics: dict) -> str:
    """
    Formats the analytics dict into a clean, readable response
    for display directly in the chat.
    """
    a = analytics

    if a["total_orders"] == 0:
        return (
            "📊 **Your Activity Summary**\n\n"
            "You haven't placed any orders yet.\n\n"
            "Start shopping by saying: *Order 1 laptop*"
        )

    # Orders section
    lines = ["📊 **Your Activity Summary**\n"]

    lines.append(f"**🛍️ Orders**")
    lines.append(f"   Total Orders Placed: **{a['total_orders']}**")
    lines.append(f"   Total Items Ordered: **{a['total_items_ordered']}**")

    if a["top_products"]:
        top_str = ", ".join(
            f"{p['product']} ({p['quantity']} units)" for p in a["top_products"]
        )
        lines.append(f"   Most Ordered: **{a['most_ordered_product']}** ({a['most_ordered_quantity']} units)")
        if len(a["top_products"]) > 1:
            lines.append(f"   Top Products: {top_str}")

    if a["last_order"]:
        lo = a["last_order"]
        lines.append(
            f"   Last Order: **{lo['order_id']}** — {lo['product_name']} × {lo['quantity']}"
            f" ({lo['status']}) on {lo['order_date']}"
        )

    if a["orders_by_status"]:
        status_str = ", ".join(f"{k}: {v}" for k, v in a["orders_by_status"].items())
        lines.append(f"   Order Statuses: {status_str}")

    # Tickets section
    lines.append(f"\n**🎫 Support Tickets**")
    lines.append(f"   Total Tickets Raised: **{a['total_tickets']}**")
    if a["tickets_by_status"]:
        status_str = ", ".join(f"{k}: {v}" for k, v in a["tickets_by_status"].items())
        lines.append(f"   Ticket Statuses: {status_str}")

    # Returns section
    lines.append(f"\n**↩️ Returns**")
    lines.append(f"   Total Returns Requested: **{a['total_returns']}**")
    lines.append(f"   Return Rate: **{a['return_rate']}%**")
    if a["returns_by_status"]:
        status_str = ", ".join(f"{k}: {v}" for k, v in a["returns_by_status"].items())
        lines.append(f"   Return Statuses: {status_str}")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────
# DATE-BASED ORDER ANALYTICS
# Answers: "on which date did I place most of my orders?"
# ─────────────────────────────────────────────────────────────────────

def get_orders_by_date(user_id: str) -> list:
    """Returns orders grouped by date, sorted by count descending."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DATE(order_date) as order_day, COUNT(*) as count
        FROM   orders
        WHERE  user_id = ?
        GROUP  BY order_day
        ORDER  BY count DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"date": r[0], "count": r[1]} for r in rows]


# ─────────────────────────────────────────────────────────────────────
# SPECIFIC ANALYTICS RESPONSE
# Returns a targeted answer based on what the user actually asked.
# Falls back to full summary ONLY when explicitly asked for one.
# ─────────────────────────────────────────────────────────────────────

def get_specific_analytics_response(user_text: str, analytics: dict) -> str:
    """
    Detect what the user specifically asked about and return only that.

    Priority (first match wins):
      1. Date of most orders
      2. Most ordered / frequently ordered product
      3. Ticket count / status
      4. Return count / rate
      5. Order count (how many orders)
      6. Last / recent order
      7. Top products
      8. Full summary (explicit "summary", "stats", "activity")
    """
    lowered = user_text.lower()
    a       = analytics

    if a["total_orders"] == 0:
        return (
            "You haven't placed any orders yet.\n\n"
            "Start shopping by saying: *Order 1 laptop*"
        )

    # ── 1. Date-based query ────────────────────────────────────────
    if any(p in lowered for p in [
        "which date", "on which date", "what date", "which day",
        "placed most", "most orders on", "date i placed",
        "mostly order", "when do i", "when did i order most",
    ]):
        from backend.db import get_orders_by_date as _get_obd
        date_data = _get_obd(a.get("_user_id", ""))
        if date_data:
            top = date_data[0]
            others = date_data[1:3]
            resp = (
                f"📅 You placed the most orders on **{top['date']}** "
                f"({top['count']} order{'s' if top['count'] > 1 else ''})."
            )
            if others:
                other_str = ", ".join(f"{d['date']} ({d['count']})" for d in others)
                resp += f"\n   Other active days: {other_str}"
            return resp
        return "I couldn't find enough order history to determine your most active day."

    # ── 2. Most / frequently ordered product ──────────────────────
    if any(p in lowered for p in [
        "most ordered", "most purchased", "frequently ordered",
        "most frequent", "order most", "ordered most",
        "which product", "favourite product", "favorite product",
        "most popular", "buy most", "bought most",
    ]):
        if a["most_ordered_product"]:
            return (
                f"🛍️ Your most ordered product is **{a['most_ordered_product']}** "
                f"with **{a['most_ordered_quantity']} units** ordered in total."
            )
        return "You haven't ordered anything yet."

    # ── 3. Ticket-specific ─────────────────────────────────────────
    if any(p in lowered for p in ["ticket", "tickets", "complaint", "complaints"]):
        count  = a["total_tickets"]
        plural = "s" if count != 1 else ""
        if count == 0:
            return (
                "🎫 You haven't raised any support tickets yet.\n\n"
                "Raise one by saying: *Raise a ticket for ORD-XXXX — your issue*"
            )
        status_str = (
            ", ".join(f"{k}: {v}" for k, v in a["tickets_by_status"].items())
            if a["tickets_by_status"] else "None"
        )
        return (
            f"🎫 You have raised **{count} support ticket{plural}** so far.\n"
            f"   Status breakdown: {status_str}"
        )

    # ── 4. Return-specific ─────────────────────────────────────────
    if any(p in lowered for p in [
        "return", "returns", "returned", "refund", "sent back",
    ]):
        count  = a["total_returns"]
        plural = "s" if count != 1 else ""
        if count == 0:
            return (
                "↩️ You haven't made any return requests yet.\n\n"
                "Start one by saying: *Return ORD-XXXX — reason*"
            )
        status_str = (
            ", ".join(f"{k}: {v}" for k, v in a["returns_by_status"].items())
            if a["returns_by_status"] else "None"
        )
        return (
            f"↩️ You have made **{count} return request{plural}** so far.\n"
            f"   Return rate: **{a['return_rate']}%**\n"
            f"   Status breakdown: {status_str}"
        )

    # ── 5. Order count ─────────────────────────────────────────────
    if any(p in lowered for p in [
        "how many orders", "total orders", "number of orders",
        "orders placed", "times have i ordered", "how many times",
        "how many order", "how many purchase",
    ]):
        return (
            f"🛍️ You have placed **{a['total_orders']} orders** so far, "
            f"totalling **{a['total_items_ordered']} items**."
        )

    # ── 6. Last / most recent order ────────────────────────────────
    if any(p in lowered for p in [
        "last order", "recent order", "latest order", "newest order",
    ]):
        lo = a["last_order"]
        if lo:
            return (
                f"📦 Your most recent order is **{lo['order_id']}** — "
                f"{lo['product_name']} × {lo['quantity']} ({lo['status']}) "
                f"placed on {lo['order_date']}."
            )

    # ── 7. Top products ────────────────────────────────────────────
    if any(p in lowered for p in [
        "top product", "popular product", "best product",
    ]):
        if a["top_products"]:
            top_str = ", ".join(
                f"**{p['product']}** ({p['quantity']} units)"
                for p in a["top_products"]
            )
            return f"🛍️ Your top ordered products are: {top_str}"

    # ── 8. Full summary (explicit request) ────────────────────────
    return format_analytics_response(analytics)

# ─────────────────────────────────────────────────────────────────────
# USERNAME LOOKUP
# Used by router for personalised responses ("BhumiRaj")
# ─────────────────────────────────────────────────────────────────────

def get_username(user_id: str) -> str:
    """Returns the display username for a given user_id, or 'there' as fallback."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else "there"