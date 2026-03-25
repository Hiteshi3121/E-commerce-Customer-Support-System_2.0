import sqlite3, uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.db import DB_NAME

router = APIRouter(prefix="/auth", tags=["auth"])

class AuthReq(BaseModel):
    username: str
    password: str


@router.post("/signup")
def signup(req: AuthReq):
    user_id = f"user_{uuid.uuid4().hex[:6]}"

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users VALUES (?, ?, ?)",
            (user_id, req.username, req.password)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username exists")
    finally:
        conn.close()

    return {"user_id": user_id}


@router.post("/login")
def login(req: AuthReq):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT user_id FROM users WHERE username=? AND password=?",
        (req.username, req.password)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"user_id": row[0]}
