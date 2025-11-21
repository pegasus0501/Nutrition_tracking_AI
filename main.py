import os
import base64
import json
from typing import Optional, List, Dict, Any, Set
from datetime import datetime, date, time, timedelta
from starlette.responses import RedirectResponse
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func
from dotenv import load_dotenv

load_dotenv()

from fastapi import (
    FastAPI,
    Request,
    Depends,
    Form,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_302_FOUND

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    func,
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session as DBSession

from passlib.hash import pbkdf2_sha256
import requests
from openai import OpenAI

# ---------------- CONFIG ----------------

# ---------------- CONFIG ----------------

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

USDA_API_KEY = os.getenv("USDA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "change-me-in-prod")

if not USDA_API_KEY:
    print("WARNING: USDA_API_KEY not set – USDA search will fail until configured.")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set – image analysis / estimation will fail until configured.")

client_oa = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------- WEBSOCKET CONNECTION MANAGER ----------------

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    def __init__(self):
        # Store connections by user_id: Set[WebSocket]
        self.active_connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)

    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

    async def broadcast_to_user(self, user_id: int, message: dict):
        """Send message to all connections for a specific user"""
        if user_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.add(connection)
            # Clean up disconnected connections
            for conn in disconnected:
                self.active_connections[user_id].discard(conn)

manager = ConnectionManager()

# ---------------- FASTAPI SETUP ----------------

app = FastAPI(title="Food Nutrition Tracker")

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------- DB SETUP ----------------

Base = declarative_base()

# Handle database connection args (check_same_thread is only for SQLite)
connect_args = {}
if "sqlite" in DATABASE_URL:
    connect_args["check_same_thread"] = False

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)


class FoodLog(Base):
    """
    Stores each food item the user logs.
    Macros are stored as per-serving (or per-100g) values at log time.
    """
    __tablename__ = "food_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    description = Column(String, nullable=False)
    serving_size_label = Column(String, nullable=False)
    basis = Column(String, nullable=False)       # e.g. "per serving", "per 100 g (assumed)"
    source = Column(String, nullable=False)      # "USDA" or "Estimated"

    # NEW – which meal this belongs to
    meal_type = Column(String, nullable=False, default="Dinner")  # Breakfast/Lunch/Dinner/Snack

    energy_kcal = Column(Float, nullable=True)
    protein_g = Column(Float, nullable=True)
    fat_g = Column(Float, nullable=True)
    carbs_g = Column(Float, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class UserGoal(Base):
    """
    Per-user daily goals for calories & macros.
    These names match what dashboard.html expects:
      daily_calories, daily_protein_g, daily_carbs_g, daily_fat_g
    """
    __tablename__ = "user_goals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)

    daily_calories = Column(Float, nullable=True)
    daily_protein_g = Column(Float, nullable=True)
    daily_carbs_g = Column(Float, nullable=True)
    daily_fat_g = Column(Float, nullable=True)

class PersonalInfo(Base):
    """
    Stores user's personal information for metabolic calculations.
    """
    __tablename__ = "personal_info"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)

    name = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)
    gender = Column(String, nullable=True)  # "Male", "Female", "Other"
    activity_level = Column(String, nullable=True)  # "Sedentary", "Light", "Moderate", "Active", "Very Active"
    weight_goal = Column(String, nullable=True)  # "Lose", "Maintain", "Gain"

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------- AUTH HELPERS ----------------

def get_current_user(request: Request, db: DBSession) -> Optional[User]:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


# ---------------- PASSWORD HELPERS (pbkdf2_sha256) ----------------

def hash_password(password: str) -> str:
    """
    Hash password using PBKDF2-SHA256.
    """
    return pbkdf2_sha256.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify password against stored PBKDF2-SHA256 hash.
    """
    return pbkdf2_sha256.verify(password, hashed)


# ---------------- USDA HELPERS ----------------

USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# nutrientNumber codes (kcal, protein, fat, carbs)
NUTRIENT_CODES = {
    "energy_kcal": "1008",
    "protein_g": "1003",
    "fat_g": "1004",
    "carbs_g": "1005",
}


def _extract_macros_from_detail(food_detail: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Given a /food/{fdcId} detail payload, try to extract kcal, protein, fat, carbs.

    We check:
    1) foodNutrients (by nutrientNumber OR nutrientId)
    2) labelNutrients (for branded / label-style entries)
    """
    macros = {"energy_kcal": None, "protein_g": None, "fat_g": None, "carbs_g": None}

    # ---- 1) foodNutrients path ----
    nutrients = food_detail.get("foodNutrients", []) or []
    for n in nutrients:
        num = str(
            n.get("nutrientNumber")
            or n.get("number")
            or ""
        )
        nid = str(
            n.get("nutrientId")
            or n.get("id")
            or ""
        )
        val = n.get("amount") if "amount" in n else n.get("value")

        if val is None:
            continue

        if num == NUTRIENT_CODES["energy_kcal"] or nid == NUTRIENT_CODES["energy_kcal"]:
            macros["energy_kcal"] = val
        elif num == NUTRIENT_CODES["protein_g"] or nid == NUTRIENT_CODES["protein_g"]:
            macros["protein_g"] = val
        elif num == NUTRIENT_CODES["fat_g"] or nid == NUTRIENT_CODES["fat_g"]:
            macros["fat_g"] = val
        elif num == NUTRIENT_CODES["carbs_g"] or nid == NUTRIENT_CODES["carbs_g"]:
            macros["carbs_g"] = val

    # If we already have something, stop
    if any(v is not None for v in macros.values()):
        return macros

    # ---- 2) labelNutrients fallback ----
    label = food_detail.get("labelNutrients") or {}
    if isinstance(label, dict):

        def from_label(keys: List[str]) -> Optional[float]:
            for k in keys:
                node = label.get(k)
                if isinstance(node, dict) and "value" in node:
                    return node["value"]
            return None

        macros["energy_kcal"] = macros["energy_kcal"] or from_label(
            ["calories", "energy", "Energy (kcal)"]
        )
        macros["protein_g"] = macros["protein_g"] or from_label(
            ["protein", "Protein (NLEA)"]
        )
        macros["fat_g"] = macros["fat_g"] or from_label(
            ["fat", "Total fat", "Total fat (NLEA)"]
        )
        macros["carbs_g"] = macros["carbs_g"] or from_label(
            ["carbohydrates", "carbs", "Total carbohydrate (NLEA)"]
        )

    return macros


def _extract_serving_info(food_detail: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Extract serving size and unit if available.
    For branded foods, servingSize & servingSizeUnit are present.
    Otherwise we assume 100 g basis.
    """
    serving_size = food_detail.get("servingSize")
    serving_unit = food_detail.get("servingSizeUnit")
    household = food_detail.get("householdServingFullText")

    if serving_size and serving_unit:
        basis = "per serving"
        size_label = f"{serving_size} {serving_unit}"
        if household:
            size_label = f"{size_label} ({household})"
    else:
        # assume 100 g for SR/other generic entries
        basis = "per 100 g (assumed)"
        size_label = "100 g"

    return {
        "serving_size_label": size_label,
        "basis": basis,
    }


def _fetch_detail_for_fdc_id(fdc_id: int) -> Dict[str, Any]:
    """
    Call /food/{fdcId} with nutrient filter for macros.
    """
    if not USDA_API_KEY:
        raise RuntimeError("USDA_API_KEY not configured")

    params = {
        "api_key": USDA_API_KEY,
        "nutrients": ",".join(NUTRIENT_CODES.values()),
    }
    url = f"{USDA_BASE_URL}/food/{fdc_id}"
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------- LLM ESTIMATION (for missing macros) ----------------

def estimate_macros_with_llm(description: str) -> Dict[str, Optional[float]]:
    """
    Use OpenAI to estimate kcal, protein, carbs, fat per serving
    when USDA doesn't provide macros.
    """
    if client_oa is None:
        return {"energy_kcal": None, "protein_g": None, "fat_g": None, "carbs_g": None}

    system_prompt = (
        "You are a nutrition expert. Given the name of a food, "
        "estimate typical nutrition per serving. "
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "energy_kcal": number,\n'
        '  "protein_g": number,\n'
        '  "carbs_g": number,\n'
        '  "fat_g": number\n'
        "}\n"
        "Do not include any explanation, only raw JSON."
    )

    response = client_oa.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Food name: {description}. Provide typical nutrition per serving.",
            },
        ],
        response_format={"type": "json_object"},
    )

    try:
        json_str = response.choices[0].message.content
        parsed = json.loads(json_str)
        return {
            "energy_kcal": float(parsed.get("energy_kcal")) if parsed.get("energy_kcal") is not None else None,
            "protein_g": float(parsed.get("protein_g")) if parsed.get("protein_g") is not None else None,
            "carbs_g": float(parsed.get("carbs_g")) if parsed.get("carbs_g") is not None else None,
            "fat_g": float(parsed.get("fat_g")) if parsed.get("fat_g") is not None else None,
        }
    except Exception as e:
        print("Estimation via LLM failed:", e)
        return {"energy_kcal": None, "protein_g": None, "fat_g": None, "carbs_g": None}


def search_usda_foods(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    1) Search /foods/search for basic info + FDC IDs.
    2) For each FDC ID, call /food/{fdcId} to pull macros and serving info.
    3) If macros missing, estimate via LLM and mark as 'Estimated'.
    """
    if not USDA_API_KEY:
        raise RuntimeError("USDA_API_KEY not configured")

    # Step 1: search
    search_params = {
        "api_key": USDA_API_KEY,
        "query": query,
        "pageSize": limit,
    }
    search_url = f"{USDA_BASE_URL}/foods/search"
    resp = requests.get(search_url, params=search_params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    foods = data.get("foods", []) or []
    results: List[Dict[str, Any]] = []

    # Step 2: hydrate each result with full macros + serving
    for food in foods:
        fdc_id = food.get("fdcId")
        description = food.get("description")
        brand_owner = food.get("brandOwner")
        data_type = food.get("dataType")

        macros = {"energy_kcal": None, "protein_g": None, "fat_g": None, "carbs_g": None}
        serving_label = None
        basis = None
        source = "USDA"

        if fdc_id:
            try:
                detail = _fetch_detail_for_fdc_id(int(fdc_id))
                macros = _extract_macros_from_detail(detail)
                serving_info = _extract_serving_info(detail)
                serving_label = serving_info["serving_size_label"]
                basis = serving_info["basis"]
            except Exception as e:
                print(f"USDA detail fetch failed for {fdc_id}: {e}")

        # If all macros are missing, try LLM estimation
        if not any(v is not None for v in macros.values()):
            est = estimate_macros_with_llm(description or "")
            if any(v is not None for v in est.values()):
                macros = est
                source = "Estimated"
                # For estimated values, define basis as "per serving (estimated)"
                if not serving_label:
                    serving_label = "1 serving (estimated)"
                if not basis:
                    basis = "per serving (estimated)"

        # If still no serving info, fall back to 100 g assumption
        if not serving_label:
            serving_label = "100 g"
        if not basis:
            basis = "per 100 g (assumed)"

        results.append(
            {
                "fdcId": fdc_id,
                "description": description,
                "brandOwner": brand_owner,
                "dataType": data_type,
                "serving_size_label": serving_label,
                "basis": basis,
                "source": source,
                "macros": macros,
            }
        )

    return results


# ---------------- OPENAI VISION (image → food names) ----------------

def analyze_image_to_food_names(image_bytes: bytes) -> List[str]:
    if client_oa is None or not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    system_prompt = (
        "You are a nutrition assistant. The user sends a food image. "
        "Identify the main edible items in the meal. "
        "Return ONLY valid JSON with this schema: "
        '{ "items": [ {"name": string} ] }.\n'
        "Do not include explanations or markdown."
    )

    response = client_oa.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this meal."},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
    )

    try:
        json_str = response.choices[0].message.content
        parsed = json.loads(json_str)
        items = parsed.get("items", []) or []
        names = [str(it.get("name", "")).strip() for it in items if it.get("name")]
        return [n for n in names if n]
    except Exception as e:
        print("Error parsing LLM response:", e)
        return []


# ---------------- SMALL HELPERS ----------------

def parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def today_range_utc():
    """
    Return (start_datetime, end_datetime) for *today* in UTC,
    matching the created_at timestamps we store with datetime.utcnow().
    """
    today_utc = datetime.utcnow().date()
    start = datetime.combine(today_utc, time.min)
    end = datetime.combine(today_utc, time.max)
    return start, end


# ---------------- ROUTES ----------------

@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    # No results by default on GET
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "user": user,
            "results": None,
            "image_results": None,
            "error": None,
        },
    )


@app.post("/search", response_class=HTMLResponse)
async def search_food(
    request: Request,
    query: str = Form(...),
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    error = None
    results = None
    try:
        results = search_usda_foods(query, limit=10)
    except Exception as e:
        error = f"USDA search failed: {e}"

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "user": user,
            "results": results,
            "image_results": None,
            "error": error,
        },
    )


@app.post("/upload-image", response_class=HTMLResponse)
async def upload_image(
    request: Request,
    image: UploadFile = File(...),
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    error = None
    image_results = None

    if image.content_type not in ["image/jpeg", "image/png"]:
        error = "Please upload a JPEG or PNG image."
    else:
        try:
            data = await image.read()
            names = analyze_image_to_food_names(data)
            aggregated = []

            for name in names:
                foods = search_usda_foods(name, limit=1)
                if foods:
                    aggregated.append(
                        {
                            "detected_name": name,
                            "food": foods[0],
                        }
                    )
            image_results = aggregated
        except Exception as e:
            error = f"Image analysis failed: {e}"

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "user": user,
            "results": None,
            "image_results": image_results,
            "error": error,
        },
    )


# -------- HELPER FUNCTIONS FOR DASHBOARD DATA --------

async def get_today_dashboard_data(user_id: int, db: DBSession) -> Dict[str, Any]:
    """Get today's dashboard totals and meal breakdown"""
    start, end = today_range_utc()

    # Overall totals for today
    totals_row = (
        db.query(
            func.coalesce(func.sum(FoodLog.energy_kcal), 0.0),
            func.coalesce(func.sum(FoodLog.protein_g), 0.0),
            func.coalesce(func.sum(FoodLog.carbs_g), 0.0),
            func.coalesce(func.sum(FoodLog.fat_g), 0.0),
        )
        .filter(
            FoodLog.user_id == user_id,
            FoodLog.created_at >= start,
            FoodLog.created_at <= end,
        )
        .one()
    )

    totals = {
        "energy_kcal": float(totals_row[0]),
        "protein_g": float(totals_row[1]),
        "carbs_g": float(totals_row[2]),
        "fat_g": float(totals_row[3]),
    }

    # Get all logs for today
    today_logs = (
        db.query(FoodLog)
        .filter(
            FoodLog.user_id == user_id,
            FoodLog.created_at >= start,
            FoodLog.created_at <= end,
        )
        .order_by(FoodLog.created_at.asc())
        .all()
    )

    # Group by meal
    meal_order = ["Breakfast", "Lunch", "Dinner", "Snack"]
    logs_by_meal: Dict[str, List[Dict]] = {m.lower(): [] for m in meal_order}
    meal_totals: Dict[str, Dict[str, float]] = {}

    for log in today_logs:
        mt = (log.meal_type or "").lower()
        if mt in logs_by_meal:
            logs_by_meal[mt].append({
                "id": log.id,
                "description": log.description,
                "serving_size_label": log.serving_size_label,
                "energy_kcal": log.energy_kcal or 0.0,
                "protein_g": log.protein_g or 0.0,
                "carbs_g": log.carbs_g or 0.0,
                "fat_g": log.fat_g or 0.0,
            })

    # Calculate meal totals
    for meal in meal_order:
        key = meal.lower()
        logs = logs_by_meal.get(key, [])
        meal_totals[key] = {
            "energy_kcal": sum(log["energy_kcal"] for log in logs),
            "protein_g": sum(log["protein_g"] for log in logs),
            "carbs_g": sum(log["carbs_g"] for log in logs),
            "fat_g": sum(log["fat_g"] for log in logs),
        }

    # Get goals
    goal = db.query(UserGoal).filter(UserGoal.user_id == user_id).first()
    goal_data = None
    if goal:
        goal_data = {
            "daily_calories": goal.daily_calories,
            "daily_protein_g": goal.daily_protein_g,
            "daily_carbs_g": goal.daily_carbs_g,
            "daily_fat_g": goal.daily_fat_g,
        }

    # Calculate progress
    def compute_progress(consumed: float, goal_value: Optional[float], unit: str):
        has_goal = bool(goal_value and goal_value > 0)
        pct = None
        pct_text = None
        if has_goal:
            pct = (consumed / goal_value) * 100.0 if goal_value > 0 else 0.0
            pct = max(0.0, min(pct, 300.0))
            pct_text = f"{pct:.1f}%"
        return {
            "has_goal": has_goal,
            "consumed": consumed,
            "goal": goal_value,
            "pct": pct,
            "pct_text": pct_text,
            "unit": unit,
        }

    nutrient_progress = {
        "calories": compute_progress(totals["energy_kcal"], goal_data["daily_calories"] if goal_data else None, "kcal"),
        "protein": compute_progress(totals["protein_g"], goal_data["daily_protein_g"] if goal_data else None, "g"),
        "carbs": compute_progress(totals["carbs_g"], goal_data["daily_carbs_g"] if goal_data else None, "g"),
        "fat": compute_progress(totals["fat_g"], goal_data["daily_fat_g"] if goal_data else None, "g"),
    }

    # Macro breakdown
    p = totals["protein_g"] or 0.0
    c = totals["carbs_g"] or 0.0
    f = totals["fat_g"] or 0.0
    cal_p = p * 4.0
    cal_c = c * 4.0
    cal_f = f * 9.0
    total_macro_cal = cal_p + cal_c + cal_f

    if total_macro_cal > 0:
        protein_pct = (cal_p / total_macro_cal) * 100.0
        carbs_pct = (cal_c / total_macro_cal) * 100.0
        fat_pct = (cal_f / total_macro_cal) * 100.0
        has_data = True
    else:
        protein_pct = carbs_pct = fat_pct = 0.0
        has_data = False

    macro_breakdown = {
        "has_data": has_data,
        "protein_pct": protein_pct,
        "carbs_pct": carbs_pct,
        "fat_pct": fat_pct,
        "total_macro_kcal": total_macro_cal,
    }

    return {
        "totals": totals,
        "logs_by_meal": logs_by_meal,
        "meal_order": meal_order,
        "meal_totals": meal_totals,
        "goal": goal_data,
        "nutrient_progress": nutrient_progress,
        "macro_breakdown": macro_breakdown,
    }


# -------- WEBSOCKET ENDPOINT --------

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back or handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)


# -------- API ENDPOINT FOR DASHBOARD DATA --------

@app.get("/api/dashboard/today")
async def api_dashboard_today(
    request: Request,
    db: DBSession = Depends(get_db),
):
    """API endpoint to get today's dashboard data"""
    user = get_current_user(request, db)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    data = await get_today_dashboard_data(user.id, db)
    return JSONResponse(data)


# -------- LOGGING FOOD (FROM SEARCH / IMAGE) --------

@app.post("/log/add")
async def add_food_log(
    request: Request,
    description: Optional[str] = Form(None),
    serving_size_label: Optional[str] = Form(None),
    basis: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    energy_kcal: Optional[str] = Form(None),
    protein_g: Optional[str] = Form(None),
    fat_g: Optional[str] = Form(None),
    carbs_g: Optional[str] = Form(None),
    meal_type: Optional[str] = Form("Dinner"),
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        if request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

    # Check if this is a JSON request (AJAX)
    is_json = request.headers.get("content-type", "").startswith("application/json")
    
    if is_json:
        # Parse JSON body
        body = await request.json()
        description = body.get("description")
        serving_size_label = body.get("serving_size_label")
        basis = body.get("basis")
        source = body.get("source")
        energy_kcal = str(body.get("energy_kcal", "")) if body.get("energy_kcal") is not None else None
        protein_g = str(body.get("protein_g", "")) if body.get("protein_g") is not None else None
        fat_g = str(body.get("fat_g", "")) if body.get("fat_g") is not None else None
        carbs_g = str(body.get("carbs_g", "")) if body.get("carbs_g") is not None else None
        meal_type = body.get("meal_type", "Dinner")

    if not description:
        if is_json:
            return JSONResponse({"error": "Missing required fields"}, status_code=400)
        return RedirectResponse(url="/", status_code=HTTP_302_FOUND)

    # normalise meal type
    allowed = {"Breakfast", "Lunch", "Dinner", "Snack"}
    if meal_type not in allowed:
        meal_type = "Dinner"

    log = FoodLog(
        user_id=user.id,
        description=description,
        serving_size_label=serving_size_label or "",
        basis=basis or "",
        source=source or "",
        meal_type=meal_type,
        energy_kcal=parse_optional_float(energy_kcal),
        protein_g=parse_optional_float(protein_g),
        fat_g=parse_optional_float(fat_g),
        carbs_g=parse_optional_float(carbs_g),
        created_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    # Get updated dashboard data
    dashboard_data = await get_today_dashboard_data(user.id, db)

    # Broadcast update via WebSocket
    await manager.broadcast_to_user(user.id, {
        "type": "food_logged",
        "data": dashboard_data,
    })

    # Return appropriate response
    if is_json:
        return JSONResponse({
            "success": True,
            "message": "Food logged successfully",
            "dashboard": dashboard_data,
        })
    else:
        # Traditional form submission - redirect
        return RedirectResponse(url="/dashboard", status_code=HTTP_302_FOUND)


@app.post("/log/add-multiple")
async def add_multiple_food_logs(
    request: Request,
    meal_type: Optional[str] = Form(None),
    description: Optional[List[str]] = Form(None),
    serving_size_label: Optional[List[str]] = Form(None),
    basis: Optional[List[str]] = Form(None),
    source: Optional[List[str]] = Form(None),
    energy_kcal: Optional[List[Optional[str]]] = Form([]),
    protein_g: Optional[List[Optional[str]]] = Form([]),
    fat_g: Optional[List[Optional[str]]] = Form([]),
    carbs_g: Optional[List[Optional[str]]] = Form([]),
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        if request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

    # Check if this is a JSON request (AJAX)
    is_json = request.headers.get("content-type", "").startswith("application/json")
    
    if is_json:
        # Parse JSON body
        body = await request.json()
        foods = body.get("foods", [])
        meal_type = body.get("meal_type", "Dinner")
        
        description = [f.get("description") for f in foods]
        serving_size_label = [f.get("serving_size_label", "") for f in foods]
        basis = [f.get("basis", "") for f in foods]
        source = [f.get("source", "") for f in foods]
        energy_kcal = [str(f.get("energy_kcal", "")) if f.get("energy_kcal") is not None else None for f in foods]
        protein_g = [str(f.get("protein_g", "")) if f.get("protein_g") is not None else None for f in foods]
        fat_g = [str(f.get("fat_g", "")) if f.get("fat_g") is not None else None for f in foods]
        carbs_g = [str(f.get("carbs_g", "")) if f.get("carbs_g") is not None else None for f in foods]

    if not description or len(description) == 0:
        if is_json:
            return JSONResponse({"error": "No foods to add"}, status_code=400)
        return RedirectResponse(url="/", status_code=HTTP_302_FOUND)

    # normalise meal type coming from the dropdown (breakfast/lunch/...)
    allowed = {"Breakfast", "Lunch", "Dinner", "Snack"}
    meal_norm = meal_type.title() if meal_type else "Dinner"
    if meal_norm not in allowed:
        meal_norm = "Dinner"

    n = len(description)
    for i in range(n):
        log = FoodLog(
            user_id=user.id,
            description=description[i],
            serving_size_label=serving_size_label[i] if i < len(serving_size_label) else "",
            basis=basis[i] if i < len(basis) else "",
            source=source[i] if i < len(source) else "",
            meal_type=meal_norm,
            energy_kcal=parse_optional_float(energy_kcal[i] if i < len(energy_kcal) else None),
            protein_g=parse_optional_float(protein_g[i] if i < len(protein_g) else None),
            fat_g=parse_optional_float(fat_g[i] if i < len(fat_g) else None),
            carbs_g=parse_optional_float(carbs_g[i] if i < len(carbs_g) else None),
            created_at=datetime.utcnow(),
        )
        db.add(log)

    db.commit()

    # Get updated dashboard data
    dashboard_data = await get_today_dashboard_data(user.id, db)

    # Broadcast update via WebSocket
    await manager.broadcast_to_user(user.id, {
        "type": "food_logged",
        "data": dashboard_data,
    })

    # Return appropriate response
    if is_json:
        return JSONResponse({
            "success": True,
            "message": f"{n} food(s) logged successfully",
            "dashboard": dashboard_data,
        })
    else:
        return RedirectResponse(url="/dashboard", status_code=HTTP_302_FOUND)

@app.post("/log/delete/{log_id}")
async def delete_food_log(
    request: Request,
    log_id: int,
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        if request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

    is_json = request.headers.get("content-type", "").startswith("application/json")

    log = (
        db.query(FoodLog)
        .filter(FoodLog.id == log_id, FoodLog.user_id == user.id)
        .first()
    )
    if log:
        db.delete(log)
        db.commit()

        # Get updated dashboard data
        dashboard_data = await get_today_dashboard_data(user.id, db)

        # Broadcast update via WebSocket
        await manager.broadcast_to_user(user.id, {
            "type": "food_deleted",
            "data": dashboard_data,
        })

        if is_json:
            return JSONResponse({
                "success": True,
                "message": "Food deleted successfully",
                "dashboard": dashboard_data,
            })

    if is_json:
        return JSONResponse({"success": True, "message": "Food not found"})

    return RedirectResponse(url="/dashboard", status_code=HTTP_302_FOUND)


# -------- AUTH: REGISTER / LOGIN / LOGOUT --------

@app.get("/register", response_class=HTMLResponse)
async def register_get(request: Request):
    return templates.TemplateResponse(
        "register.html",
        {"request": request, "error": None},
    )


@app.post("/register", response_class=HTMLResponse)
async def register_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: DBSession = Depends(get_db),
):
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Email already registered."},
        )

    hashed = hash_password(password)
    user = User(email=email, password_hash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)

    response = RedirectResponse(url="/", status_code=HTTP_302_FOUND)
    request.session["user_id"] = user.id
    request.session["user_email"] = user.email
    return response


@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": None},
    )


@app.post("/login", response_class=HTMLResponse)
async def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: DBSession = Depends(get_db),
):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid email or password."},
        )

    response = RedirectResponse(url="/", status_code=HTTP_302_FOUND)
    request.session["user_id"] = user.id
    request.session["user_email"] = user.email
    return response


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=HTTP_302_FOUND)


# --------- GOALS (SET / UPDATE) ---------

# --------- GOALS (SET / UPDATE / DELETE) ---------

@app.get("/goals", response_class=HTMLResponse)
async def goals_get(
    request: Request,
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

    goal = db.query(UserGoal).filter(UserGoal.user_id == user.id).first()
    has_goal = goal is not None

    return templates.TemplateResponse(
        "goals.html",
        {
            "request": request,
            "user": user,
            "goal": goal,
            "has_goal": has_goal,
        },
    )


@app.post("/goals", response_class=HTMLResponse)
async def goals_post(
    request: Request,
        daily_calories: Optional[str] = Form(None),
        daily_protein_g: Optional[str] = Form(None),
        daily_carbs_g: Optional[str] = Form(None),
        daily_fat_g: Optional[str] = Form(None),
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

    goal = db.query(UserGoal).filter(UserGoal.user_id == user.id).first()
    if not goal:
        goal = UserGoal(user_id=user.id)

    # map form fields → actual DB columns
    goal.daily_calories = parse_optional_float(daily_calories)
    goal.daily_protein_g = parse_optional_float(daily_protein_g)
    goal.daily_carbs_g = parse_optional_float(daily_carbs_g)
    goal.daily_fat_g = parse_optional_float(daily_fat_g)

    db.add(goal)
    db.commit()

    return RedirectResponse(url="/dashboard", status_code=HTTP_302_FOUND)

@app.post("/goals/delete", response_class=HTMLResponse)
async def goals_delete(
    request: Request,
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

    goal = db.query(UserGoal).filter(UserGoal.user_id == user.id).first()
    if goal:
        db.delete(goal)
        db.commit()

    # Back to goals page so user sees empty state
    return RedirectResponse(url="/goals", status_code=HTTP_302_FOUND)

# --------- PERSONAL INFORMATION ---------

@app.get("/personal-info", response_class=HTMLResponse)
async def personal_info_get(
    request: Request,
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

    personal_info = db.query(PersonalInfo).filter(PersonalInfo.user_id == user.id).first()

    return templates.TemplateResponse(
        "personal_info.html",
        {
            "request": request,
            "user": user,
            "personal_info": personal_info,
        },
    )


@app.post("/personal-info", response_class=HTMLResponse)
async def personal_info_post(
    request: Request,
    name: Optional[str] = Form(None),
    age: Optional[str] = Form(None),
    height_cm: Optional[str] = Form(None),
    weight_kg: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    activity_level: Optional[str] = Form(None),
    weight_goal: Optional[str] = Form(None),
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

    personal_info = db.query(PersonalInfo).filter(PersonalInfo.user_id == user.id).first()
    if not personal_info:
        personal_info = PersonalInfo(user_id=user.id)

    # Update fields
    if name is not None:
        personal_info.name = name.strip() if name.strip() else None
    personal_info.age = parse_optional_float(age)
    personal_info.height_cm = parse_optional_float(height_cm)
    personal_info.weight_kg = parse_optional_float(weight_kg)
    if gender:
        personal_info.gender = gender
    if activity_level:
        personal_info.activity_level = activity_level
    if weight_goal:
        personal_info.weight_goal = weight_goal
    personal_info.updated_at = datetime.utcnow()

    db.add(personal_info)
    db.commit()

    return RedirectResponse(url="/dashboard", status_code=HTTP_302_FOUND)
# --------- DASHBOARD (TODAY'S LOG + TOTALS, GROUPED BY MEAL) ---------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    view: Optional[str] = None,  # "today" or "twoweeks"
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

    # Default to "today" if not specified
    if view is None:
        view = "today"

    # Get personal info
    personal_info = db.query(PersonalInfo).filter(PersonalInfo.user_id == user.id).first()

    if view == "twoweeks":
        # Calculate date range for last 14 days
        today = datetime.utcnow().date()
        start_date = today - timedelta(days=13)  # 14 days total (today + 13 days back)
        start_datetime = datetime.combine(start_date, time.min)
        end_datetime = datetime.combine(today, time.max)

        # Get all logs in the 14-day range
        logs_14days = (
            db.query(FoodLog)
            .filter(
                FoodLog.user_id == user.id,
                FoodLog.created_at >= start_datetime,
                FoodLog.created_at <= end_datetime,
            )
            .all()
        )

        # Group by date
        daily_totals = {}
        for log in logs_14days:
            log_date = log.created_at.date()
            if log_date not in daily_totals:
                daily_totals[log_date] = {
                    "energy_kcal": 0.0,
                    "protein_g": 0.0,
                    "carbs_g": 0.0,
                    "fat_g": 0.0,
                }
            daily_totals[log_date]["energy_kcal"] += log.energy_kcal or 0.0
            daily_totals[log_date]["protein_g"] += log.protein_g or 0.0
            daily_totals[log_date]["carbs_g"] += log.carbs_g or 0.0
            daily_totals[log_date]["fat_g"] += log.fat_g or 0.0

        # Create list of all 14 days with data
        two_week_data = []
        for i in range(14):
            date = start_date + timedelta(days=i)
            data = daily_totals.get(date, {
                "energy_kcal": 0.0,
                "protein_g": 0.0,
                "carbs_g": 0.0,
                "fat_g": 0.0,
            })
            two_week_data.append({
            "date": date.isoformat(),     # or date.strftime("%Y-%m-%d")
            "date_str": date.strftime("%m/%d"),
            "is_today": date == today,
            "energy_kcal": data["energy_kcal"],
            "protein_g": data["protein_g"],
            "carbs_g": data["carbs_g"],
            "fat_g": data["fat_g"],
        })

        # Calculate statistics
        total_calories = sum(d["energy_kcal"] for d in two_week_data)
        avg_daily_calories = total_calories / 14.0 if len(two_week_data) > 0 else 0.0

        # Get goal for "on track" calculation
        goal = db.query(UserGoal).filter(UserGoal.user_id == user.id).first()
        goal_calories = goal.daily_calories if goal and goal.daily_calories else None

        days_on_track = 0
        if goal_calories:
            for day_data in two_week_data:
                if day_data["energy_kcal"] > 0 and day_data["energy_kcal"] <= goal_calories:
                    days_on_track += 1

        # Find best day (highest calories)
        best_day = max(two_week_data, key=lambda x: x["energy_kcal"]) if two_week_data else None

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "user": user,
                "view": "twoweeks",
                "personal_info": personal_info,
                "two_week_data": two_week_data,
                "total_calories": total_calories,
                "avg_daily_calories": avg_daily_calories,
                "days_on_track": days_on_track,
                "best_day": best_day,
                "goal_calories": goal_calories,
            },
        )

    else:  # view == "today"
        # Existing today view code...
        start, end = today_range_utc()

        # All logs for *today*
        today_logs = (
            db.query(FoodLog)
            .filter(
                FoodLog.user_id == user.id,
                FoodLog.created_at >= start,
                FoodLog.created_at <= end,
            )
            .order_by(FoodLog.created_at.asc())
            .all()
        )

        # Meal order for UI + grouping key
        meal_order = ["Breakfast", "Lunch", "Dinner", "Snack"]
        logs_by_meal: dict[str, list[FoodLog]] = {m.lower(): [] for m in meal_order}
        for log in today_logs:
            mt = (log.meal_type or "").lower()
            if mt in logs_by_meal:
                logs_by_meal[mt].append(log)

        # Per-meal totals
        meal_totals: dict[str, dict[str, float]] = {}
        for meal in meal_order:
            key = meal.lower()
            logs = logs_by_meal.get(key, [])
            energy = sum((log.energy_kcal or 0.0) for log in logs)
            protein = sum((log.protein_g or 0.0) for log in logs)
            carbs = sum((log.carbs_g or 0.0) for log in logs)
            fat = sum((log.fat_g or 0.0) for log in logs)
            meal_totals[key] = {
                "energy_kcal": energy,
                "protein_g": protein,
                "carbs_g": carbs,
                "fat_g": fat,
            }

        # Overall totals for today
        totals_row = (
            db.query(
                func.coalesce(func.sum(FoodLog.energy_kcal), 0.0),
                func.coalesce(func.sum(FoodLog.protein_g), 0.0),
                func.coalesce(func.sum(FoodLog.carbs_g), 0.0),
                func.coalesce(func.sum(FoodLog.fat_g), 0.0),
            )
            .filter(
                FoodLog.user_id == user.id,
                FoodLog.created_at >= start,
                FoodLog.created_at <= end,
            )
            .one()
        )

        totals = {
            "energy_kcal": float(totals_row[0]),
            "protein_g": float(totals_row[1]),
            "carbs_g": float(totals_row[2]),
            "fat_g": float(totals_row[3]),
        }

        # Goals
        goal: UserGoal | None = (
            db.query(UserGoal).filter(UserGoal.user_id == user.id).first()
        )

        goal_kcal = goal.daily_calories if goal else None
        goal_protein = goal.daily_protein_g if goal else None
        goal_carbs = goal.daily_carbs_g if goal else None
        goal_fat = goal.daily_fat_g if goal else None

        def compute_progress(consumed: float, goal_value: float | None, unit: str):
            has_goal = bool(goal_value and goal_value > 0)
            pct = None
            pct_text = None
            if has_goal:
                pct = (consumed / goal_value) * 100.0 if goal_value > 0 else 0.0
                pct = max(0.0, min(pct, 300.0))
                pct_text = f"{pct:.1f}%"
            return {
                "has_goal": has_goal,
                "consumed": consumed,
                "goal": goal_value,
                "pct": pct,
                "pct_text": pct_text,
                "unit": unit,
            }

        nutrient_progress = {
            "calories": compute_progress(totals["energy_kcal"], goal_kcal, "kcal"),
            "protein": compute_progress(totals["protein_g"], goal_protein, "g"),
            "carbs": compute_progress(totals["carbs_g"], goal_carbs, "g"),
            "fat": compute_progress(totals["fat_g"], goal_fat, "g"),
        }

        # Macro kcal breakdown
        p = totals["protein_g"] or 0.0
        c = totals["carbs_g"] or 0.0
        f = totals["fat_g"] or 0.0

        cal_p = p * 4.0
        cal_c = c * 4.0
        cal_f = f * 9.0
        total_macro_cal = cal_p + cal_c + cal_f

        if total_macro_cal > 0:
            protein_pct = (cal_p / total_macro_cal) * 100.0
            carbs_pct = (cal_c / total_macro_cal) * 100.0
            fat_pct = (cal_f / total_macro_cal) * 100.0
            has_data = True
        else:
            protein_pct = carbs_pct = fat_pct = 0.0
            has_data = False

        macro_breakdown = {
            "has_data": has_data,
            "protein_pct": protein_pct,
            "carbs_pct": carbs_pct,
            "fat_pct": fat_pct,
            "total_macro_kcal": total_macro_cal,
        }

        recent_logs = (
            db.query(FoodLog)
            .filter(FoodLog.user_id == user.id)
            .order_by(FoodLog.created_at.desc())
            .limit(20)
            .all()
        )

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "user": user,
                "view": "today",
                "personal_info": personal_info,
                "logs_by_meal": logs_by_meal,
                "meal_order": meal_order,
                "meal_totals": meal_totals,
                "totals": totals,
                "goal": goal,
                "nutrient_progress": nutrient_progress,
                "macro_breakdown": macro_breakdown,
                "recent_logs": recent_logs,
            },
        )

# -------- AI RECOMMENDATIONS --------

@app.post("/api/recommendations/generate")
async def generate_ai_recommendations(
    request: Request,
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    if not client_oa:
        return JSONResponse({"error": "OpenAI API key not configured"}, status_code=500)

    # 1. Gather User Context
    personal_info = db.query(PersonalInfo).filter(PersonalInfo.user_id == user.id).first()
    goal = db.query(UserGoal).filter(UserGoal.user_id == user.id).first()
    
    # Get today's logs
    start, end = today_range_utc()
    today_logs = (
        db.query(FoodLog)
        .filter(
            FoodLog.user_id == user.id,
            FoodLog.created_at >= start,
            FoodLog.created_at <= end,
        )
        .all()
    )

    # Calculate totals
    total_kcal = sum(l.energy_kcal or 0 for l in today_logs)
    total_protein = sum(l.protein_g or 0 for l in today_logs)
    total_carbs = sum(l.carbs_g or 0 for l in today_logs)
    total_fat = sum(l.fat_g or 0 for l in today_logs)

    # Format foods list
    foods_eaten = []
    for log in today_logs:
        foods_eaten.append(f"- {log.meal_type}: {log.description} ({log.serving_size_label})")
    
    foods_str = "\n".join(foods_eaten) if foods_eaten else "Nothing eaten yet today."

    # Format Profile
    profile_str = "Unknown"
    if personal_info:
        profile_str = (
            f"Age: {personal_info.age}, Gender: {personal_info.gender}, "
            f"Height: {personal_info.height_cm}cm, Weight: {personal_info.weight_kg}kg, "
            f"Activity: {personal_info.activity_level}, Goal: {personal_info.weight_goal}"
        )

    # Format Goals
    goals_str = "None set"
    if goal:
        goals_str = (
            f"Daily Targets -> Calories: {goal.daily_calories}, "
            f"Protein: {goal.daily_protein_g}g, Carbs: {goal.daily_carbs_g}g, Fat: {goal.daily_fat_g}g"
        )

    # 2. Construct Prompt
    system_prompt = (
        "You are an expert Nutritionist and Health Coach. "
        "Your goal is to analyze the user's profile, daily goals, and food intake for today, "
        "and provide 4-5 specific, actionable, and encouraging recommendations. "
        "Focus on what they can do for the rest of the day or tomorrow to stay on track or improve. "
        "Return ONLY valid JSON with this schema: "
        '{ "recommendations": [ "rec1", "rec2", "rec3", "rec4" ] }'
    )

    user_message = (
        f"User Profile: {profile_str}\n"
        f"Daily Goals: {goals_str}\n"
        f"Consumed Today (Totals): {total_kcal:.0f} kcal, {total_protein:.1f}g Protein, {total_carbs:.1f}g Carbs, {total_fat:.1f}g Fat.\n"
        f"Foods Eaten Today:\n{foods_str}\n\n"
        "Please provide 4-5 personalized recommendations."
    )

    try:
        response = client_oa.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content
        parsed = json.loads(content)
        recommendations = parsed.get("recommendations", [])
        
        return JSONResponse({"recommendations": recommendations})

    except Exception as e:
        print(f"AI Recommendation Error: {e}")
        return JSONResponse({"error": "Failed to generate recommendations"}, status_code=500)


@app.post("/api/plan/generate")
async def generate_ai_plan(
    request: Request,
    db: DBSession = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    if not client_oa:
        return JSONResponse({"error": "OpenAI API key not configured"}, status_code=500)

    # 1. Gather User Context
    personal_info = db.query(PersonalInfo).filter(PersonalInfo.user_id == user.id).first()
    
    # Fetch recent logs to understand eating habits
    recent_logs = (
        db.query(FoodLog)
        .filter(FoodLog.user_id == user.id)
        .order_by(FoodLog.created_at.desc())
        .limit(20)
        .all()
    )
    
    recent_foods = [f"{l.description} ({l.meal_type})" for l in recent_logs]
    recent_foods_str = ", ".join(recent_foods) if recent_foods else "No recent food logs found."

    # Format Profile
    profile_str = "Unknown"
    weight_goal = "Maintain"
    if personal_info:
        weight_goal = personal_info.weight_goal or "Maintain"
        profile_str = (
            f"Age: {personal_info.age}, Gender: {personal_info.gender}, "
            f"Height: {personal_info.height_cm}cm, Weight: {personal_info.weight_kg}kg, "
            f"Activity: {personal_info.activity_level}, Goal: {weight_goal}"
        )

    # 2. Construct Prompt
    system_prompt = (
        "You are an elite Health Strategist and High-Performance Coach. "
        "Your goal is to design a comprehensive, long-term 'Transformation Strategy' for the user. "
        "Unlike a daily check-in, this is a high-level roadmap. "
        "Analyze their profile and recent eating habits to provide unique, specific, and advanced advice. "
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "exercise_plan": "string (Detailed workout split, specific exercises, and intensity guidelines)",\n'
        '  "diet_plan": "string (Specific dietary philosophy, key foods to include/exclude, and meal timing strategy)",\n'
        '  "habit_plan": "string (3 unique, psychological or behavioral habits to optimize their relationship with food and lifestyle)"\n'
        "}"
    )

    user_message = (
        f"User Profile: {profile_str}\n"
        f"Primary Goal: {weight_goal}\n"
        f"Recent Eating Habits (Last 20 items): {recent_foods_str}\n\n"
        "Create a unique, personalized strategy plan. Avoid generic advice. Be specific."
    )

    try:
        response = client_oa.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content
        parsed = json.loads(content)
        
        return JSONResponse(parsed)

    except Exception as e:
        print(f"AI Plan Error: {e}")
        return JSONResponse({"error": "Failed to generate plan"}, status_code=500)

@app.get("/health")
async def health():
    return {"status": "ok"}
