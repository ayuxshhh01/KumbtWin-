import os
import cv2
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import List, Optional, Any

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, APIRouter, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Boolean, TIMESTAMP, ForeignKey, DECIMAL, JSON, TEXT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

# ----------------------------
# 1. Configuration
# ----------------------------
DATABASE_URL = "sqlite:///./event_alert_system.db"
SECRET_KEY = "super_secret_key_change_this_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# ----------------------------
# 2. Database Setup
# ----------------------------
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ----------------------------
# 3. SQLAlchemy Models
# ----------------------------
class UserRole(Base):
    __tablename__ = "User_Roles"
    user_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    role_id = Column(Integer, ForeignKey("Roles.role_id"), primary_key=True)

class Role(Base):
    __tablename__ = "Roles"
    role_id = Column(Integer, primary_key=True, index=True)
    role_name = Column(String(50), unique=True, nullable=False)
    users = relationship("User", secondary="User_Roles", back_populates="roles")

class User(Base):
    __tablename__ = "Users"
    user_id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    roles = relationship("Role", secondary="User_Roles", back_populates="users")

class Event(Base):
    __tablename__ = "Events"
    event_id = Column(Integer, primary_key=True, index=True)
    event_name = Column(String(255), nullable=False)
    start_timestamp = Column(TIMESTAMP, nullable=False)
    end_timestamp = Column(TIMESTAMP)
    zones = relationship("Zone", back_populates="event")

class Zone(Base):
    __tablename__ = "Zones"
    zone_id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("Events.event_id"), nullable=True) # Linked to Event
    zone_name = Column(String, nullable=False)
    geo_boundary_data = Column(JSON, nullable=False)  # list of [lat, lng]
    event = relationship("Event", back_populates="zones")

class AlertRule(Base):
    __tablename__ = "Alert_Rules"
    rule_id = Column(Integer, primary_key=True, index=True)
    zone_id = Column(Integer, ForeignKey("Zones.zone_id"), nullable=False)
    threshold_type = Column(String(100), nullable=False)
    threshold_value = Column(DECIMAL, nullable=False)
    severity_level = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True)

class Alert(Base):
    __tablename__ = "Alerts"
    alert_id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("Events.event_id"), nullable=True)
    zone_id = Column(Integer, ForeignKey("Zones.zone_id"), nullable=True)
    alert_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    status = Column(String, default="Active")
    timestamp = Column(TIMESTAMP, server_default=func.now())
    details = Column(TEXT)
    acknowledged_by = Column(Integer, ForeignKey("Users.user_id"), nullable=True)

class SystemConfiguration(Base):
    __tablename__ = "System_Configurations"
    config_id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String(255), unique=True, nullable=False)
    config_value = Column(TEXT, nullable=False)

class AuditLog(Base):
    __tablename__ = "Audit_Logs"
    log_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("Users.user_id"))
    action = Column(String(255), nullable=False)
    details = Column(TEXT)
    timestamp = Column(TIMESTAMP, server_default=func.now())

# Create Tables
Base.metadata.create_all(bind=engine)

# ----------------------------
# 4. Security & Utils
# ----------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------------
# 5. Pydantic Schemas
# ----------------------------
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class RoleSchema(BaseModel):
    role_id: int
    role_name: str
    class Config:
        orm_mode = True

class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role_id: int

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
    role_ids: Optional[List[int]] = None

class UserSchema(BaseModel):
    user_id: int
    full_name: str
    email: str
    is_active: bool
    roles: List[RoleSchema] = []
    class Config:
        orm_mode = True

class EventCreate(BaseModel):
    event_name: str
    start_timestamp: datetime
    end_timestamp: Optional[datetime] = None

class EventSchema(EventCreate):
    event_id: int
    class Config:
        orm_mode = True

class ZoneCreate(BaseModel):
    zone_name: str
    geo_boundary_data: List[List[float]] # Expecting [[lat, lng], [lat, lng]]

class ZoneSchema(ZoneCreate):
    zone_id: int
    event_id: Optional[int]
    class Config:
        orm_mode = True

class AlertRuleCreate(BaseModel):
    threshold_type: str
    threshold_value: float
    severity_level: str
    is_active: bool = True

class AlertRuleSchema(AlertRuleCreate):
    rule_id: int
    zone_id: int
    class Config:
        orm_mode = True

class AlertSchema(BaseModel):
    alert_id: int
    alert_type: str
    severity: str
    status: str
    details: Optional[str]
    timestamp: datetime
    class Config:
        orm_mode = True

class ManualAlertCreate(BaseModel):
    event_id: int
    details: str
    severity: str
    recipient_role_ids: List[int]

class SystemConfigurationSchema(BaseModel):
    config_key: str
    config_value: str
    class Config:
        orm_mode = True

class AuditLogSchema(BaseModel):
    log_id: int
    user_id: Optional[int]
    action: str
    timestamp: datetime
    class Config:
        orm_mode = True

# ----------------------------
# 6. Auth Dependencies
# ----------------------------
def get_user(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = get_user(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

class RoleChecker:
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_active_user)):
        user_roles = {role.role_name for role in current_user.roles}
        # If user has ANY of the allowed roles
        if not any(role in self.allowed_roles for role in user_roles):
             # For dev purposes, if list is empty or "Admin" bypass (optional)
            if "System Administrator" in user_roles:
                return current_user
                
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user

# Define Permissions
allow_admin = RoleChecker(["System Administrator"])
allow_staff = RoleChecker(["System Administrator", "Event Coordinator", "Security Personnel"])

# ----------------------------
# 7. FastAPI App Setup
# ----------------------------
app = FastAPI(title="Event Alert System")

# CORS (Allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ----------------------------
# 8. WebSockets & YOLO
# ----------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass # Connection likely closed

manager = ConnectionManager()

def is_point_in_zone(lat, lng, zone_poly_data):
    # zone_poly_data is list of [lat, lng]. Shapely needs (x, y) which is (lng, lat)
    poly_points = [(p[1], p[0]) for p in zone_poly_data]
    if len(poly_points) < 3: return False
    point = Point(lng, lat)
    polygon = Polygon(poly_points)
    return polygon.contains(point)

# Load YOLO model (will download on first run)
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Warning: YOLO model failed to load. Ensure internet access or local file. {e}")
    model = None

async def live_monitor():
    """Background task to read camera and broadcast alerts"""
    cap = cv2.VideoCapture(0) # 0 for webcam
    if not cap.isOpened():
        print("Camera not accessible. Live monitoring disabled.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(1)
            continue
        
        # Only run YOLO every 5th frame to save CPU
        # For simplicity in this demo, running every frame but adding sleep
        if model:
            results = model(frame, verbose=False)
            total_count = 0
            zone_counts = {}

            # Create a fresh DB session for this iteration
            with SessionLocal() as db:
                zones = db.query(Zone).all()

                for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                    if int(cls) == 0:  # Class 0 is Person in COCO dataset
                        x1, y1, x2, y2 = box
                        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                        total_count += 1

                        # Simulate mapping screen pixels to Geo-Coordinates (Demo Logic)
                        # Center of screen is mapped to a base Lat/Lng
                        lat = 40.74 + (cy / 10000.0)
                        lng = -73.99 + (cx / 10000.0)

                        for zone in zones:
                            # Parse JSON boundary data
                            try:
                                boundary = json.loads(zone.geo_boundary_data) if isinstance(zone.geo_boundary_data, str) else zone.geo_boundary_data
                                if is_point_in_zone(lat, lng, boundary):
                                    z_name = zone.zone_name
                                    zone_counts[z_name] = zone_counts.get(z_name, 0) + 1
                                    
                                    # Simple threshold check
                                    if zone_counts[z_name] > 5: # Threshold of 5 people
                                        new_alert = Alert(
                                            alert_type="Overcrowding",
                                            severity="High",
                                            zone_id=zone.zone_id,
                                            details=f"Zone {z_name} has {zone_counts[z_name]} people."
                                        )
                                        db.add(new_alert)
                                        db.commit()
                                        
                                        # Notify Frontend
                                        await manager.broadcast({
                                            "type": "ALERT",
                                            "zone": z_name,
                                            "count": zone_counts[z_name],
                                            "severity": "High"
                                        })
                            except Exception as e:
                                print(f"Error processing zone {zone.zone_id}: {e}")

                # Broadcast Heatmap Data
                await manager.broadcast({
                    "type": "HEATMAP",
                    "total_count": total_count,
                    "zone_counts": zone_counts
                })

        await asyncio.sleep(0.5) # Adjust for FPS

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(live_monitor())

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep connection open
    except:
        manager.disconnect(websocket)

# ----------------------------
# 9. API Routes
# ----------------------------

# --- HTML Pages ---
@app.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login_page.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("main_dashboard.html", {"request": request})
# --- HTML Pages ---
@app.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login_page.html", {"request": request})

@app.get("/main_dashboard.html", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("main_dashboard.html", {"request": request})

@app.get("/user_management.html", response_class=HTMLResponse)
def user_management_page(request: Request):
    return templates.TemplateResponse("user_management.html", {"request": request})

@app.get("/zone_management.html", response_class=HTMLResponse)
def zone_management_page(request: Request):
    return templates.TemplateResponse("zone_management.html", {"request": request})

@app.get("/alert_details.html", response_class=HTMLResponse)
def alert_details_page(request: Request):
    return templates.TemplateResponse("alert_details.html", {"request": request})

@app.get("/reports_page.html", response_class=HTMLResponse)
def reports_page(request: Request):
    return templates.TemplateResponse("reports_page.html", {"request": request})

@app.get("/system_settings.html", response_class=HTMLResponse)
def system_settings_page(request: Request):
    return templates.TemplateResponse("system_settings.html", {"request": request})

# --- Auth ---
@app.post("/auth/login")
def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(db, email=form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "user_id": user.user_id}

# --- Users ---
users_router = APIRouter(prefix="/users", tags=["Users"])

@users_router.post("/", response_model=UserSchema)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Note: In real app, restrict this to Admin
    db_user = get_user(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(email=user.email, full_name=user.full_name, password_hash=hashed_password)
    
    # Auto-create role if not exists (for demo)
    role = db.query(Role).filter(Role.role_id == user.role_id).first()
    if not role:
        # Create default roles if missing
        if user.role_id == 1:
            role = Role(role_id=1, role_name="System Administrator")
        else:
            role = Role(role_id=user.role_id, role_name="Staff")
        db.add(role)
        db.commit()
    
    new_user.roles.append(role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@users_router.get("/", response_model=List[UserSchema])
def read_users(db: Session = Depends(get_db), current_user: User = Depends(allow_admin)):
    return db.query(User).all()

# --- Events & Zones ---
events_router = APIRouter(prefix="/events", tags=["Events"])

@events_router.post("/", response_model=EventSchema)
def create_event(event: EventCreate, db: Session = Depends(get_db), user: User = Depends(allow_admin)):
    db_event = Event(**event.dict())
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event

@events_router.post("/{event_id}/zones", response_model=ZoneSchema)
def create_zone(event_id: int, zone: ZoneCreate, db: Session = Depends(get_db), user: User = Depends(allow_admin)):
    # zone.geo_boundary_data is a list. We store it as JSON or directly if DB supports JSON types.
    # SQLite supports JSON via SQLAlchemy generic types easily.
    db_zone = Zone(
        zone_name=zone.zone_name,
        geo_boundary_data=zone.geo_boundary_data, # SQLAlchemy handles JSON conversion
        event_id=event_id
    )
    db.add(db_zone)
    db.commit()
    db.refresh(db_zone)
    return db_zone

# --- Alerts ---
alerts_router = APIRouter(prefix="/alerts", tags=["Alerts"])

@alerts_router.get("/", response_model=List[AlertSchema])
def get_alerts(db: Session = Depends(get_db), user: User = Depends(allow_staff)):
    return db.query(Alert).order_by(Alert.timestamp.desc()).limit(50).all()

@alerts_router.post("/manual")
def create_manual_alert(alert: ManualAlertCreate, db: Session = Depends(get_db), user: User = Depends(allow_admin)):
    db_alert = Alert(
        event_id=alert.event_id,
        alert_type="Manual Broadcast",
        severity=alert.severity,
        details=alert.details
    )
    db.add(db_alert)
    db.commit()
    return {"status": "Alert Broadcasted"}

# Register Routers
app.include_router(users_router)
app.include_router(events_router)
app.include_router(alerts_router)

if __name__ == "__main__":
    # Ensure dummy template exists so app doesn't crash on '/'
    if not os.path.exists("templates/login_page.html"):
        with open("templates/login_page.html", "w") as f:
            f.write("<h1>Login Page (Placeholder)</h1>")
    
    if not os.path.exists("templates/main_dashboard.html"):
        with open("templates/main_dashboard.html", "w") as f:
            f.write("<h1>Dashboard (Placeholder)</h1>")

    uvicorn.run(app, host="127.0.0.1", port=8000)