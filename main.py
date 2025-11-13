import os
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey, Float, Boolean, Text
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-change")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 12

# Prefer MySQL URL from environment; fall back to SQLite for demo if not set
MYSQL_URL = os.getenv("MYSQL_URL") or os.getenv("DATABASE_URL")
if MYSQL_URL:
    SQLALCHEMY_DATABASE_URL = MYSQL_URL
else:
    # Fallback for ephemeral demo; replace with MySQL URL in production
    SQLALCHEMY_DATABASE_URL = "sqlite:///./hotel_ops.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class Branch(Base):
    __tablename__ = "branches"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    location = Column(String(255), nullable=False)
    currency = Column(String(10), default="USD")
    created_at = Column(DateTime, default=datetime.utcnow)

    employees = relationship("Employee", back_populates="branch")
    payments = relationship("Payment", back_populates="branch")
    maintenance_requests = relationship("MaintenanceRequest", back_populates="branch")


class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="staff")  # staff | manager | finance | admin
    hashed_password = Column(String(255), nullable=False)
    branch_id = Column(Integer, ForeignKey("branches.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    branch = relationship("Branch", back_populates="employees")
    schedules = relationship("Schedule", back_populates="employee")


class Schedule(Base):
    __tablename__ = "schedules"
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"))
    shift_date = Column(DateTime, nullable=False)
    shift_start = Column(String(10), nullable=False)  # e.g., 09:00
    shift_end = Column(String(10), nullable=False)    # e.g., 17:00
    notes = Column(String(255), nullable=True)

    employee = relationship("Employee", back_populates="schedules")


class Payment(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True, index=True)
    branch_id = Column(Integer, ForeignKey("branches.id"))
    service_name = Column(String(255), nullable=False)  # Non-room services
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    status = Column(String(20), default="pending")  # pending, completed, failed, refunded
    reference = Column(String(255), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    branch = relationship("Branch", back_populates="payments")


class MaintenanceRequest(Base):
    __tablename__ = "maintenance_requests"
    id = Column(Integer, primary_key=True, index=True)
    branch_id = Column(Integer, ForeignKey("branches.id"))
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    priority = Column(String(20), default="medium")  # low, medium, high
    status = Column(String(20), default="open")  # open, in_progress, completed
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    branch = relationship("Branch", back_populates="maintenance_requests")


class LoyaltyMember(Base):
    __tablename__ = "loyalty_members"
    id = Column(Integer, primary_key=True, index=True)
    member_type = Column(String(20), default="guest")  # guest or staff
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    branch_id = Column(Integer, ForeignKey("branches.id"), nullable=True)
    points_balance = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class LoyaltyTransaction(Base):
    __tablename__ = "loyalty_transactions"
    id = Column(Integer, primary_key=True, index=True)
    member_id = Column(Integer, ForeignKey("loyalty_members.id"))
    branch_id = Column(Integer, ForeignKey("branches.id"), nullable=True)
    points = Column(Integer, nullable=False)  # positive for earn, negative for redeem
    reason = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str
    role: str = Field(default="staff", pattern="^(staff|manager|finance|admin)$")
    branch_id: Optional[int] = None


class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: str
    branch_id: Optional[int]
    is_active: bool

    class Config:
        from_attributes = True


class BranchIn(BaseModel):
    name: str
    location: str
    currency: str = "USD"


class BranchOut(BaseModel):
    id: int
    name: str
    location: str
    currency: str

    class Config:
        from_attributes = True


class ScheduleIn(BaseModel):
    employee_id: int
    shift_date: datetime
    shift_start: str
    shift_end: str
    notes: Optional[str] = None


class ScheduleOut(BaseModel):
    id: int
    employee_id: int
    shift_date: datetime
    shift_start: str
    shift_end: str
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class PaymentIn(BaseModel):
    branch_id: int
    service_name: str
    amount: float
    currency: str = "USD"
    status: str = "pending"
    reference: Optional[str] = None


class PaymentOut(BaseModel):
    id: int
    branch_id: int
    service_name: str
    amount: float
    currency: str
    status: str
    reference: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class MaintenanceIn(BaseModel):
    branch_id: int
    title: str
    description: Optional[str] = None
    priority: str = Field(default="medium", pattern="^(low|medium|high)$")


class MaintenanceUpdate(BaseModel):
    status: Optional[str] = Field(default=None, pattern="^(open|in_progress|completed)$")
    resolved: Optional[bool] = None


class MaintenanceOut(BaseModel):
    id: int
    branch_id: int
    title: str
    description: Optional[str]
    priority: str
    status: str
    created_at: datetime
    resolved_at: Optional[datetime]

    class Config:
        from_attributes = True


class LoyaltyMemberIn(BaseModel):
    member_type: str = Field(default="guest", pattern="^(guest|staff)$")
    full_name: str
    email: EmailStr
    branch_id: Optional[int] = None


class LoyaltyMemberOut(BaseModel):
    id: int
    member_type: str
    full_name: str
    email: EmailStr
    branch_id: Optional[int]
    points_balance: int

    class Config:
        from_attributes = True


class LoyaltyTxnIn(BaseModel):
    member_id: int
    branch_id: Optional[int] = None
    points: int
    reason: str


class LoyaltyTxnOut(BaseModel):
    id: int
    member_id: int
    branch_id: Optional[int]
    points: int
    reason: str
    created_at: datetime

    class Config:
        from_attributes = True


# -----------------------------------------------------------------------------
# Auth utilities
# -----------------------------------------------------------------------------
class TokenData(BaseModel):
    email: Optional[str] = None


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user_by_email(db: Session, email: str) -> Optional[Employee]:
    return db.query(Employee).filter(Employee.email == email).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[Employee]:
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> Employee:
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
    user = get_user_by_email(db, token_data.email)  # type: ignore[arg-type]
    if user is None:
        raise credentials_exception
    return user


def require_roles(*roles: str):
    def role_dep(user: Employee = Depends(get_current_user)):
        if user.role not in roles and user.role != "admin":
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return role_dep

# -----------------------------------------------------------------------------
# FastAPI init
# -----------------------------------------------------------------------------
app = FastAPI(title="Hotel Operations API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/")
def root():
    return {"message": "Hotel Operations API running", "db": "mysql" if MYSQL_URL else "sqlite"}

# -----------------------------------------------------------------------------
# Auth routes
# -----------------------------------------------------------------------------
@app.post("/auth/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db), _: Employee = Depends(require_roles("admin"))):
    if db.query(Employee).filter(Employee.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if user.branch_id:
        if not db.query(Branch).filter(Branch.id == user.branch_id).first():
            raise HTTPException(status_code=404, detail="Branch not found")
    new_user = Employee(
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        hashed_password=get_password_hash(user.password),
        branch_id=user.branch_id,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}

# -----------------------------------------------------------------------------
# Branches
# -----------------------------------------------------------------------------
@app.post("/branches", response_model=BranchOut)
def create_branch(data: BranchIn, db: Session = Depends(get_db), _: Employee = Depends(require_roles("manager", "finance", "admin"))):
    branch = Branch(name=data.name, location=data.location, currency=data.currency)
    db.add(branch)
    db.commit()
    db.refresh(branch)
    return branch


@app.get("/branches", response_model=List[BranchOut])
def list_branches(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), _: Employee = Depends(get_current_user)):
    return db.query(Branch).offset(skip).limit(limit).all()

# -----------------------------------------------------------------------------
# Employees & Schedules
# -----------------------------------------------------------------------------
@app.get("/employees", response_model=List[UserOut])
def list_employees(branch_id: Optional[int] = None, db: Session = Depends(get_db), _: Employee = Depends(require_roles("manager", "finance", "admin"))):
    q = db.query(Employee)
    if branch_id:
        q = q.filter(Employee.branch_id == branch_id)
    return q.all()


@app.post("/schedules", response_model=ScheduleOut)
def create_schedule(data: ScheduleIn, db: Session = Depends(get_db), _: Employee = Depends(require_roles("manager", "admin"))):
    if not db.query(Employee).filter(Employee.id == data.employee_id).first():
        raise HTTPException(status_code=404, detail="Employee not found")
    sched = Schedule(**data.model_dump())
    db.add(sched)
    db.commit()
    db.refresh(sched)
    return sched


@app.get("/schedules", response_model=List[ScheduleOut])
def list_schedules(employee_id: Optional[int] = None, branch_id: Optional[int] = None, db: Session = Depends(get_db), _: Employee = Depends(get_current_user)):
    q = db.query(Schedule)
    if employee_id:
        q = q.filter(Schedule.employee_id == employee_id)
    if branch_id:
        q = q.join(Employee).filter(Employee.branch_id == branch_id)
    return q.order_by(Schedule.shift_date.desc()).all()

# -----------------------------------------------------------------------------
# Payments (non-booking)
# -----------------------------------------------------------------------------
@app.post("/payments", response_model=PaymentOut)
def create_payment(data: PaymentIn, db: Session = Depends(get_db), _: Employee = Depends(require_roles("finance", "manager", "admin"))):
    if not db.query(Branch).filter(Branch.id == data.branch_id).first():
        raise HTTPException(status_code=404, detail="Branch not found")
    ref = data.reference or f"PAY-{int(datetime.utcnow().timestamp())}-{data.branch_id}"
    payment = Payment(
        branch_id=data.branch_id,
        service_name=data.service_name,
        amount=data.amount,
        currency=data.currency,
        status=data.status,
        reference=ref,
    )
    db.add(payment)
    db.commit()
    db.refresh(payment)
    return payment


@app.get("/payments", response_model=List[PaymentOut])
def list_payments(branch_id: Optional[int] = None, status_filter: Optional[str] = None, db: Session = Depends(get_db), _: Employee = Depends(require_roles("finance", "manager", "admin"))):
    q = db.query(Payment)
    if branch_id:
        q = q.filter(Payment.branch_id == branch_id)
    if status_filter:
        q = q.filter(Payment.status == status_filter)
    return q.order_by(Payment.created_at.desc()).all()

# -----------------------------------------------------------------------------
# Maintenance & Alerts
# -----------------------------------------------------------------------------
@app.post("/maintenance", response_model=MaintenanceOut)
def create_maintenance(data: MaintenanceIn, db: Session = Depends(get_db), _: Employee = Depends(require_roles("staff", "manager", "admin"))):
    if not db.query(Branch).filter(Branch.id == data.branch_id).first():
        raise HTTPException(status_code=404, detail="Branch not found")
    req = MaintenanceRequest(
        branch_id=data.branch_id,
        title=data.title,
        description=data.description,
        priority=data.priority,
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    return req


@app.get("/maintenance", response_model=List[MaintenanceOut])
def list_maintenance(branch_id: Optional[int] = None, status_filter: Optional[str] = None, db: Session = Depends(get_db), _: Employee = Depends(get_current_user)):
    q = db.query(MaintenanceRequest)
    if branch_id:
        q = q.filter(MaintenanceRequest.branch_id == branch_id)
    if status_filter:
        q = q.filter(MaintenanceRequest.status == status_filter)
    return q.order_by(MaintenanceRequest.created_at.desc()).all()


@app.patch("/maintenance/{req_id}", response_model=MaintenanceOut)
def update_maintenance(req_id: int, data: MaintenanceUpdate, db: Session = Depends(get_db), _: Employee = Depends(require_roles("manager", "admin"))):
    req = db.query(MaintenanceRequest).filter(MaintenanceRequest.id == req_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    if data.status:
        req.status = data.status
        if data.status == "completed":
            req.resolved_at = datetime.utcnow()
    if data.resolved:
        req.status = "completed"
        req.resolved_at = datetime.utcnow()
    db.commit()
    db.refresh(req)
    return req

# -----------------------------------------------------------------------------
# Loyalty & Rewards
# -----------------------------------------------------------------------------
@app.post("/loyalty/members", response_model=LoyaltyMemberOut)
def add_member(data: LoyaltyMemberIn, db: Session = Depends(get_db), _: Employee = Depends(require_roles("staff", "manager", "admin"))):
    if db.query(LoyaltyMember).filter(LoyaltyMember.email == data.email).first():
        raise HTTPException(status_code=400, detail="Member already exists")
    member = LoyaltyMember(**data.model_dump())
    db.add(member)
    db.commit()
    db.refresh(member)
    return member


@app.get("/loyalty/members", response_model=List[LoyaltyMemberOut])
def list_members(branch_id: Optional[int] = None, db: Session = Depends(get_db), _: Employee = Depends(get_current_user)):
    q = db.query(LoyaltyMember)
    if branch_id:
        q = q.filter(LoyaltyMember.branch_id == branch_id)
    return q.all()


@app.post("/loyalty/transactions", response_model=LoyaltyTxnOut)
def add_loyalty_txn(data: LoyaltyTxnIn, db: Session = Depends(get_db), _: Employee = Depends(require_roles("staff", "manager", "admin"))):
    member = db.query(LoyaltyMember).filter(LoyaltyMember.id == data.member_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    member.points_balance += data.points
    txn = LoyaltyTransaction(**data.model_dump())
    db.add(txn)
    db.commit()
    db.refresh(txn)
    return txn


@app.get("/loyalty/transactions", response_model=List[LoyaltyTxnOut])
def list_loyalty_txns(member_id: Optional[int] = None, branch_id: Optional[int] = None, db: Session = Depends(get_db), _: Employee = Depends(get_current_user)):
    q = db.query(LoyaltyTransaction)
    if member_id:
        q = q.filter(LoyaltyTransaction.member_id == member_id)
    if branch_id:
        q = q.filter(LoyaltyTransaction.branch_id == branch_id)
    return q.order_by(LoyaltyTransaction.created_at.desc()).all()

# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
@app.get("/reports/branch/{branch_id}")
def branch_report(branch_id: int, db: Session = Depends(get_db), _: Employee = Depends(require_roles("manager", "finance", "admin"))):
    branch = db.query(Branch).filter(Branch.id == branch_id).first()
    if not branch:
        raise HTTPException(status_code=404, detail="Branch not found")
    payments_sum = db.query(Payment).filter(Payment.branch_id == branch_id, Payment.status == "completed").all()
    revenue = sum(p.amount for p in payments_sum)
    maintenance_open = db.query(MaintenanceRequest).filter(MaintenanceRequest.branch_id == branch_id, MaintenanceRequest.status != "completed").count()
    members = db.query(LoyaltyMember).filter(LoyaltyMember.branch_id == branch_id).count()
    return {
        "branch": {"id": branch.id, "name": branch.name, "location": branch.location, "currency": branch.currency},
        "revenue_completed": revenue,
        "open_maintenance": maintenance_open,
        "loyalty_members": members,
    }


@app.get("/reports/finance-summary")
def finance_summary(db: Session = Depends(get_db), _: Employee = Depends(require_roles("finance", "admin"))):
    payments = db.query(Payment).all()
    completed = sum(p.amount for p in payments if p.status == "completed")
    pending = sum(p.amount for p in payments if p.status == "pending")
    failed = sum(p.amount for p in payments if p.status == "failed")
    by_currency = {}
    for p in payments:
        by_currency.setdefault(p.currency, 0.0)
        by_currency[p.currency] += p.amount
    return {
        "total_payments": len(payments),
        "completed_amount": completed,
        "pending_amount": pending,
        "failed_amount": failed,
        "by_currency": by_currency,
    }


# Utility endpoint to create an initial admin if none exists
@app.post("/setup/seed-admin")
def seed_admin(email: EmailStr, password: str, full_name: str = "Admin", db: Session = Depends(get_db)):
    if db.query(Employee).filter(Employee.role == "admin").first():
        raise HTTPException(status_code=400, detail="Admin already exists")
    admin = Employee(
        email=email,
        full_name=full_name,
        role="admin",
        hashed_password=get_password_hash(password),
    )
    db.add(admin)
    db.commit()
    return {"message": "Admin created"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
