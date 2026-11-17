import os
import datetime
import bcrypt
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False)
    employee_id = Column(String(64), unique=True, nullable=False, index=True)
    phone = Column(String(32), nullable=False)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    profile_pic = Column(String(255), nullable=True, default="/static/uploads/default_avatar.svg")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "employee_id": self.employee_id,
            "phone": self.phone,
            "email": self.email,
            "profile_pic": self.profile_pic or "/static/uploads/default_avatar.svg",
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else ""
        }

# Database Connection Initialization
raw_pg_url = os.environ.get(
    "DATABASE_URL", 
    "postgresql+psycopg2://postgres:postgres@localhost:5432/approxmac_db"
)
# Normalize connection scheme for Supabase / Heroku / Render (postgres:// -> postgresql+psycopg2://)
if raw_pg_url.startswith("postgres://"):
    PG_URL = raw_pg_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif raw_pg_url.startswith("postgresql://") and "+psycopg2" not in raw_pg_url:
    PG_URL = raw_pg_url.replace("postgresql://", "postgresql+psycopg2://", 1)
else:
    PG_URL = raw_pg_url

SQLITE_URL = "sqlite:///./employees.db"

engine = None
active_db_type = "Unknown"

try:
    # Attempt to connect to PostgreSQL
    pg_engine = create_engine(
        PG_URL, 
        pool_pre_ping=True, 
        connect_args={"connect_timeout": 3}
    )
    with pg_engine.connect() as test_conn:
        pass
    engine = pg_engine
    active_db_type = "PostgreSQL"
    print(f"[Database] Successfully connected to PostgreSQL: {PG_URL.split('@')[-1] if '@' in PG_URL else PG_URL}")
except Exception as e:
    # Fallback to local SQLite if PostgreSQL service is not reachable
    print(f"[Database] PostgreSQL connection failed: {e}")
    print("[Database] Using SQLite fallback (sqlite:///./employees.db) to ensure continuous uptime.")
    engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
    active_db_type = "SQLite (Fallback - Start PostgreSQL & set DATABASE_URL to switch)"

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    print(f"[Database] Initialized tables ({active_db_type}).")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(plain_password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False
