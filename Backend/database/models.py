# Backend/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, Date, Boolean, Table
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON


DATABASE_URL = "sqlite:///./career_recommendation.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()

# Association table for the many-to-many relationship between User and Course
user_courses_table = Table(
    'user_courses',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('course_id', ForeignKey('courses.id'), primary_key=True)
)

# Association table for the many-to-many relationship between User and Gig
user_gigs_table = Table(
    'user_gigs',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('gig_id', ForeignKey('gigs.id'), primary_key=True)
)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Profile fields
    first_name: Mapped[str | None] = mapped_column(String)
    last_name: Mapped[str | None] = mapped_column(String)
    date_of_birth: Mapped[str | None] = mapped_column(String)
    gender: Mapped[str | None] = mapped_column(String)
    bio: Mapped[str | None] = mapped_column(String)
    location: Mapped[str | None] = mapped_column(String)
    profile_picture: Mapped[str | None] = mapped_column(String)

    certifications: Mapped[list["Certification"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    quiz_responses: Mapped[list["QuizResponse"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    
    # Many-to-many relationships
    enrolled_courses: Mapped[list["Course"]] = relationship(secondary=user_courses_table, back_populates="users")
    completed_gigs: Mapped[list["Gig"]] = relationship(secondary=user_gigs_table, back_populates="users")

    quizzes: Mapped[list["Quiz"]] = relationship(back_populates="user")
    recommendations: Mapped[list["Recommendation"]] = relationship(back_populates="user")


class Certification(Base):
    __tablename__ = "certifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    
    title: Mapped[str] = mapped_column(String)
    issuer: Mapped[str] = mapped_column(String)
    earned_on: Mapped[datetime] = mapped_column(Date)
    verification_id: Mapped[str | None] = mapped_column(String)
    view_url: Mapped[str | None] = mapped_column(String)
    download_url: Mapped[str | None] = mapped_column(String)

    user: Mapped["User"] = relationship(back_populates="certifications")


class Quiz(Base):
    __tablename__ = "quizzes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    quiz_answers: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    
    user: Mapped["User"] = relationship(back_populates="quizzes")
    recommendations: Mapped[list["Recommendation"]] = relationship(back_populates="quiz")


class Recommendation(Base):
    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    career_title: Mapped[str | None] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(Text)
    skills: Mapped[str | None] = mapped_column(Text)
    personality_match: Mapped[str | None] = mapped_column(String)
    education_required: Mapped[str | None] = mapped_column(String)
    average_salary_usd: Mapped[float | None] = mapped_column(Float)
    job_outlook: Mapped[str | None] = mapped_column(String)
    learning_resources: Mapped[str | None] = mapped_column(Text)
    similarity_score: Mapped[float | None] = mapped_column(Float)

    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    quiz_id: Mapped[int | None] = mapped_column(ForeignKey("quizzes.id"))

    user: Mapped["User"] = relationship(back_populates="recommendations")
    quiz: Mapped["Quiz"] = relationship(back_populates="recommendations")


class Career(Base):
    __tablename__ = "careers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    salary: Mapped[float | None] = mapped_column(Float)
    resources: Mapped[dict | None] = mapped_column(JSON)

    courses: Mapped[list["Course"]] = relationship(back_populates="career", cascade="all, delete")
    gigs: Mapped[list["Gig"]] = relationship(back_populates="career", cascade="all, delete")


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    career_id: Mapped[int] = mapped_column(ForeignKey("careers.id"))

    title: Mapped[str] = mapped_column(String)
    provider: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    tags: Mapped[str | None] = mapped_column(String)
    rating: Mapped[float | None] = mapped_column(Float)
    students_enrolled: Mapped[int | None] = mapped_column(Integer)
    duration_weeks: Mapped[int | None] = mapped_column(Integer)
    cost_type: Mapped[str | None] = mapped_column(String)
    level: Mapped[str | None] = mapped_column(String)
    url: Mapped[str] = mapped_column(String)

    career: Mapped["Career"] = relationship(back_populates="courses")
    users: Mapped[list["User"]] = relationship(secondary=user_courses_table, back_populates="enrolled_courses")


class Gig(Base):
    __tablename__ = "gigs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    career_id: Mapped[int] = mapped_column(ForeignKey("careers.id"))

    title: Mapped[str] = mapped_column(String)
    company: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    budget_min_usd: Mapped[float | None] = mapped_column(Float)
    budget_max_usd: Mapped[float | None] = mapped_column(Float)
    duration_weeks: Mapped[int | None] = mapped_column(Integer)
    location: Mapped[str | None] = mapped_column(String)
    applicants_count: Mapped[int | None] = mapped_column(Integer)
    required_skills: Mapped[str | None] = mapped_column(String)
    category: Mapped[str | None] = mapped_column(String)
    posted_hours_ago: Mapped[int | None] = mapped_column(Integer)
    url: Mapped[str] = mapped_column(String)

    career: Mapped["Career"] = relationship(back_populates="gigs")
    users: Mapped[list["User"]] = relationship(secondary=user_gigs_table, back_populates="completed_gigs")


class QuizResponse(Base):
    __tablename__ = "quiz_responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    answers: Mapped[dict] = mapped_column(JSON)

    user: Mapped["User"] = relationship(back_populates="quiz_responses")


def create_db_tables():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    create_db_tables()