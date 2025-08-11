# Backend/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, Date, Boolean, Table
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os
from dotenv import load_dotenv
from sqlalchemy.exc import IntegrityError
import pandas as pd

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
print(f"Using database URL: {DATABASE_URL}")

engine = create_engine(
    DATABASE_URL
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
    username: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Profile fields
    first_name: Mapped[str | None] = mapped_column(String(255))
    last_name: Mapped[str | None] = mapped_column(String(255))
    date_of_birth: Mapped[str | None] = mapped_column(String(255))
    gender: Mapped[str | None] = mapped_column(String(255))
    bio: Mapped[str | None] = mapped_column(String(255))
    location: Mapped[str | None] = mapped_column(String(255))
    profile_picture: Mapped[str | None] = mapped_column(String(255))

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
    
    title: Mapped[str] = mapped_column(String(255))
    issuer: Mapped[str] = mapped_column(String(255))
    earned_on: Mapped[datetime] = mapped_column(Date)
    verification_id: Mapped[str | None] = mapped_column(String(255))
    view_url: Mapped[str | None] = mapped_column(String(255))
    download_url: Mapped[str | None] = mapped_column(String(255))

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
    career_title: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    skills: Mapped[str | None] = mapped_column(Text)
    personality_match: Mapped[str | None] = mapped_column(String(255))
    education_required: Mapped[str | None] = mapped_column(String(255))
    average_salary_usd: Mapped[float | None] = mapped_column(Float)
    job_outlook: Mapped[str | None] = mapped_column(String(255))
    learning_resources: Mapped[str | None] = mapped_column(Text)
    similarity_score: Mapped[float | None] = mapped_column(Float)

    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    quiz_id: Mapped[int | None] = mapped_column(ForeignKey("quizzes.id"))

    user: Mapped["User"] = relationship(back_populates="recommendations")
    quiz: Mapped["Quiz"] = relationship(back_populates="recommendations")


class Career(Base):
    __tablename__ = "careers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    salary: Mapped[float | None] = mapped_column(Float)
    resources: Mapped[dict | None] = mapped_column(Text)

    courses: Mapped[list["Course"]] = relationship(back_populates="career", cascade="all, delete")
    gigs: Mapped[list["Gig"]] = relationship(back_populates="career", cascade="all, delete")


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    career_id: Mapped[int] = mapped_column(ForeignKey("careers.id"))

    title: Mapped[str] = mapped_column(String(255))
    provider: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(String(255))
    tags: Mapped[str | None] = mapped_column(String(255))
    rating: Mapped[float | None] = mapped_column(Float)
    students_enrolled: Mapped[int | None] = mapped_column(Integer)
    duration_weeks: Mapped[int | None] = mapped_column(Integer)
    cost_type: Mapped[str | None] = mapped_column(String(255))
    level: Mapped[str | None] = mapped_column(String(255))
    url: Mapped[str] = mapped_column(String(255))

    career: Mapped["Career"] = relationship(back_populates="courses")
    users: Mapped[list["User"]] = relationship(secondary=user_courses_table, back_populates="enrolled_courses")


class Gig(Base):
    __tablename__ = "gigs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    career_id: Mapped[int] = mapped_column(ForeignKey("careers.id"))

    title: Mapped[str] = mapped_column(String(255))
    company: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(String(255))
    budget_min_usd: Mapped[float | None] = mapped_column(Float)
    budget_max_usd: Mapped[float | None] = mapped_column(Float)
    duration_weeks: Mapped[int | None] = mapped_column(Integer)
    location: Mapped[str | None] = mapped_column(String(255))
    applicants_count: Mapped[int | None] = mapped_column(Integer)
    required_skills: Mapped[str | None] = mapped_column(String(255))
    category: Mapped[str | None] = mapped_column(String(255))
    posted_hours_ago: Mapped[int | None] = mapped_column(Integer)
    url: Mapped[str] = mapped_column(String(255))

    career: Mapped["Career"] = relationship(back_populates="gigs")
    users: Mapped[list["User"]] = relationship(secondary=user_gigs_table, back_populates="completed_gigs")


class QuizResponse(Base):
    __tablename__ = "quiz_responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    answers: Mapped[dict] = mapped_column(Text)

    user: Mapped["User"] = relationship(back_populates="quiz_responses")


def create_db_tables():
    """Creates all database tables based on the SQLAlchemy models."""
    Base.metadata.create_all(bind=engine)

def drop_db_tables():
    """Drops all database tables based on the SQLAlchemy models."""
    Base.metadata.drop_all(bind=engine)



Session = SessionLocal

# --- Function to load data from CSVs ---
def load_data_from_csvs():
    """
    Loads data from careers.csv, gigs.csv, and courses.csv into the database.
    This function assumes the tables are already created.
    """
    session = Session()
    try:
        # Load Careers data first, as it's the parent table
        print("Loading data from careers.csv...")
        careers_df = pd.read_csv(r'C:\Users\ayemi\OneDrive\Documents\Team4\careers.csv')
        for index, row in careers_df.iterrows():
            career = Career(
                name=row['career_title'],
                description=row['description'],
                salary=row['average_salary_usd'],
                # You may need to handle the list/JSON data from 'resources' and 'skills'
                # For this script, we'll store them as-is.
                resources={}, # Assuming resources is a JSON field and needs an empty dict
            )
            session.add(career)
        session.commit()
        print(f"Successfully loaded {len(careers_df)} careers.")

        # Load Gigs data
        print("Loading data from gigs.csv...")
        gigs_df = pd.read_csv(r'C:\Users\ayemi\OneDrive\Documents\Team4\gigs.csv')
        for index, row in gigs_df.iterrows():
            # Find the parent Career based on career_title
            parent_career = session.query(Career).filter_by(name=row['career_title']).first()
            if parent_career:
                gig = Gig(
                    career_id=parent_career.id,
                    title=row['gig_title'],
                    company=row['company'],
                    description=row['description'],
                    budget_min_usd=row['budget_min_usd'],
                    budget_max_usd=row['budget_max_usd'],
                    duration_weeks=row['duration_weeks'],
                    location=row['location'],
                    required_skills=row['required_skills'],
                    url=row['url'],
                )
                session.add(gig)
        session.commit()
        print(f"Successfully loaded {len(gigs_df)} gigs.")

        # Load Courses data
        print("Loading data from courses.csv...")
        courses_df = pd.read_csv(r'C:\Users\ayemi\OneDrive\Documents\Team4\courses.csv')
        for index, row in courses_df.iterrows():
            # Find the parent Career based on career_title
            parent_career = session.query(Career).filter_by(name=row['career_title']).first()
            if parent_career:
                course = Course(
                    career_id=parent_career.id,
                    title=row['course_title'],
                    provider=row['provider'],
                    description=row['description'],
                    tags=row['tags'],
                    rating=row['rating'],
                    students_enrolled=row['students_enrolled'],
                    duration_weeks=row['duration_weeks'],
                    cost_type=row['cost_type'],
                    level=row['level'],
                    url=row['url'],
                )
                session.add(course)
        session.commit()
        print(f"Successfully loaded {len(courses_df)} courses.")

    except IntegrityError as e:
        session.rollback()
        print(f"Database integrity error: {e}")
    except Exception as e:
        session.rollback()
        print(f"An error occurred: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    print("Dropping existing database tables and recreating them...")
    drop_db_tables()
    create_db_tables()
    print("Database tables created successfully.")

    # Load initial data from CSV files
    load_data_from_csvs()
    