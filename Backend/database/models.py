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
import json
import random


load_dotenv()
db_path = os.path.join(os.path.dirname(__file__), "../../app.db")
db_path = os.path.abspath(db_path)  # make sure it's an absolute path

DATABASE_URL = f"sqlite:///{db_path}"
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
    skills: Mapped[str | None] = mapped_column(String(255))
    personality_match: Mapped[str | None] = mapped_column(String(255))
    education_required: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    salary: Mapped[float | None] = mapped_column(Float)
    job_outlook: Mapped[str | None] = mapped_column(String(255))
    resources: Mapped[str | None] = mapped_column(String)

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
    # Changed to String to store a comma-separated list of names
    students_enrolled: Mapped[str | None] = mapped_column(String(255))
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
    # Changed to String to include 'weeks' text
    duration_weeks: Mapped[str | None] = mapped_column(String(255))
    location: Mapped[str | None] = mapped_column(String(255))
    # Changed to String to store a comma-separated list of names
    applicants: Mapped[str | None] = mapped_column(String(255))
    required_skills: Mapped[str | None] = mapped_column(String(255))
    category: Mapped[str | None] = mapped_column(String(255))
    posted_hours_ago: Mapped[int | None] = mapped_column(Integer)
    url: Mapped[str] = mapped_column(String(255))
    # New column for status
    status: Mapped[str | None] = mapped_column(String(255))

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
import os
import json
import pandas as pd
import random
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from Backend.database.models import SessionLocal, Career, Gig, Course

def load_data_from_csvs():
    """
    Loads data from careers.csv, gigs.csv, and courses.csv into the database.
    This function assumes the tables are already created.
    """
    session = SessionLocal()

    # List of random names for students
    student_names = [
        "Liam Johnson", "Olivia Smith", "Noah Williams", "Emma Brown", "Oliver Davis",
        "Ava Miller", "Elijah Wilson", "Charlotte Moore", "James Taylor", "Amelia Anderson",
        "Benjamin Thomas", "Isabella Jackson", "Lucas White", "Mia Harris", "Mason Martin",
        "Harper Thompson", "Ethan Garcia", "Evelyn Martinez", "Alexander Robinson",
        "Abigail Clark", "Henry Rodriguez", "Elizabeth Lewis", "Sebastian Lee", "Sofia Walker"
    ]
     
    try:
        # Load Careers data first, as it's the parent table
        print("Loading data from careers.csv...")
        file_path = os.path.join(os.path.dirname(__file__), '../../careers.csv')
        careers_df = pd.read_csv(file_path)
        for index, row in careers_df.iterrows():
            career = Career(
                name=row['career_title'],
                skills=row['skills'],
                personality_match=row['personality_match'],
                education_required=row['education_requirement'],
                description=row['description'],
                salary=row['average_salary_usd'],
                job_outlook=row['job_outlook'],
                resources=row['learning_resources']
            )
            session.add(career)
        session.commit()
        print(f"Successfully loaded {len(careers_df)} careers.")

        # Load Courses data
        print("Loading data from courses.csv...")
        file_path = os.path.join(os.path.dirname(__file__), '../../courses.csv')
        courses_df = pd.read_csv(file_path)
        for index, row in courses_df.iterrows():
            # Find the parent Career based on career_title
            parent_career = session.query(Career).filter_by(name=row['career_title']).first()
            if parent_career:
                # Join the list of tags into a single string
                tags_list = [tag.strip() for tag in row['tags'].split(',')] if pd.notna(row['tags']) else []
                tags_string = ', '.join(tags_list)
                
                # Generate a random number of students enrolled and join them into a string
                num_students = random.randint(0, len(student_names))
                students_list = random.sample(student_names, num_students)
                students_string = ', '.join(students_list)
                
                course = Course(
                    career_id=parent_career.id,
                    title=row['course_title'],
                    provider=row['provider'],
                    description=row['description'],
                    tags=tags_string,
                    rating=row['rating'],
                    students_enrolled=students_string,
                    duration_weeks=row['duration_weeks'],
                    cost_type=row['cost_type'],
                    level=row['level'],
                    url=row['url'],
                )
                session.add(course)
        
        session.commit()  # Commit courses before loading gigs
        print(f"Successfully loaded {len(courses_df)} courses.")
        
        # Load Gigs data
        print("Loading data from gigs.csv...")
        file_path = os.path.join(os.path.dirname(__file__), '../../gigs.csv')
        gigs_df = pd.read_csv(file_path)

        # Define category mapping based on career titles in the dataset
        category_mapping = {
            "Data Scientist": "Data Science", "AI Engineer": "Artificial Intelligence",
            "Cybersecurity Analyst": "Cybersecurity", "UX/UI Designer": "Design",
            "Cloud Solutions Architect": "Cloud Computing", "Blockchain Developer": "Blockchain",
            "Game Developer": "Game Development", "Digital Marketing Specialist": "Digital Marketing",
            "Robotics Engineer": "Robotics", "Mobile App Developer": "Mobile Development",
            "Data Analyst": "Data Analysis", "Full Stack Developer": "Web Development",
            "Product Manager": "Product Management", "Bioinformatics Scientist": "Biotechnology",
            "Embedded Systems Engineer": "Hardware Engineering", "Machine Learning Engineer": "Machine Learning",
            "Network Engineer": "Networking", "DevOps Engineer": "DevOps",
            "AR/VR Developer": "AR/VR", "Ethical Hacker": "Cybersecurity",
            "Full-Stack Developer": "Web Development", "Data Engineer": "Data Engineering",
            "UX Researcher": "User Research", "IoT Developer": "Internet of Things",
            "Cloud Security Engineer": "Cloud Security", "Technical Writer": "Technical Writing",
            "Site Reliability Engineer (SRE)": "Site Reliability", "Financial Analyst": "Finance",
            "Quantum Computing Engineer": "Quantum Computing", "Database Administrator": "Database Management",
            "Aerospace Engineer": "Aerospace", "Human Resources Manager": "Human Resources",
            "Project Manager": "Project Management", "Social Media Manager": "Social Media",
            "Electrical Engineer": "Electrical Engineering", "Business Analyst": "Business Analysis",
            "Robotics Process Automation (RPA) Developer": "Process Automation", "Game Designer": "Game Design",
            "Data Visualization Specialist": "Data Visualization", "Network Security Engineer": "Network Security",
            "Systems Analyst": "Systems Analysis", "Content Strategist": "Content Strategy",
            "UX Writer": "UX Writing", "Podiatrist": "Healthcare",
            "Urban Planner": "Urban Planning", "Dietetic Technician": "Healthcare",
            "Forensic Scientist": "Forensic Science", "Chiropractor": "Healthcare",
            "Copywriter": "Copywriting", "Optometrist": "Healthcare",
            "Industrial Designer": "Industrial Design", "Public Relations Specialist": "Public Relations",
            "Occupational Therapist": "Healthcare", "Linguist": "Linguistics",
            "Animator": "Animation", "Medical Assistant": "Healthcare",
            "Curator": "Arts & Culture", "Biomedical Engineer": "Biomedical Engineering",
            "Forest Ranger": "Environmental Science", "Financial Quantitative Analyst": "Quantitative Finance",
            "Marine Biologist": "Marine Biology", "Travel Agent": "Travel & Tourism",
            "Pharmacist": "Healthcare", "Market Research Analyst": "Market Research",
            "School Counselor": "Education", "Urban Farmer": "Agriculture",
            "Event Planner": "Event Management", "Interior Designer": "Interior Design",
            "Graphic Designer": "Graphic Design"
        }

        # List of random names to use for applicants
        applicant_names = [
            "John Doe", "Jane Smith", "Peter Jones", "Mary Brown", "Alex Williams",
            "Sarah Miller", "Michael Davis", "Jessica Garcia", "David Rodriguez",
            "Laura Wilson", "James Martinez", "Patricia Anderson", "Robert Thomas",
            "Jennifer Jackson", "Charles White", "Linda Harris", "Daniel Martin",
            "Elizabeth Thompson", "Joseph Garcia", "Susan Robinson"
        ]

        gigs_added = 0
        for index, row in gigs_df.iterrows():
            # Find the parent Career based on career_title
            parent_career = session.query(Career).filter_by(name=row['career_title']).first()
            if parent_career:
                # Get category based on career title mapping
                category = category_mapping.get(row['career_title'], "General")
                
                # Generate a random number of applicants (between 0 and the number of names)
                num_applicants = random.randint(0, len(applicant_names))
                # Select a random sample of names and join them into a string
                applicants_list = random.sample(applicant_names, num_applicants)
                applicants_string = ', '.join(applicants_list)
                
                # Generate a random status for the gig
                gig_status = random.choice(["Active", "Completed"])
                
                gig = Gig(
                    career_id=parent_career.id,
                    title=row['gig_title'],
                    company=row['company'],
                    description=row['description'],
                    budget_min_usd=row['budget_min_usd'],
                    budget_max_usd=row['budget_max_usd'],
                    # Append " weeks" to the duration and convert to string
                    duration_weeks=f"{row['duration_weeks']} weeks",
                    location=row['location'],
                    applicants=applicants_string,
                    required_skills=row['required_skills'],
                    category=category,  # Category mapped from career_title
                    posted_hours_ago=random.randint(1, 720),  # Random hours ago between 1-720 (30 days)
                    url=row['url'],
                    status=gig_status
                )
                session.add(gig)
                gigs_added += 1
                print(f"Added gig: {row['gig_title']} with category: {category} and status: {gig_status}")
            else:
                print(f"Warning: No career found for '{row['career_title']}'")
        
        session.commit()  # Commit all gigs
        print(f"Successfully loaded {gigs_added} gigs.")

    except IntegrityError as e:
        session.rollback()
        print(f"Database integrity error: {e}")
        raise
    except Exception as e: 
        session.rollback()
        print(f"An error occurred: {e}")
        raise
    finally:
        session.close()
if __name__ == "__main__":
    # print("Dropping existing database tables and recreating them...")
    # drop_db_tables()
    create_db_tables()
    print("Database tables created successfully.")

    # # Load initial data from CSV files
    load_data_from_csvs()
