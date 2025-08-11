import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from Backend.database.models import create_db_tables, engine, SessionLocal, Career, Course, Gig

# --- Database Session Setup ---
# SessionLocal is a factory to create new Session objects
# It is important to close the session after use.
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

# --- Main execution block ---
if __name__ == "__main__":
    print("\nStarting data migration...")
    load_data_from_csvs()
    print("\nData migration complete!")
