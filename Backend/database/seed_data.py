import pandas as pd
import numpy as np
import random

# --- Function to generate gigs data ---
def generate_gigs_data(career_titles):
    """
    Generates a list of dictionaries with sample gigs data
    for each career title.
    """
    gigs_data = []
    companies = ["Upwork", "Fiverr", "Toptal", "Freelancer", "Guru", "PeoplePerHour"]
    locations = ["Remote", "Onsite", "Hybrid"]

    for title in career_titles:
        for i in range(1, 3):  # Generate 2 example gigs per career
            gigs_data.append({
                'gig_title': f'Freelance Project: {title} Expert Needed',
                'company': random.choice(companies),
                'description': f'Seeking a skilled professional in {title} for a project on a popular freelance platform.',
                'budget_min_usd': np.random.randint(500, 2000),
                'budget_max_usd': np.random.randint(2500, 5000),
                'duration_weeks': np.random.randint(2, 8),
                'location': random.choice(locations),
                'required_skills': f'Skill A, Skill B, {title} expertise',
                'url': f'https://example.com/gig/{title.replace(" ", "-").lower()}-{i}',
                'career_title': title
            })
    return gigs_data

# --- Function to generate courses data ---
def generate_courses_data(career_titles):
    """
    Generates a list of dictionaries with sample courses data
    for each career title.
    """
    courses_data = []
    providers = ["Coursera", "Udemy", "edX", "Pluralsight", "LinkedIn Learning"]
    levels = ["Beginner", "Intermediate", "Advanced"]
    cost_types = ["Free", "Premium"]

    for title in career_titles:
        for i in range(1, 3):  # Generate 2 example courses per career
            courses_data.append({
                'course_title': f'The Ultimate {title} Bootcamp',
                'provider': random.choice(providers),
                'description': f'A comprehensive course to master the essentials of {title}.',
                'tags': f'tag1, tag2, {title}',
                'rating': round(random.uniform(4.0, 5.0), 1),
                'students_enrolled': np.random.randint(1000, 50000),
                'duration_weeks': np.random.randint(4, 16),
                'cost_type': random.choice(cost_types),
                'level': random.choice(levels),
                'url': f'https://example.com/course/{title.replace(" ", "-").lower()}-{i}',
                'career_title': title
            })
    return courses_data

# --- Main script ---
if __name__ == "__main__":
    try:
        # Read the original dataset
        df = pd.read_csv(r'C:\Users\ayemi\OneDrive\Documents\Team4\Backend\data\careers_dataset.csv')
        career_titles = df['career_title'].unique()
        
        # 1. Create the careers.csv file
        careers_df = df[['career_title', 'description', 'skills', 'personality_match', 'education_requirement', 'average_salary_usd', 'job_outlook']]
        careers_df.to_csv('careers.csv', index=False)
        print("Generated careers.csv from original data.")
        
        # 2. Generate and save the gigs.csv file
        gigs_data = generate_gigs_data(career_titles)
        gigs_df = pd.DataFrame(gigs_data)
        gigs_df.to_csv('gigs.csv', index=False)
        print("Generated gigs.csv with sample data.")

        # 3. Generate and save the courses.csv file
        courses_data = generate_courses_data(career_titles)
        courses_df = pd.DataFrame(courses_data)
        courses_df.to_csv('courses.csv', index=False)
        print("Generated courses.csv with sample data.")

        print("\nAll files have been successfully created.")

    except FileNotFoundError:
        print("Error: 'unified_career_dataset.csv' not found. Please ensure the file is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
