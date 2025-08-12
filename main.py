import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
import json # Added json import

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn

# Ensure your models and auth are correctly imported
from Backend.app.api.routes import auth
from Backend.app.api.routes.auth import get_current_user
from Backend.database import models
from Backend.database.models import SessionLocal, Career, Recommendation, Quiz

class SemanticRecommender:
    """
    Loads dataset, builds embeddings for career items, and exposes a recommend() method.
    Can be saved and loaded using joblib for persistence.
    """

    def __init__(self, df: pd.DataFrame, model_name="all-MiniLM-L6-v2"):
        # The recommender now takes a DataFrame directly
        self.df = df
        self.model_name = model_name
        self.cache_dir = "cache" # Use a fixed cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Prepare combined text for each career (description + skills + personality)
        self.df["combined_text"] = self.df.apply(
            lambda row: " ".join([
                str(row.get("description", "")),
                str(row.get("skills", "")),
                str(row.get("personality_match", ""))
            ]),
            axis=1
        )

        # Load sentence-transformers model
        print("Loading sentence-transformers model:", self.model_name)
        self.model = SentenceTransformer(self.model_name)

        # Load or compute career embeddings
        self.embeddings_path = os.path.join(self.cache_dir, "career_embeddings.npy")
        self.titles_path = os.path.join(self.cache_dir, "career_titles.pkl")

        if os.path.exists(self.embeddings_path) and os.path.exists(self.titles_path):
            try:
                self.career_embeddings = np.load(self.embeddings_path)
                self.career_titles = joblib.load(self.titles_path)
                # sanity check: same length as df
                if len(self.career_embeddings) != len(self.df):
                    raise ValueError("Cached embeddings length mismatch; recomputing.")
                print("Loaded cached career embeddings.")
            except Exception as e:
                print("Failed to load cache - recomputing embeddings:", e)
                self._compute_and_cache_embeddings()
        else:
            self._compute_and_cache_embeddings()

    def _compute_and_cache_embeddings(self):
        texts = self.df["combined_text"].tolist()
        print(f"Computing embeddings for {len(texts)} careers...")
        self.career_embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        np.save(self.embeddings_path, self.career_embeddings)
        joblib.dump(self.df["career_title"].tolist(), self.titles_path)
        print("Saved career embeddings to cache.")

    def recommend(self, quiz_answers_text: str, top_n: int = 5):
        """
        Given quiz answers as a text string, return top_n career recommendations with metadata and similarity score.
        """
        # Create user embedding
        user_emb = self.model.encode([quiz_answers_text], convert_to_numpy=True)
        # Compute cosine similarities
        sims = cosine_similarity(user_emb, self.career_embeddings)[0]  # (N,)
        # Get top indices
        top_idx = sims.argsort()[-top_n:][::-1]

        recommendations = []
        for idx in top_idx:
            row = self.df.iloc[idx]
            item = {
                "career_title": row["career_title"],
                "description": row.get("description", ""),
                "skills": row.get("skills", ""),
                "personality_match": row.get("personality_match", ""),
                "education_required": row.get("education_required", ""),
                "average_salary_usd": float(row.get("average_salary_usd", 0) or 0),
                "job_outlook": row.get("job_outlook", ""),
                "learning_resources": row.get("learning_resources", ""),
                "similarity_score": float(sims[idx])
            }
            recommendations.append(item)
        return recommendations


# Initialize app
app = FastAPI(title="AI Career Recommendation (Semantic + Auth)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Schemas
class RecommendRequest(BaseModel):
    quiz_answers: str
    top_n: int = 5

class CareerItem(BaseModel):
    career_title: str
    description: str
    skills: str
    personality_match: str
    education_required: str
    average_salary_usd: float
    job_outlook: str
    learning_resources: str
    similarity_score: float

class RecommendResponse(BaseModel):
    recommendations: List[CareerItem]

class UserCreate(BaseModel):
    username: str
    email: str # Added email field
    password: str

class ProfileCreate(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: str
    gender: str
    bio: str
    location: str
    profile_picture: str


class CareerBase(BaseModel):
    name: str
    description: str
    salary: Optional[float] = None
    resources: Optional[Dict] = None

class CareerCreate(CareerBase):
    pass

class CareerRead(CareerBase):
    id: int
    class Config:
        from_attributes = True


class CourseBase(BaseModel):
    career_id: int
    title: str
    link: str

class CourseCreate(CourseBase):
    pass

class CourseRead(CourseBase):
    id: int
    class Config:
        from_attributes = True

# ----------------------------------------
# UPDATED GIG SCHEMAS TO MATCH YOUR MODEL
# ----------------------------------------
class GigBase(BaseModel):
    title: str  # Changed from gig_title
    company: str
    description: str
    budget_min_usd: Optional[float] = None
    budget_max_usd: Optional[float] = None
    duration_weeks: Optional[int] = None
    location: Optional[str] = None  # Updated to be optional as per your model
    applicants_count: Optional[int] = None
    required_skills: Optional[str] = None
    category: Optional[str] = None
    posted_hours_ago: Optional[int] = None
    url: str
    career_id: int

class GigCreate(GigBase):
    pass

class GigRead(GigBase):
    id: int
    class Config:
        from_attributes = True
# ----------------------------------------


class QuizResponseBase(BaseModel):
    user_id: int
    answers: str # Change to string to match quiz_answers field

class QuizResponseCreate(QuizResponseBase):
    pass

class QuizResponseRead(QuizResponseBase):
    id: int
    class Config:
        from_attributes = True


# Routes
@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(auth.get_db)):
    db_user = auth.get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_pw = auth.get_password_hash(user.password)
    # Pass the email to the models.User object
    new_user = models.User(username=user.username, email=user.email, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

@app.post("/profile")
# create endpoint for first name, last name, bio
def create_profile(user: ProfileCreate, db: Session = Depends(auth.get_db)):
    db_user = auth.get_user(db, username=user.username)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    profile = models.Profile(
        user_id=db_user.id,
        first_name=user.first_name,
        last_name=user.last_name,
        date_of_birth=user.date_of_birth)
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return {"message": "Profile created successfully"}

@app.get("/profile")
def get_profile(current_user: models.User = Depends(get_current_user), db: Session = Depends(auth.get_db)):
    profile = db.query(models.Profile).filter(models.Profile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {
        "first_name": profile.first_name,
        "last_name": profile.last_name,
        "date_of_birth": profile.date_of_birth
    }


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(auth.get_db)):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------
# Career Endpoints
# ---------------------------

@app.post("/careers", response_model=CareerRead)
def create_career(career: CareerCreate, db: Session = Depends(auth.get_db)):
    db_career = models.Career(**career.dict())
    db.add(db_career)
    db.commit()
    db.refresh(db_career)
    return db_career

@app.get("/careers", response_model=List[CareerRead])
def get_careers(db: Session = Depends(auth.get_db)):
    return db.query(models.Career).all()


# ---------------------------
# Course Endpoints
# ---------------------------

@app.post("/courses", response_model=CourseRead)
def create_course(course: CourseCreate, db: Session = Depends(auth.get_db)):
    db_course = models.Course(**course.dict())
    db.add(db_course)
    db.commit()
    db.refresh(db_course)
    return db_course

@app.get("/courses", response_model=List[CourseRead])
def get_courses(db: Session = Depends(auth.get_db)):
    return db.query(models.Course).all()


# ---------------------------
# Gig Endpoints
# ---------------------------

@app.post("/gigs", response_model=GigRead)
def create_gig(gig: GigCreate, db: Session = Depends(auth.get_db)):
    # You may need to adjust how you create the db_gig object based on your
    # models.py and relationships, but this should work for basic insertion.
    db_gig = models.Gig(**gig.dict())
    db.add(db_gig)
    db.commit()
    db.refresh(db_gig)
    return db_gig

@app.get("/gigs", response_model=List[GigRead])
def get_gigs(db: Session = Depends(auth.get_db)):
    return db.query(models.Gig).all()


# ---------------------------
# Recommendation Endpoints
# ---------------------------

@app.post("/recommend", response_model=RecommendResponse)
def recommend(
    payload: RecommendRequest,
    db: Session = Depends(auth.get_db),
    current_user: models.User = Depends(get_current_user)
):
    if not payload.quiz_answers.strip():
        raise HTTPException(status_code=400, detail="quiz_answers required")

    # Fetch career data from the database
    careers_data = db.query(Career).all()
    if not careers_data:
        raise HTTPException(status_code=404, detail="No career data found in the database.")

    # Create a DataFrame from the database query result
    # We map 'name' from the model to 'career_title' for the recommender
    df_data = [{
        'career_title': c.name,
        'description': c.description,
        'skills': '',  # Add skills from a related table if available, or leave empty
        'personality_match': '', # Add personality match if available
        'education_required': '', # Add education if available
        'average_salary_usd': c.salary,
        'job_outlook': '', # Add job outlook if available
        'learning_resources': c.resources,
    } for c in careers_data]
    
    careers_df = pd.DataFrame(df_data)

    # Initialize the recommender with the database-driven DataFrame
    recommender = SemanticRecommender(careers_df)

    # Save quiz
    quiz_entry = Quiz(quiz_answers=payload.quiz_answers, user_id=current_user.id)
    db.add(quiz_entry)
    db.commit()
    db.refresh(quiz_entry)

    recs = recommender.recommend(payload.quiz_answers, top_n=payload.top_n)

    # Save recommendations
    for rec in recs:
        # Convert dictionary to JSON string for database storage
        rec['learning_resources'] = json.dumps(rec.get('learning_resources', {}))
        db.add(Recommendation(
            user_id=current_user.id, quiz_id=quiz_entry.id, **rec
        ))
    db.commit()

    return {"recommendations": recs}


@app.get("/history")
def get_user_history(db: Session = Depends(auth.get_db), current_user: models.User = Depends(get_current_user)):
    history = (
        db.query(Recommendation)
        .filter(Recommendation.user_id == current_user.id)
        .all()
    )
    
    results = []
    for rec in history:
        results.append({
            "career_title": rec.career_title,
            "description": rec.description,
            "salary": rec.average_salary_usd
        })
    return results


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
