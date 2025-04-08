# main.py
import os
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import json
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_mistralai.embeddings import MistralAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
LANGSMITH_TRACING = True
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "shl_assgn"

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

EMBED_MODEL = "mistral-embed"

# Initialize components
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
embedder = MistralAIEmbeddings(model=EMBED_MODEL)
llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

# Create FastAPI app
app = FastAPI(title="SHL Assessment Recommender API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Request models
class JobDescriptionRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None

# Response models
class Assessment(BaseModel):
    name: str
    url: str
    remote_testing: str
    adaptive_irt: str
    details: str
    test_type: str

class AssessmentResponse(BaseModel):
    recommendations: List[Assessment]

# Helper Functions
def scrape_job_description(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join(soup.get_text().split()[:1000])  # Return first 1000 words
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error scraping URL: {str(e)}")

def rag_query(query_text, top_k=10):
    # Generate embedding for query
    query_embedding = embedder.embed_query(query_text)
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return [match.metadata for match in results.matches]

def format_recommendations_json(context, query):
    prompt = ChatPromptTemplate.from_template("""
    You are an SHL assessment expert. Recommend assessments based on the job requirements.
    
    Job Description: {query}
    
    Available Assessments:
    {context}
    
    Return ONLY a JSON array of assessment objects. Each object should have these properties:
    - name: The assessment name
    - url: The link to the assessment
    - remote_testing: Whether remote testing is available
    - adaptive_irt: Whether it's adaptive/IRT
    - details: Assessment details
    - test_type: The type of test
    
    Include only assessments matching the job requirements. Maximum 10 recommendations.
    Format your response as a valid JSON array with no additional text or explanation.
    """)
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query}).content
    
    # Try to extract JSON if it's embedded in markdown code blocks
    if "```json" in response:
        json_content = response.split("```json")[1].split("```")[0].strip()
        return json.loads(json_content)
    elif "```" in response:
        json_content = response.split("```")[1].strip()
        return json.loads(json_content)
    else:
        return json.loads(response)

# API Endpoints
@app.get("/", tags=["Health"])
def health_check():
    return {"status": "healthy", "message": "SHL Assessment Recommender API is running"}

@app.post("/recommend", response_model=AssessmentResponse, tags=["Recommendations"])
async def get_recommendations(request: JobDescriptionRequest):
    query_text = None
    
    # Check input source
    if request.url:
        query_text = scrape_job_description(request.url)
    elif request.text:
        query_text = request.text
    else:
        raise HTTPException(status_code=400, detail="Either text or URL must be provided")
    
    # Process the query
    results = rag_query(query_text)
    context = "\n".join([
        f"Name: {res['title']}\nURL: {res['link']}\n"
        f"Remote: {res['remote_testing']}\nAdaptive: {res['adaptive_irt']}\n"
        f"Details: {res['details']}\nType: {res['test_type']}"
        for res in results
    ])
    
    # Get recommendations in JSON format
    recommendations = format_recommendations_json(context, query_text)
    
    return {"recommendations": recommendations}

@app.get("/recommend", response_model=AssessmentResponse, tags=["Recommendations"])
async def get_recommendations_get(
    text: Optional[str] = Query(None, description="Job description text"),
    url: Optional[str] = Query(None, description="URL of job posting to scrape")
):
    query_text = None
    
    # Check input source
    if url:
        query_text = scrape_job_description(url)
    elif text:
        query_text = text
    else:
        raise HTTPException(status_code=400, detail="Either text or url parameter must be provided")
    
    # Process the query
    results = rag_query(query_text)
    context = "\n".join([
        f"Name: {res['title']}\nURL: {res['link']}\n"
        f"Remote: {res['remote_testing']}\nAdaptive: {res['adaptive_irt']}\n"
        f"Details: {res['details']}\nType: {res['test_type']}"
        for res in results
    ])
    
    # Get recommendations in JSON format
    recommendations = format_recommendations_json(context, query_text)
    
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)