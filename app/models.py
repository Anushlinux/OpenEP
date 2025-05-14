# app/models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PDFUploadResponse(BaseModel):
    message: str
    pdf_id: str # An identifier for the processed PDF
    filename: str

class ChatQuery(BaseModel):
    pdf_id: str # To associate query with a specific processed PDF
    query: str
    chat_history: Optional[List[Dict[str, str]]] = [] # e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

class Topic(BaseModel):
    name: str
    priority: int # e.g., 1-5 (1 being highest)
    reasoning: Optional[str] = None

class RankedTopicsResponse(BaseModel):
    topics: List[Topic]

class ProbableQuestion(BaseModel):
    question: str
    source_hint: Optional[str] = None # e.g., page number or section

class ProbableQuestionsResponse(BaseModel):
    questions: List[ProbableQuestion]

class ConceptExplanationResponse(BaseModel):
    explanation: str
    analogies: Optional[List[str]] = []
    examples: Optional[List[str]] = []

class ChatResponse(BaseModel):
    response_type: str  # "explanation", "ranked_topics", "probable_questions", "error"
    data: Dict[str, Any]  # Will hold one of the above response models or an error message
    chat_history: List[Dict[str, str]]
    final_text: str  # A plain text version of the response for the frontend
