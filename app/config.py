import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ChromaDB settings
CHROMA_PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
# Collection name can be dynamic per user/PDF in a real app, or a general one
DEFAULT_COLLECTION_NAME = "exam_prep_docs"
