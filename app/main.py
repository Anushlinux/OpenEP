# app/main.py
import os
import uuid
import json # For parsing LLM string output that should be JSON
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # If your Swift app runs on a different port during dev

from .models import (
    PDFUploadResponse, ChatQuery, ChatResponse,
    RankedTopicsResponse, ProbableQuestionsResponse, ConceptExplanationResponse
)
from .document_processor import extract_text_from_pdf, chunk_text
from .vector_store import add_documents_to_collection, DEFAULT_COLLECTION_NAME
from .agent_workflow import run_agentic_workflow
from .config import CHROMA_PERSIST_DIRECTORY # For ensuring directory exists

# Ensure data directories exist
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads'), exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)


app = FastAPI(title="Exam Prep Agentic RAG API")

# CORS (Cross-Origin Resource Sharing) - Allow your Swift app
# Update origins as needed for production
origins = [
    "http://localhost", # Common for local dev
    "http://localhost:8080", # Example if your Swift UI dev server runs here
    # Add your Swift app's origin when deployed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload-pdf/", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF allowed.")

    pdf_id = str(uuid.uuid4())
    file_location = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads', f"{pdf_id}_{file.filename}")

    try:
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # 1. Process PDF
        print(f"Extracting text from {file_location}...")
        text = extract_text_from_pdf(file_location)
        print(f"Text extracted. Length: {len(text)}")

        chunks = chunk_text(text)
        print(f"Text chunked into {len(chunks)} chunks.")

        # 2. Add to Vector Store
        # We'll create a unique collection per PDF to keep things simple and isolated.
        # The collection name will be pdf_id + default_collection_name suffix.
        pdf_collection_name = f"{pdf_id}_{DEFAULT_COLLECTION_NAME}"
        add_documents_to_collection(collection_name=pdf_collection_name, doc_chunks=chunks, pdf_id=pdf_id)
        print(f"PDF processed and added to collection: {pdf_collection_name}")

        return PDFUploadResponse(
            message="PDF processed successfully.",
            pdf_id=pdf_id,
            filename=file.filename
        )
    except Exception as e:
        # Clean up saved file if processing fails
        if os.path.exists(file_location):
            os.remove(file_location)
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        # Optionally, remove the uploaded PDF after processing if it's no longer needed locally
        # For debugging, you might want to keep it.
        # if os.path.exists(file_location):
        #     os.remove(file_location)
        pass


@app.post("/chat/", response_model=ChatResponse)
async def chat_with_agent(query: ChatQuery = Body(...)):
    try:
        print(f"Received chat query for PDF ID {query.pdf_id}: {query.query}")
        
        # Run the agentic workflow
        result = run_agentic_workflow(
            pdf_id=query.pdf_id,
            query=query.query,
            chat_history=query.chat_history or []
        )
        
        response_type = result.get("response_type", "error")
        data_payload = result.get("data", {})
        final_text = ""  # Initialize the final text response
        
        # Process data_payload if it's raw JSON string
        if 'raw_generation' in data_payload and isinstance(data_payload['raw_generation'], str):
            try:
                parsed_generation = json.loads(data_payload['raw_generation'])
                
                if response_type == "ranked_topics":
                    data_payload = RankedTopicsResponse(**parsed_generation).dict()
                    
                    # Format topics with better structure and markdown
                    topics_text = ["# 📚 Priority Topics for Your Exam\n"]
                    
                    # Group topics by priority level
                    priority_groups = {}
                    for topic in parsed_generation.get("topics", []):
                        priority = topic.get('priority', 5)  # Default to lowest priority if missing
                        if priority not in priority_groups:
                            priority_groups[priority] = []
                        priority_groups[priority].append(topic)
                        
                    # Add topics by priority groups
                    for priority in sorted(priority_groups.keys()):
                        priority_label = "🔴 Highest" if priority == 1 else "🟠 High" if priority == 2 else "🟡 Medium" if priority == 3 else "🟢 Lower" if priority == 4 else "🔵 Background"
                        topics_text.append(f"\n## Priority {priority}: {priority_label}\n")
                        
                        for topic in priority_groups[priority]:
                            topics_text.append(f"### {topic['name']}\n")
                            topics_text.append(f"**Why it matters**: {topic['reasoning']}\n")
                            
                    final_text = "\n".join(topics_text)
                    
                elif response_type == "probable_questions":
                    data_payload = ProbableQuestionsResponse(**parsed_generation).dict()
                    
                    # Format questions with better structure and markdown
                    questions_text = ["# 📝 Probable Exam Questions\n"]
                    questions_text.append("*Based on your study materials, these questions are likely to appear:*\n")
                    
                    # Group questions by source if available
                    source_groups = {"No specific source": []}
                    for q in parsed_generation.get("questions", []):
                        source = q.get('source_hint', "No specific source")
                        if source not in source_groups:
                            source_groups[source] = []
                        source_groups[source].append(q)
                    
                    # Add questions by source groups
                    for i, (source, questions) in enumerate(source_groups.items()):
                        if questions:  # Only add sections with questions
                            if source != "No specific source":
                                questions_text.append(f"\n## From {source}:\n")
                            elif i > 0:  # Only add this header if there are other sections
                                questions_text.append(f"\n## Additional Questions:\n")
                            
                            for j, q in enumerate(questions, 1):
                                questions_text.append(f"{j}. **{q['question']}**\n")
                            
                    final_text = "\n".join(questions_text)
                    
                elif response_type == "explanation":
                    data_payload = ConceptExplanationResponse(explanation=data_payload['raw_generation']).dict()
                    
                    # Format explanation with better structure, adding title and sections
                    explanation = data_payload.get('explanation', data_payload['raw_generation'])
                    title_candidate = explanation.split('.')[0].strip()
                    title = title_candidate if len(title_candidate) <= 60 else "Concept Explanation"
                        
                    formatted_text = [
                        f"# 🔍 {title}\n",
                        explanation
                    ]
                    final_text = "\n".join(formatted_text)
                    
            except json.JSONDecodeError:
                print(f"Warning: Could not parse raw_generation for {response_type} as JSON.")
                if response_type == "explanation":
                    data_payload = ConceptExplanationResponse(explanation=data_payload.get('raw_generation', "")).dict()
                    final_text = f"# 🔍 Explanation\n\n{data_payload.get('explanation', '')}"
                else:
                    data_payload = {"raw_text": data_payload.get('raw_generation', "")}
                    final_text = f"# Response\n\n{data_payload.get('raw_text', '')}"
            except Exception as pydantic_e:
                print(f"Warning: Pydantic validation error for {response_type}: {pydantic_e}")
                data_payload = {"raw_text": data_payload.get('raw_generation', "Error parsing LLM output.")}
                final_text = f"# Response\n\n{data_payload.get('raw_text', 'Error parsing LLM output.')}"
        else:
            # Handle case when data_payload doesn't have raw_generation
            if response_type == "error":
                final_text = f"# ❌ Error\n\n{data_payload.get('error_message', 'An error occurred')}"
            else:
                # Try to extract meaningful text from data_payload
                if isinstance(data_payload, dict):
                    # Look for likely content fields
                    for field in ['explanation', 'text', 'content', 'message']:
                        if field in data_payload and isinstance(data_payload[field], str):
                            final_text = f"# Response\n\n{data_payload[field]}"
                            break
                    if not final_text:  # If no field was found
                        final_text = f"# Response\n\n{str(data_payload)}"
                else:
                    final_text = f"# Response\n\n{str(data_payload)}"

        return ChatResponse(
            response_type=response_type,
            data=data_payload,
            chat_history=result.get("chat_history", []),
            final_text=final_text
        )

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        error_message = f"Sorry, an error occurred: {str(e)}"
        error_chat_history = query.chat_history or []
        if not error_chat_history or error_chat_history[-1].get("content") != query.query:
            error_chat_history.append({"role": "user", "content": query.query})
        error_chat_history.append({"role": "assistant", "content": error_message})
        
        return ChatResponse(
            response_type="error",
            data={"error_message": str(e)},
            chat_history=error_chat_history,
            final_text=f"# ❌ Error\n\n{error_message}"
        )

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        error_message = f"Sorry, an error occurred: {str(e)}"
        error_chat_history = query.chat_history or []
        if not error_chat_history or error_chat_history[-1].get("content") != query.query:
            error_chat_history.append({"role": "user", "content": query.query})
        error_chat_history.append({"role": "assistant", "content": error_message})
        
        return ChatResponse(
            response_type="error",
            data={"error_message": str(e)},
            chat_history=error_chat_history,
            final_text=error_message
        )

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        error_message = f"Sorry, an error occurred: {str(e)}"
        error_chat_history = query.chat_history or []
        if not error_chat_history or error_chat_history[-1].get("content") != query.query:
            error_chat_history.append({"role": "user", "content": query.query})
        error_chat_history.append({"role": "assistant", "content": error_message})
        
        return ChatResponse(
            response_type="error",
            data={"error_message": str(e)},
            chat_history=error_chat_history,
            final_text=error_message
        )

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        # Ensure chat_history is returned even on error, with the user's last message
        error_chat_history = query.chat_history or []
        if not error_chat_history or error_chat_history[-1].get("content") != query.query:
            error_chat_history.append({"role": "user", "content": query.query})
        error_chat_history.append({"role": "assistant", "content": f"Sorry, an error occurred: {str(e)}"})
        
        return ChatResponse(
            response_type="error",
            data={"error_message": str(e)},
            chat_history=error_chat_history
        )


@app.get("/")
async def root():
    return {"message": "Exam Prep Agentic RAG API is running!"}

