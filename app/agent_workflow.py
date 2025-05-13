# app/agent_workflow.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Annotated, Sequence
import operator

from .config import GOOGLE_API_KEY
from .vector_store import query_collection, DEFAULT_COLLECTION_NAME
from .models import Topic, ProbableQuestion # For structuring output

# --- LLM Initialization ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)

# --- Agent State ---
class AgentState(TypedDict):
    pdf_id: str
    user_query: str
    chat_history: List[Dict[str, str]] # [ {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."} ]
    retrieved_docs: List[str]
    generation: str # Raw LLM output
    parsed_output: Dict # Structured output (topics, questions, explanation)
    response_type: str # "explanation", "ranked_topics", "probable_questions"
    error_message: str # For any errors during processing

# --- Agent Nodes ---

def route_query_node(state: AgentState):
    """Determines the type of query and routes to the appropriate path."""
    print("---ROUTING QUERY---")
    user_query = state["user_query"].lower()
    
    # Determine route based on query
    if "explain" in user_query or "what is" in user_query or "tell me about" in user_query:
        next_step = "retrieve_for_explanation" 
    elif "priority topics" in user_query or "important topics" in user_query:
        next_step = "retrieve_for_topic_ranking"
    elif "probable questions" in user_query or "exam questions" in user_query:
        next_step = "retrieve_for_question_prediction"
    else:
        next_step = "retrieve_for_explanation"  # Default
    
    # Return a dictionary that updates the state
    return {"response_type": next_step}


def retrieve_documents_node(state: AgentState):
    """Retrieves documents from the vector store based on the user query."""
    print(f"---RETRIEVING DOCUMENTS for PDF: {state['pdf_id']}---")
    # For simplicity, we use the user query directly for retrieval.
    # For topic ranking/question prediction, you might want to retrieve broader context
    # or specific sections if your PDF metadata allows.
    documents = query_collection(
        collection_name=f"{state['pdf_id']}_{DEFAULT_COLLECTION_NAME}", # Assuming one collection per PDF for now
        query_text=state["user_query"],
        n_results=5 
    )
    print(f"Retrieved {len(documents)} documents.")
    return {"retrieved_docs": documents}

def generate_explanation_node(state: AgentState):
    """Generates a concept explanation."""
    print("---GENERATING EXPLANATION---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert tutor. Explain the concept clearly based on the provided context. Use analogies and examples if helpful. Context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{query}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    context_str = "\n\n".join(state["retrieved_docs"])
    
    # Prepare chat history for LangChain format if needed
    # For now, simple pass-through assuming it's compatible or handled by MessagesPlaceholder
    
    generation = chain.invoke({
        "context": context_str,
        "query": state["user_query"],
        "chat_history": [(msg["role"], msg["content"]) for msg in state.get("chat_history", [])]
    })
    return {"generation": generation, "response_type": "explanation"}

def generate_priority_topics_node(state: AgentState):
    """Generates a list of priority-ranked topics."""
    print("---GENERATING PRIORITY TOPICS---")
    # For this, we might want to use a larger context than just query-specific retrieval.
    # E.g., retrieve a significant portion or all chunks from the "study notes" PDF.
    # For now, we use the retrieved_docs which might be query-specific.
    context_str = "\n\n".join(state["retrieved_docs"])
    
    parser = JsonOutputParser(pydantic_object=Topic) # Define what a Topic looks like
    
    prompt_str = """Based on the following study material, identify and rank the top 5-7 most important topics for exam revision.
Provide your reasoning for each topic's priority.
Respond with ONLY a JSON object containing a list of topics. Each topic should have 'name', 'priority' (integer 1-5, 1 is highest), and 'reasoning'.
Example JSON:
{{
  "topics": [
    {{ "name": "Topic A", "priority": 1, "reasoning": "Frequently mentioned and foundational." }},
    {{ "name": "Topic B", "priority": 2, "reasoning": "Appears in multiple complex examples." }}
  ]
}}

Study Material:
{context}

User Request: {query}
"""
    prompt = ChatPromptTemplate.from_template(prompt_str)
    
    # Simple chain for now
    # chain = prompt | llm | StrOutputParser() # If expecting raw JSON string
    
    # For structured output, you might need to experiment with JsonOutputParser
    # Or just parse the string output carefully. For robust JSON, an LLM that reliably outputs JSON is key.
    # Let's try a direct approach and catch errors if JSON is malformed.
    
    chain = prompt | llm | StrOutputParser() # Get string first
    
    raw_json_output = chain.invoke({
        "context": context_str,
        "query": state["user_query"]
    })
    
    # Attempt to parse the JSON from the raw_json_output
    # This part can be tricky and might need iteration on the prompt or a more robust JSON parsing strategy.
    # For now, we'll just pass the raw string and let the FastAPI endpoint try to structure it.
    # A better approach is to use LLM functions/tools if the LLM supports it for structured output.
    return {"generation": raw_json_output, "response_type": "ranked_topics"}


def generate_probable_questions_node(state: AgentState):
    """Generates probable exam questions."""
    print("---GENERATING PROBABLE QUESTIONS---")
    context_str = "\n\n".join(state["retrieved_docs"])
    
    prompt_str = """Based on the following content, which seems to be from past exam papers or core study material, generate 3-5 probable exam questions.
For each question, you can optionally provide a hint about where it might be derived from if evident in the context.
Respond with ONLY a JSON object containing a list of questions. Each question should have 'question' and optionally 'source_hint'.
Example JSON:
{{
  "questions": [
    {{ "question": "What is X?", "source_hint": "Section 1.2" }},
    {{ "question": "Explain Y in detail." }}
  ]
}}

Content:
{context}

User Request: {query}
"""
    prompt = ChatPromptTemplate.from_template(prompt_str)
    chain = prompt | llm | StrOutputParser()
    
    raw_json_output = chain.invoke({
        "context": context_str,
        "query": state["user_query"]
    })
    return {"generation": raw_json_output, "response_type": "probable_questions"}


def format_output_node(state: AgentState):
    """Formats the LLM generation into the Pydantic models for the API response."""
    print("---FORMATTING OUTPUT---")
    response_type = state["response_type"]
    generation = state["generation"]
    
    # This is where you would parse the 'generation' string (which might be JSON)
    # into the structures defined in models.py (RankedTopicsResponse, ProbableQuestionsResponse, etc.)
    # For now, we'll just pass the raw generation and response_type.
    # The FastAPI endpoint will be responsible for trying to fit this into ChatResponse.data
    
    # A more robust implementation would parse the JSON string from 'generation' here
    # and populate the 'parsed_output' field in the state.
    
    # Add current generation to chat history
    updated_chat_history = state.get("chat_history", []) + [{"role": "assistant", "content": generation}]
    
    return {"parsed_output": {"raw_generation": generation}, "chat_history": updated_chat_history}


# --- Build the Graph ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", route_query_node)
workflow.add_node("retriever_explanation", retrieve_documents_node)
workflow.add_node("generator_explanation", generate_explanation_node)
workflow.add_node("retriever_topics", retrieve_documents_node) # Can customize retrieval later
workflow.add_node("generator_topics", generate_priority_topics_node)
workflow.add_node("retriever_questions", retrieve_documents_node) # Can customize retrieval later
workflow.add_node("generator_questions", generate_probable_questions_node)
workflow.add_node("formatter", format_output_node)


# Define edges
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    lambda state: state["response_type"],  # Use the response_type from state
    {
        "retrieve_for_explanation": "retriever_explanation",
        "retrieve_for_topic_ranking": "retriever_topics",
        "retrieve_for_question_prediction": "retriever_questions",
    }
)

workflow.add_edge("retriever_explanation", "generator_explanation")
workflow.add_edge("generator_explanation", "formatter")

workflow.add_edge("retriever_topics", "generator_topics")
workflow.add_edge("generator_topics", "formatter")

workflow.add_edge("retriever_questions", "generator_questions")
workflow.add_edge("generator_questions", "formatter")

workflow.add_edge("formatter", END)

# Compile the graph
agent_graph = workflow.compile()

# --- Main Invocation Function ---
def run_agentic_workflow(pdf_id: str, query: str, chat_history: List[Dict[str, str]] = None) -> Dict:
    if chat_history is None:
        chat_history = []
        
    initial_state = AgentState(
        pdf_id=pdf_id,
        user_query=query,
        chat_history=chat_history,
        retrieved_docs=[],
        generation="",
        parsed_output={},
        response_type="", # Router will set this
        error_message=""
    )
    
    # Add user query to chat history for the current turn
    current_turn_history = chat_history + [{"role": "user", "content": query}]
    initial_state["chat_history"] = current_turn_history

    try:
        # Before invoking, the router needs to run to set response_type for conditional_edges
        # This is a bit of a simplification. LangGraph typically handles this flow more smoothly
        # if the router node itself sets the 'response_type' in the state, which conditional_edges then reads.
        # Let's adjust the router to directly update the state.

        # Manually run the router first to determine the path
        routing_decision = route_query_node(initial_state) # This returns the next node name
        
        # Re-structure agent state based on routing.
        # The actual routing logic for LangGraph is within add_conditional_edges.
        # The router node should just return the key for the conditional edge.
        # The `route_query_node` above is now more like a pre-router.
        # Let's adjust how `route_query_node` works for LangGraph `add_conditional_edges`

        # Corrected approach for conditional_edges:
        # The `route_query_node` should return a dictionary that updates the state,
        # and the `conditional_edge_mapping` function passed to `add_conditional_edges`
        # will use a field from that state (e.g., `state['next_node_key']`)

        # For simplicity in this example, we'll assume the `route_query_node` correctly
        # sets up the state for the `add_conditional_edges` to work.
        # The critical part is that the `route_query_node` must output a dictionary
        # that becomes part of the state, and the conditional function uses that.

        # Let's refine the router and conditional edge for clarity.

        # (Revisiting the router logic for conditional edges)
        # The 'router' node should set a field in the state, e.g., `next_step_key`.
        # The conditional edge function then reads `state['next_step_key']`.

        # For now, we'll assume the graph setup handles this.
        # The critical part in `agent_graph.stream` or `agent_graph.invoke` is that the
        # initial state is passed correctly.

        final_state = {}
        # Using stream to see intermediate steps (optional)
        for s in agent_graph.stream(initial_state, {"recursion_limit": 10}):
            print(f"Current Node: {list(s.keys())[0]}")
            print(f"State: {s[list(s.keys())[0]]}")
            print("----")
            final_state.update(s[list(s.keys())[0]]) # Accumulate state changes

        # Or just invoke for final result
        # final_state = agent_graph.invoke(initial_state, {"recursion_limit": 10})

        return {
            "response_type": final_state.get("response_type", "error"),
            "data": final_state.get("parsed_output", {"raw_generation": final_state.get("generation")}),
            "chat_history": final_state.get("chat_history", current_turn_history)
        }
    except Exception as e:
        print(f"Error in agentic workflow: {e}")
        import traceback
        traceback.print_exc()
        return {
            "response_type": "error",
            "data": {"error_message": str(e)},
            "chat_history": current_turn_history
        }

# Small test (run this file directly if you want to test)
if __name__ == "__main__":
    # You'd need a dummy PDF processed and in ChromaDB for this to fully work.
    # Let's simulate a query to the explanation path.
    
    # This is a placeholder test. Actual testing needs PDF upload and vector store population first.
    # For now, the router is simple keyword based.
    
    # Mock initial state (assuming router determines 'explanation')
    mock_initial_state_for_invoke = AgentState(
        pdf_id="test_pdf_123", # This PDF ID needs to exist in Chroma for retriever
        user_query="Explain photosynthesis.",
        chat_history=[],
        retrieved_docs=["Photosynthesis is a process used by plants.", "It converts light energy into chemical energy."], # Mock retrieved docs
        generation="",
        parsed_output={},
        response_type="explanation", # Manually set for testing a specific path after routing
        error_message=""
    )
    
    # To test a path, you'd skip the router and start from retriever or generator:
    # Test explanation generation
    # result_explanation = generate_explanation_node(mock_initial_state_for_invoke)
    # print("Test Explanation Result:", result_explanation)

    # To test the full graph from a specific point, you'd need to carefully craft the input state.
    # The `run_agentic_workflow` is the intended entry point from FastAPI.
    print("Agent workflow module loaded. Run FastAPI app to test endpoints.")

