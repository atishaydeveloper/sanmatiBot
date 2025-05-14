import os
import logging
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Constants
DB_PATH = "faiss_index"
TEXT_FILE = os.getenv("SCHOOL_DATA_PATH", "attached_assets/main2.md")

def load_and_split_text(file_path):
    """Load the text file and split it into chunks for vectorization."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        logger.debug(f"Loaded and split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error loading or splitting text: {str(e)}")
        raise

def get_or_create_faiss(chunks=None):
    """Initialize or load the FAISS vector store."""
    try:
        # Create a simple in-memory store of text chunks for search
        if chunks is None:
            chunks = load_and_split_text(TEXT_FILE)
        
        # Return the chunks directly - we'll implement a simple search instead of FAISS
        return chunks
    except Exception as e:
        logger.error(f"Error preparing text chunks: {str(e)}")
        raise

def get_gemini_response(context, query):
    """Generate a response using Google's Gemini model."""
    try:
        prompt = f"""
You are Saanvi, an intelligent and friendly AI assistant designed to help parents learn more about Sanmati H.S. School.

Your task is to understand the parent's question and decide:

1. If the question is general in nature (like parenting tips, learning styles, etc.), answer directly without using the school's context.
2. If the question is specifically about Sanmati H.S. School (like admission process, facilities, timings, mission, curriculum, etc.), use the provided school information (context) to generate a detailed and helpful answer.

Always be polite, informative, and supportive in your tone. Use clear and complete sentences. If the answer is about the school, ensure it is based on the provided school information.

Keep your responses concise and suitable for SMS format (under 1600 characters) while still being helpful and complete.

---

Context (use this only if the question is about the school):
{context}

---

Parent's Question:
{query}

---

Saanvi's Response:
"""
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating Gemini response: {str(e)}")
        raise

def simple_search(query, chunks, num_results=3):
    """Simple keyword-based search through text chunks."""
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Score each chunk based on keyword matching
    scored_chunks = []
    for chunk in chunks:
        # Simple scoring: count how many query words appear in the chunk
        score = 0
        for word in query_lower.split():
            if len(word) > 3 and word.lower() in chunk.lower():  # Only consider words longer than 3 chars
                score += 1
        
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Sort by score (highest first) and take top results
    scored_chunks.sort(reverse=True)
    results = [chunk for score, chunk in scored_chunks[:num_results]]
    
    # If we found nothing, return some default chunks
    if not results:
        return chunks[:num_results]
    
    return results

def process_message(message):
    """Process an incoming message and return a response."""
    try:
        # Load text chunks
        chunks = load_and_split_text(TEXT_FILE)
        
        # Perform simple search
        relevant_chunks = simple_search(message, chunks, num_results=3)
        context = "\n\n".join(relevant_chunks)
        
        # Generate response
        response = get_gemini_response(context, message)
        
        # Ensure response is not too long for SMS
        if isinstance(response, str) and len(response) > 1600:
            response = response[:1597] + "..."
            
        return response
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return "I'm sorry, I encountered a technical issue. Please try again later."

def reset_vector_store():
    """Reset function - not needed with the simplified implementation."""
    logger.debug("Reset called - but not needed with simplified implementation")
    return {"message": "System reset successfully."}
