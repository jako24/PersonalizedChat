import os
import pathlib
from dotenv import load_dotenv
import langwatch

load_dotenv()

# Get the project root directory (parent of the app directory)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX_DIR = os.getenv("INDEX_DIR", str(PROJECT_ROOT / "data" / "index"))
CATALOG_FILE = os.getenv("CATALOG_FILE", str(PROJECT_ROOT / "data" / "catalog.txt"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", "8"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.62"))

from typing import Any, Dict, List, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langdetect import detect, LangDetectException
from .catalog_utils import normalize_name, fuzzy_best_match, is_placeholder

def build_and_save_index():
    """Builds and saves the FAISS index."""
    def load_catalog() -> List[str]:
        with open(CATALOG_FILE, "r", encoding="utf-8") as f:
            raw = [line.strip() for line in f.readlines()]
        # Clean, normalize display names (keep original), filter placeholders
        clean: List[str] = []
        seen_norm = set()
        for name in raw:
            if not name:
                continue
            if is_placeholder(name):
                continue
            norm = normalize_name(name)
            if not norm or norm in seen_norm:
                continue
            seen_norm.add(norm)
            clean.append(name.strip())
        if not clean:
            raise RuntimeError("No valid products found in catalog.txt")
        return clean

    def build_docs(names: List[str]) -> List[Document]:
        docs: List[Document] = []
        for n in names:
            norm = normalize_name(n)
            # Content for embedding: emphasize the product name and plausible query variants.
            page_content = f"PRODUCT_NAME: {n}\nALSO_KNOWN_AS: {norm}\nTYPE: ingredient or product\n"
            docs.append(Document(page_content=page_content, metadata={"display_name": n, "normalized_name": norm}))
        return docs

    names = load_catalog()
    print(f"Loaded {len(names)} unique catalog items.")
    docs = build_docs(names)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(docs, embeddings)

    pathlib.Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}")

# Load vectorstore once
_embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
    print("FAISS index not found. Building and saving a new one...")
    build_and_save_index()

_vector = FAISS.load_local(INDEX_DIR, _embeddings, allow_dangerous_deserialization=True)

# Preload all display names for fuzzy fallback
_all_names: List[str] = [d.metadata["display_name"] for d in _vector.docstore._dict.values()]

def _language_for(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"  # Default to English if detection fails



@langwatch.span(type="llm", name="Intent Understanding")
def _understand_intent_with_llm(query: str) -> List[str]:
    """Stage 1: Use LLM to understand user intent and generate relevant search terms"""
    intent_prompt = f"""Analyze this customer request: "{query}""

You're helping a specialty Mexican ingredient and health food store. Generate search terms that match how products are named in our catalog.

Our catalog uses simple product names like:
- "ACHIOTE" not "adobo de achiote"  
- "CHILE GUAJILLO" not "chiles guajillo"
- "PIÑA" not "piña fresca"
- "SAZONADOR PASTOR" not "especias para pastor"
- "VINAGRE MANZANA" not "vinagre de manzana"

Generate 4-6 short, direct search terms that would find products for this request:
- Simple ingredient names (achiote, chile, piña)
- Sazonador types (sazonador pastor, sazonador carne)
- Basic product categories (vinagre, salsa, tortilla)

Return ONLY comma-separated terms (no articles, no prepositions):
Examples: "achiote, chile guajillo, piña, sazonador" or "vitamina, hierba, te"
"""
    
    # Update span with input details
    span = langwatch.get_current_span()
    span.update(
        input={"query": query, "prompt": intent_prompt},
        metadata={
            "model": OPENAI_MODEL,
            "temperature": 0.1,
            "stage": "intent_understanding"
        }
    )
    
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1, timeout=10)
        response = llm.invoke(intent_prompt)
        
        # Parse response into search terms
        search_terms = [term.strip() for term in response.content.split(',') if term.strip()]
        result = [query] + search_terms[:5]  # Original query + up to 5 intelligent expansions
        
        # Update span with output
        span.update(
            output={"search_terms": result},
            metadata={
                "search_terms_count": len(result),
                "raw_response": response.content,
                "model": OPENAI_MODEL,
                "temperature": 0.1,
                "stage": "intent_understanding"
            }
        )
        
        return result
        
    except Exception as e:
        span.update(
            output={"error": str(e)},
            metadata={
                "error_type": type(e).__name__,
                "model": OPENAI_MODEL,
                "temperature": 0.1,
                "stage": "intent_understanding"
            }
        )
        print(f"Intent understanding failed: {e}")
        return [query]  # Fallback to original query

@langwatch.span(type="rag", name="Product Retrieval")
def retrieve_products(query: str) -> Tuple[List[Tuple[str, float]], List[Document]]:
    """Stage 2: Intelligent product retrieval using LLM-generated search terms"""
    
    span = langwatch.get_current_span()
    span.update(
        input={"query": query},
        metadata={
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "top_k": TOP_K,
            "stage": "product_retrieval"
        }
    )
    
    # Stage 1: Understand intent and generate search terms
    search_terms = _understand_intent_with_llm(query)
    
    # Stage 2: Search catalog using intelligent terms
    all_hits: List[Tuple[str, float]] = []
    all_docs: List[Document] = []
    seen_products = set()
    
    for search_term in search_terms:
        # Vector search for each intelligent term
        results: List[Tuple[Document, float]] = _vector.similarity_search_with_relevance_scores(search_term, k=TOP_K)
        
        for d, score in results:
            if score >= RELEVANCE_THRESHOLD:
                name = d.metadata["display_name"]
                if name not in seen_products:  # Avoid duplicates
                    seen_products.add(name)
                    all_hits.append((name, score))
                    all_docs.append(d)
    
    # Track retrieval results
    if all_hits:
        # Sort by relevance score (highest first) and limit results
        all_hits.sort(key=lambda x: x[1], reverse=True)
        final_hits = all_hits[:TOP_K]
        final_docs = all_docs[:TOP_K]
        
        # Update span with successful retrieval
        span.update(
            output={"products_found": [hit[0] for hit in final_hits]},
            contexts=[
                {
                    "document_id": doc.metadata.get("display_name", "unknown"),
                    "content": doc.page_content
                } for doc in final_docs
            ],
            metadata={
                "relevance_threshold": RELEVANCE_THRESHOLD,
                "top_k": TOP_K,
                "stage": "product_retrieval",
                "products_count": len(final_hits),
                "search_method": "intelligent_vector_search",
                "search_terms_used": ", ".join(search_terms)
            }
        )
        
        return final_hits, final_docs
    
    # Fallback: Fuzzy search if intelligent search found nothing
    matches = fuzzy_best_match(query, _all_names, limit=min(TOP_K, 10))
    hits = [(cand, (score/100.0)) for cand, score, _ in matches if score >= 70]
    if hits:
        docs: List[Document] = []
        for cand, sc in hits:
            docs2 = _vector.similarity_search(f"PRODUCT_NAME: {cand}", k=1)
            if docs2:
                docs.append(docs2[0])
        
        # Update span with fuzzy fallback results
        span.update(
            output={"products_found": [hit[0] for hit in hits]},
            contexts=[
                {
                    "document_id": doc.metadata.get("display_name", "unknown"),
                    "content": doc.page_content
                } for doc in docs
            ],
            metadata={
                "relevance_threshold": RELEVANCE_THRESHOLD,
                "top_k": TOP_K,
                "stage": "product_retrieval",
                "products_count": len(hits),
                "search_method": "fuzzy_fallback"
            }
        )
        
        return hits, docs
    
    # No results found
    span.update(
        output={"products_found": []},
        metadata={
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "top_k": TOP_K,
            "stage": "product_retrieval",
            "products_count": 0,
            "search_method": "no_results"
        }
    )
        
    return [], []

SYSTEM_PROMPT = """You are a friendly and knowledgeable assistant for a specialty ingredient and health food store. Your goal is to help customers discover products, answer their questions, and have a natural, helpful conversation.

**Core Principles:**

1.  **Be Helpful and Conversational:** Your tone should be warm, approachable, and enthusiastic. Engage in a natural, flowing conversation.
2.  **Ground Responses in a real data:** Your primary goal is to provide information about the products we carry.
    *   **CRITICAL RULE:** Only recommend products that appear in the provided "Matched catalog items." NEVER invent or suggest products that are not on this list.
    *   If the user asks about a product you've already mentioned, use the conversation history to provide more details or answer follow-up questions.
3.  **Inspire and Educate:**
    *   When products are found, be enthusiastic and highlight their benefits, uses, and quality.
    *   Share interesting facts, health benefits (without making medical claims), and creative uses to inspire customers.
    *   Mention cultural significance and authentic preparation methods where appropriate.
    *   For herbs and supplements, always include a safety note: "For any health concerns, it's always a good idea to consult with a healthcare professional."

**Conversation Flow:**

*   **Greeting and Initial Query:** Start with a friendly greeting.
*   **Product Recommendations:**
    *   Clearly state the product names you find in the catalog (e.g., "I found **Chile Guajillo** and **Achiote** for you!").
    *   Describe their uses and benefits persuasively (e.g., "The **Chile Guajillo** is perfect for making authentic salsas...").
    *   Suggest complementary products *only if they are also in the catalog matches*.
*   **Handling No Matches:**
    *   If no relevant products are found, apologize gracefully. Say something like, "I couldn't find that specific item in our catalog right now, but we have a wide range of other specialty ingredients. Perhaps I can help you find something else?"
    *   Do not suggest or promise products you don't have.
*   **Follow-up Questions:**
    *   If the user asks a follow-up question (e.g., "tell me more about the achiote"), use the context of the conversation to provide a detailed answer.
    *   Look at the "history" to understand what has already been discussed.
*   **Language:** Respond in the user's language (the API will provide a hint, but adapt if the user switches).
*   **Closing:** End the conversation in a friendly and helpful manner.

**About Our Store:**

*   We specialize in unique Mexican ingredients, health foods, and specialty items.
*   We carry vitamins, herbal products, and wellness items.
*   Some design/printing services are also available.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{user_input}"),
        ("system", "Matched catalog items (highest first): {matches}\nOnly rely on these names to determine if the product is sold here."),
    ]
)

def _format_matches(hits: List[Tuple[str,float]]) -> str:
    if not hits:
        return "NONE"
    lines = [f"{i+1}. {name} (relevance {score:.2f})" for i, (name, score) in enumerate(hits[:5])]
    return "\n".join(lines)

# Runnable pipeline
def build_chain():
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2, timeout=30)
    retrieve = RunnableLambda(lambda x: retrieve_products(x["user_input"]))
    enrich = RunnableLambda(lambda x: {
        **x,
        "hits_docs": retrieve_products(x["user_input"]),
    })
    # We already need matches, so recompose neatly
    def _inject_matches(x: Dict[str, Any]) -> Dict[str, Any]:
        hits, _ = retrieve_products(x["user_input"])
        x["matches"] = _format_matches(hits)
        return x

    chain = (
        {
            "user_input": RunnablePassthrough() | (lambda x: x["user_input"]),
            "history": RunnablePassthrough() | (lambda x: x["history"])
        }
        | RunnableLambda(_inject_matches)
        | PROMPT
        | ChatOpenAI(model=OPENAI_MODEL, temperature=0.2, timeout=30)
        | StrOutputParser()
    )
    return chain

# Public function used by the API
@langwatch.span(type="llm", name="Generate Sales Response")
def answer(user_input: str, history_messages, session_lang_hint: str | None = None) -> str:
    # History comes as a LangChain BaseChatMessageHistory
    
    span = langwatch.get_current_span()
    span.update(
        input={
            "user_input": user_input,
            "language_hint": session_lang_hint,
            "history_length": len(history_messages.messages)
        },
        metadata={
            "model": OPENAI_MODEL,
            "stage": "sales_response_generation"
        }
    )
    
    chain = build_chain()
    
    # Minimal language steering
    lang = session_lang_hint or _language_for(user_input)
    if lang == "es":
        user_input = user_input
    else:
        user_input = user_input  # Let model respond in same language; system prompt enforces

    # Format history for MessagesPlaceholder and generate response
    try:
        output = chain.invoke({"user_input": user_input, "history": history_messages.messages})
        
        # Update span with successful generation
        span.update(
            output=output,
            metadata={
                "model": OPENAI_MODEL,
                "stage": "sales_response_generation",
                "detected_language": lang,
                "response_length": len(output)
            }
        )
        
        return output
        
    except Exception as e:
        span.update(
            output=f"Error: {str(e)}",
            metadata={
                "model": OPENAI_MODEL,
                "stage": "sales_response_generation",
                "error_type": type(e).__name__
            }
        )
        raise
