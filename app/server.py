import os
from dotenv import load_dotenv
load_dotenv()

# Initialize LangWatch for monitoring and analytics
import langwatch
langwatch.setup(
    api_key=os.getenv("LANGWATCH_API_KEY"),
    base_attributes={
        "service.name": "ferdek-chat",
        "service.version": "1.0.0",
        "environment": "production"
    },
    debug=os.getenv("LANGWATCH_DEBUG", "false").lower() == "true"
)

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.memory import get_history
from app.chain import answer

app = FastAPI(title="Product Chatbot", version="1.0.0")

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique user/session id")
    message: str = Field(..., description="User message")
    lang_hint: Optional[str] = Field(None, description="Optional 'es' or 'en' to steer language")

class ChatResponse(BaseModel):
    reply: str
    session_id: str

API_KEY = os.getenv("SERVICE_API_KEY")  # optional extra auth

@app.post("/chat", response_model=ChatResponse)
@langwatch.trace(name="Chat Request")
def chat(req: ChatRequest, x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # Add metadata for conversation tracking
    trace = langwatch.get_current_trace()
    trace.update(
        metadata={
            "user_id": req.session_id,
            "thread_id": req.session_id,
            "language_hint": req.lang_hint or "auto",
            "message_length": len(req.message),
            "endpoint": "/chat"
        },
        input=req.message
    )

    history = get_history(req.session_id)

    # Generate response with LangWatch tracking
    try:
        reply = answer(req.message, history, req.lang_hint)
        
        # Update trace with successful output
        trace.update(
            output=reply,
            metadata={
                "user_id": req.session_id,
                "thread_id": req.session_id,
                "language_hint": req.lang_hint or "auto",
                "message_length": len(req.message),
                "endpoint": "/chat",
                "reply_length": len(reply),
                "status": "success"
            }
        )
        
    except Exception as e:
        # Track errors in LangWatch
        trace.update(
            output=f"Error: {str(e)}",
            metadata={
                "user_id": req.session_id,
                "thread_id": req.session_id,
                "language_hint": req.lang_hint or "auto",
                "message_length": len(req.message),
                "endpoint": "/chat",
                "status": "error",
                "error_type": type(e).__name__
            }
        )
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Update history
    history.add_user_message(req.message)
    history.add_ai_message(reply)

    return ChatResponse(reply=reply, session_id=req.session_id)

if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=False)
