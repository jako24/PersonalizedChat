# import os
# from typing import Dict
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory

# # In-memory by default; switch to Redis for prod to persist sessions
# _memory_store: Dict[str, ChatMessageHistory] = {}

# def get_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in _memory_store:
#         _memory_store[session_id] = ChatMessageHistory()
#     return _memory_store[session_id]


"""
chat_history.py – Redis-backed chat-history store
-------------------------------------------------
Prereqs:
  pip install redis langchain

Required env vars (recommended > file‐based .env):
  REDIS_HOST      (default: "localhost")
  REDIS_PORT      (default: 6379)
  REDIS_PASSWORD  (optional – omit if unauthenticated)
  REDIS_DB        (default: 0)

Optionally set REDIS_TTL_SECONDS to auto-expire idle sessions.
"""
import os
from typing import Dict
from redis import Redis
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# ---------------------------------------------------------------------------
# 1.  Low-level Redis connection (single client, re-used by the history class)
# ---------------------------------------------------------------------------
_redis_client: Redis | None = None

def _redis() -> Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD") or None,
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,          # store/read plain UTF-8 strings
            health_check_interval=30,       # keep the connection fresh
        )
    return _redis_client


# --------------------------------------------------------------
# 2.  Cache of history objects so we don’t re-instantiate per call
# --------------------------------------------------------------
_history_cache: Dict[str, BaseChatMessageHistory] = {}


# ---------------------------------------------------------------------------------
# 3.  Public helper – exact same signature you already use, now backed by Redis.
# ---------------------------------------------------------------------------------
def get_history(session_id: str) -> BaseChatMessageHistory:
    """
    Return (and create if necessary) a chat-history object for the given
    session_id.  Uses Redis for storage so histories survive interpreter
    restarts and can be shared by multiple workers/containers.
    """
    if session_id not in _history_cache:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            ttl = int(os.getenv("REDIS_TTL_SECONDS", "3600")) # Expire after 1 hour
            _history_cache[session_id] = RedisChatMessageHistory(
                session_id=session_id,
                url=redis_url,
                key_prefix="chat_history:",
                ttl=ttl if ttl > 0 else None,
            )
            # Test connection
            _history_cache[session_id].messages
            print(f"✅ Successfully connected to Redis for session: {session_id}")
        except Exception as e:
            print(f"⚠️ Redis connection failed, falling back to in-memory history: {e}")
            _history_cache[session_id] = ChatMessageHistory()

    return _history_cache[session_id]


# ---------------------------------------------------------------------------------
# 4.  Optional helper to wipe a session (e.g. on user logout / GDPR delete)
# ---------------------------------------------------------------------------------
def delete_history(session_id: str) -> None:
    history = _history_cache.pop(session_id, None)
    if history:
        history.clear()
    # Also directly delete from Redis using the client
    key = f"chat_history:{session_id}"
    _redis().delete(key)