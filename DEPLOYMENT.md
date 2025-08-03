# FerdekChat Deployment Guide

## 🎯 Overview

FerdekChat consists of two components:
1. **FastAPI Backend**: Handles LLM processing, vector search, and chat logic
2. **Streamlit Frontend**: Simple chat interface that calls the backend

## 🚀 Quick Deploy to Streamlit Cloud

### 1. Fix Git Repository (Remove Secrets)

The deployment failed because your `.env` file with API keys was committed to git. Fix this:

```bash
# Remove .env from git history
git rm --cached .env

# Create your local .env from the example
cp env.example .env
# Edit .env with your actual API keys

# Commit the fix
git add .
git commit -m "Remove .env from repository, add example file"
git push origin main
```

### 2. Deploy Frontend to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository: `jako24/PersonalizedChat`
3. Set main file path: `ui/streamlit_app.py`
4. Advanced settings > Environment variables:
   ```
   BACKEND_URL=https://your-backend-url.com/chat
   ```
5. Deploy!

## 🖥️ Backend Deployment Options

The Streamlit frontend needs a deployed backend. Here are your options:

### Option A: Deploy to Railway/Render/Heroku

1. Create account on [Railway](https://railway.app) or [Render](https://render.com)
2. Connect your GitHub repo
3. Set build command: `pip install -r requirements-backend.txt`
4. Set start command: `uvicorn app.server:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   ```
   OPENAI_API_KEY=your_actual_key
   LANGWATCH_API_KEY=your_actual_key
   INDEX_DIR=/app/data/index
   RELEVANCE_THRESHOLD=0.25
   ```

### Option B: Deploy to Fly.io

Use the included Dockerfile:

```bash
# Install fly.io CLI
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch
fly deploy
```

### Option C: Local Backend + Ngrok (Testing)

```bash
# Terminal 1: Start backend
source ~/myenv/bin/activate
uvicorn app.server:app --host 0.0.0.0 --port 8000

# Terminal 2: Expose with ngrok
ngrok http 8000
# Copy the https://xxx.ngrok.io URL
```

Then update your Streamlit app's `BACKEND_URL` environment variable.

## 🔧 Configuration

### Backend Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LANGWATCH_API_KEY`: Your LangWatch key for monitoring (optional)
- `INDEX_DIR`: Path to FAISS index files
- `RELEVANCE_THRESHOLD`: Vector search threshold (0.25 recommended)
- `REDIS_URL`: Redis connection for chat history (optional)

### Frontend Environment Variables

- `BACKEND_URL`: URL to your deployed backend API

## 📁 File Structure

```
FerdekChat/
├── app/                    # Backend FastAPI application
│   ├── server.py          # Main API server
│   ├── chain.py           # LLM chain and search logic
│   ├── memory.py          # Chat history management
│   └── catalog_utils.py   # Product catalog utilities
├── ui/
│   └── streamlit_app.py   # Frontend Streamlit app
├── data/
│   ├── catalog.txt        # Product catalog
│   └── index/             # FAISS vector index
├── requirements.txt       # Frontend dependencies (minimal)
├── requirements-backend.txt # Backend dependencies (full)
├── env.example           # Environment variables template
└── Dockerfile            # Backend container config
```

## 🐛 Troubleshooting

### "No module named 'langwatch'" on Streamlit Cloud
- Make sure you're using `requirements.txt` (not `requirements-backend.txt`)
- The frontend doesn't need LangChain/FAISS dependencies

### "Error: Connection refused" in Streamlit app
- Check that `BACKEND_URL` environment variable is set correctly
- Ensure your backend is deployed and accessible

### "OpenAI API Key not found" 
- Set `OPENAI_API_KEY` in your backend deployment environment variables
- Don't commit API keys to git!

## 📊 Monitoring

Once deployed, visit [app.langwatch.ai](https://app.langwatch.ai) to monitor:
- Chat conversations and user queries
- LLM performance and costs  
- Product recommendation effectiveness
- Search system intelligence

Your two-stage intelligent search system will show detailed traces of:
1. Intent understanding (LLM generates search terms)
2. Product retrieval (vector search + fallback)
3. Sales response generation (final LLM response)