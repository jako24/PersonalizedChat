# LangWatch Setup Guide

LangWatch is now integrated into your FerdekChat application to provide comprehensive monitoring and analytics for your LLM interactions.

## 🔑 Getting Your API Key

1. **Sign up at LangWatch**: Go to [app.langwatch.ai](https://app.langwatch.ai) and create an account
2. **Find your API key**: Navigate to your project settings and copy your API key
3. **Set environment variable**: Replace `your_langwatch_api_key_here` in your `.env` file with your actual API key:

```bash
LANGWATCH_API_KEY=lw-your-actual-api-key-here
```

## 📊 What LangWatch Tracks

### **Conversation Tracking**
- ✅ **User messages** and **AI responses**
- ✅ **Session IDs** for conversation threading
- ✅ **Message lengths** and **response times**
- ✅ **Language detection** (Spanish/English)

### **Intelligent Search Analytics**
- ✅ **Intent understanding**: LLM-generated search terms
- ✅ **Product retrieval**: Vector search results and fallbacks
- ✅ **Search method tracking**: Vector search vs fuzzy fallback
- ✅ **Products found**: Names and relevance scores

### **Performance Metrics**
- ✅ **LLM calls**: Token usage, costs, latency
- ✅ **Error tracking**: Failed searches, LLM errors
- ✅ **Success rates**: Response generation success

### **Sales Intelligence**
- ✅ **Product recommendations**: What products are being suggested
- ✅ **User behavior**: Popular queries and product interests
- ✅ **Conversation flow**: Multi-turn conversation analysis

## 🎯 Key Features Available

### **Real-time Monitoring**
- Monitor live conversations as they happen
- Track system performance and response times
- Get alerts for errors or unusual patterns

### **Analytics Dashboard**
- User engagement metrics
- Popular product searches
- Conversation success rates
- Language usage patterns

### **Product Insights**
- Most recommended products
- Search term effectiveness
- Customer interest patterns
- Conversion tracking

## 🚀 Optional: Enable Debug Mode

For detailed debugging during development, add this to your `.env`:

```bash
LANGWATCH_DEBUG=true
```

## 📈 Viewing Your Data

Once your API key is set up:

1. **Start your server**: `uvicorn app.server:app --host 0.0.0.0 --port 8000`
2. **Make some chat requests** to generate data
3. **Visit your LangWatch dashboard**: [app.langwatch.ai](https://app.langwatch.ai)
4. **Explore your analytics**: View traces, conversations, and performance metrics

## 🔧 Integration Details

The integration automatically tracks:
- **Chat endpoints**: Full conversation traces
- **Intent understanding**: LLM reasoning for search terms
- **Product retrieval**: RAG search with contexts
- **Response generation**: Final LLM calls with sales responses

Your two-stage intelligent search system is now fully observable! 🎉