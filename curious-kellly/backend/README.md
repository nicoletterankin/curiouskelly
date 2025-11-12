# Curious Kellly - Backend Service
**The Daily Lesson API**

---

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Create .env file
cp .env.example .env
# Edit .env with your OpenAI API key

# Start development server
npm run dev

# Test the server
curl http://localhost:3000/health
```

---

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "ok",
  "service": "curious-kellly-backend",
  "timestamp": "2025-10-29T...",
  "version": "0.1.0"
}
```

### Test OpenAI Connection
```bash
GET /api/realtime/test
```

Response:
```json
{
  "status": "ok",
  "message": "OpenAI connection successful",
  "data": {
    "success": true,
    "response": "Hello from Curious Kellly backend!",
    "model": "gpt-4o-mini"
  }
}
```

### Get Kelly's Response
```bash
POST /api/realtime/kelly
Content-Type: application/json

{
  "age": 35,
  "topic": "leaves",
  "message": "Why do leaves change color?"
}
```

Response:
```json
{
  "status": "ok",
  "data": {
    "success": true,
    "kellyAge": 27,
    "kellyPersona": "knowledgeable-adult",
    "response": "Great question! Leaves change color because..."
  }
}
```

---

## 🏗️ Project Structure

```
backend/
├── src/
│   ├── index.js           # Main server
│   ├── api/
│   │   └── realtime.js    # OpenAI API routes
│   ├── services/
│   │   └── realtime.js    # OpenAI service logic
│   ├── models/            # Data models (coming soon)
│   └── utils/             # Helper functions (coming soon)
├── config/                # Configuration files
├── tests/                 # Test files (coming soon)
├── .env                   # Environment variables (not in git)
├── .env.example           # Environment template
├── package.json
└── README.md
```

---

## 🔧 Development

### Start Server
```bash
npm run dev    # Development with auto-reload
npm start      # Production
```

### Environment Variables
```env
NODE_ENV=development
PORT=3000
OPENAI_API_KEY=sk-proj-...
OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview-2024-10-01
LOG_LEVEL=debug
```

---

## 📈 Voice Metrics Logging

- `src/services/voice.js` now logs `request_started_at`, `response_sent_at`, persona, and calculated round-trip time for every interaction (hashed session ids only).
- Metrics append to `analytics/Kelly/voice-latency.csv` (auto-creates with header on first write). Use this file for latency dashboards and nightly regression reports.
- Keep logs free of PII—store only age bucket and hashed session identifiers. Mirror this pattern in `src/services/session.js` when session persistence is added.

---

## ✅ Day 1 Status

**Completed:**
- ✅ Project scaffolded
- ✅ Express server running
- ✅ OpenAI integration working
- ✅ Health check endpoint
- ✅ Test OpenAI endpoint
- ✅ Kelly response endpoint (basic)

**Next (Day 2):**
- [ ] Safety router with moderation
- [ ] Moderation test suite
- [ ] Safety endpoint

---

## 📞 Testing Commands

### PowerShell
```powershell
# Health check
curl http://localhost:3000/health

# Test OpenAI
curl http://localhost:3000/api/realtime/test

# Get Kelly response
curl -X POST http://localhost:3000/api/realtime/kelly `
  -H "Content-Type: application/json" `
  -d '{"age":35,"topic":"leaves","message":"Why do leaves change color?"}'
```

---

**Status**: 🟢 Day 1 Complete  
**Next**: Day 2 - Safety Router  
**Version**: 0.1.0

