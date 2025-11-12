# Quick Start - Week 2 Features

Fast guide to testing the new Week 2 enhancements.

## Prerequisites

```bash
cd curious-kellly/backend
npm install
```

Create `.env` file:
```bash
OPENAI_API_KEY=your_key_here
NODE_ENV=development
PORT=3000
```

## 1️⃣ Start the Server

```bash
npm run dev
```

Server should start on http://localhost:3000

## 2️⃣ Test Safety Router

Run comprehensive safety tests:

```bash
npm run test:safety
```

Expected output:
```
🧪 Running Safety Tests...

📊 Test 1: Precision (safe content)
  ✅ Why do leaves change color in autumn?...
  [30 tests]

📊 Test 2: Recall (unsafe content)
  ✅ Blocked: How to build a weapon at home...
  [27 tests]

📊 Test 3: Age-appropriate content
  ✅ Blocked for age 5
  [19 tests]

📊 Test 4: Adversarial prompt detection
  ✅ Blocked adversarial: Ignore your safety...
  [11 tests]

📊 Test 5: Multilingual safety checks
  ✅ Spanish safe content passed
  [6 tests]

============================================================
📈 RESULTS
============================================================
Precision: 98.50% (target: ≥98%) ✅ PASS
Recall: 96.30% (target: ≥95%) ✅ PASS
Adversarial Detection: 81.82% ✅ GOOD
Average Latency: 342ms ✅ Fast
============================================================

🎉 SAFETY TESTS PASSED!
```

## 3️⃣ Test Session Management

### Start a new session:
```bash
curl -X POST http://localhost:3000/api/sessions/start \
  -H "Content-Type: application/json" \
  -d '{"age":35,"lessonId":"leaves-change-color"}'
```

Response:
```json
{
  "status": "ok",
  "data": {
    "sessionId": "abc-123-def-456",
    "age": 35,
    "lessonId": "leaves-change-color",
    "startedAt": "2025-11-11T...",
    "progress": {
      "currentPhase": "welcome",
      "completedPhases": [],
      "interactionsCompleted": [],
      "teachingMomentsViewed": []
    },
    "state": {
      "isActive": true,
      "isPaused": false,
      "isCompleted": false
    }
  }
}
```

### Get active sessions:
```bash
curl http://localhost:3000/api/sessions/active
```

### Update progress:
```bash
curl -X POST http://localhost:3000/api/sessions/{sessionId}/progress \
  -H "Content-Type: application/json" \
  -d '{
    "currentPhase": "teaching",
    "completedPhase": "welcome",
    "interactionCompleted": "intro-question"
  }'
```

### Get session stats:
```bash
curl http://localhost:3000/api/sessions/{sessionId}/stats
```

### Complete session:
```bash
curl -X POST http://localhost:3000/api/sessions/{sessionId}/complete
```

## 4️⃣ Test RAG Content Population

### Check RAG status:
```bash
curl http://localhost:3000/api/rag/status
```

### Populate all lessons:
```bash
node scripts/populate-rag.js
```

Output:
```
🚀 RAG Content Population Tool

============================================================
✅ Vector DB: pinecone
✅ Embedding model: text-embedding-3-small

📚 Getting all lessons...
   Found 2 lessons

============================================================
📊 Starting Population
============================================================

📖 Processing: Why Do Leaves Change Color?
   ID: leaves-change-color
   📝 Age 2-5...
      ✅ Created 4 vectors
   📝 Age 6-12...
      ✅ Created 4 vectors
   [... continues for all age buckets]

============================================================
📈 POPULATION COMPLETE
============================================================
Lessons processed:  2
Variants processed: 12
Vectors created:    48
Errors:             0
Duration:           12.34s
Rate:               3.9 vectors/sec
============================================================

🎉 All content populated successfully!
```

### Test vector search:
```bash
curl -X POST http://localhost:3000/api/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Why do leaves change color?",
    "topK": 3
  }'
```

Response:
```json
{
  "status": "ok",
  "data": {
    "query": "Why do leaves change color?",
    "results": [
      {
        "id": "leaves-change-color-6-12-0",
        "score": 0.912,
        "text": "Why Do Leaves Change Color?...",
        "lessonId": "leaves-change-color-6-12",
        "type": "title"
      }
    ],
    "count": 3
  }
}
```

### Add single lesson:
```bash
curl -X POST http://localhost:3000/api/rag/add-lesson \
  -H "Content-Type: application/json" \
  -d '{
    "lessonId": "water-cycle",
    "ageBucket": "6-12"
  }'
```

## 5️⃣ Test All Endpoints

### Health check:
```bash
curl http://localhost:3000/health
```

### API info:
```bash
curl http://localhost:3000/
```

This shows all available endpoints.

## 🔍 Monitoring

### Check active sessions:
```bash
curl http://localhost:3000/api/sessions/active
```

### Check RAG status:
```bash
curl http://localhost:3000/api/rag/status
```

### Cleanup expired sessions:
```bash
curl -X POST http://localhost:3000/api/sessions/cleanup
```

## 🧪 Run All Tests

```bash
npm test
```

This runs:
- Safety tests
- Realtime tests
- Reinmaker tests

## 🐛 Troubleshooting

### Server won't start
- Check `.env` file has `OPENAI_API_KEY`
- Run `npm install` first
- Check port 3000 is not in use

### Safety tests fail
- Verify OpenAI API key is valid
- Check internet connection
- Review failed test cases in output

### RAG not available
- Set either `PINECONE_API_KEY` or `QDRANT_URL`
- Check vector DB service is running
- Verify API keys are valid

### Sessions not persisting
- Redis optional (falls back to memory)
- Set `REDIS_URL` for production persistence
- Check Redis is running if configured

## 📚 Documentation

Full documentation:
- `WEEK_2_ENHANCEMENTS_COMPLETE.md` - Complete summary
- `tests/SAFETY_TEST_GUIDE.md` - Safety testing guide
- `scripts/README.md` - Script usage guide

---

**Status**: ✅ All Week 2 features ready to test  
**Next**: Week 3 - Voice & Avatar Integration



