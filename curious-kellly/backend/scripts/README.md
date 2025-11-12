# Backend Scripts

Utility scripts for the Curious Kelly backend.

## Available Scripts

### 📊 populate-rag.js
Populate vector database with lesson content for RAG retrieval.

**Usage:**
```bash
# Populate all lessons
node scripts/populate-rag.js

# Populate single lesson
node scripts/populate-rag.js --lesson=leaves-change-color

# Dry run (preview changes)
node scripts/populate-rag.js --dry-run --verbose

# Populate and test search
node scripts/populate-rag.js --test-search
```

**Requirements:**
- OpenAI API key (for embeddings)
- Pinecone or Qdrant configured

**Environment:**
```bash
# Required
OPENAI_API_KEY=sk-...

# Option 1: Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX=curious-kellly-lessons

# Option 2: Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=curious-kellly-lessons
```

---

### ✅ verify-env.js
Verify environment variables are set correctly.

**Usage:**
```bash
node scripts/verify-env.js
```

Checks for:
- OpenAI API key
- Redis configuration (optional)
- Vector DB configuration (optional)

---

## Adding New Scripts

When creating a new script:

1. Add shebang line: `#!/usr/bin/env node`
2. Document usage with `--help` flag
3. Handle errors gracefully
4. Add to this README
5. Add to `package.json` scripts if commonly used

Example package.json script:
```json
{
  "scripts": {
    "populate-rag": "node scripts/populate-rag.js"
  }
}
```

---

## Script Conventions

### Exit Codes
- `0` - Success
- `1` - Error or validation failed

### Output Format
- Use emoji for visual hierarchy
- Show progress for long operations
- Use `=`.repeat(60) for section dividers
- Include summary statistics

### Error Handling
- Catch and log errors
- Show helpful error messages
- Suggest fixes when possible
- Exit with code 1 on error

---

## Troubleshooting

### "RAG service not available"
Set either PINECONE_API_KEY or QDRANT_URL in your .env file.

### "OpenAI API key not found"
Set OPENAI_API_KEY in your .env file.

### "Lesson not found"
Check that the lesson ID exists in config/lessons/ directory.

---

**Last Updated**: November 11, 2025



