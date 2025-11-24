/**
 * RAG API Routes
 * Vector database search and content retrieval
 */

const express = require('express');
const RAGService = require('../services/rag');
const LessonService = require('../services/lessons');

const router = express.Router();

// Initialize RAG service
let ragService;
try {
  ragService = new RAGService();
} catch (error) {
  console.warn('⚠️  RAG service not available:', error.message);
}

/**
 * Search for relevant content
 * POST /api/rag/search
 * Body: { query: "search text", topK: 5, lessonId: "optional" }
 */
router.post('/search', async (req, res) => {
  try {
    if (!ragService || !ragService.isAvailable()) {
      return res.status(503).json({
        status: 'error',
        message: 'RAG service not available. Vector database not configured.',
        hint: 'Set PINECONE_API_KEY or QDRANT_URL in environment variables'
      });
    }

    const { query, topK = 5, lessonId } = req.body;

    if (!query) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required field: query'
      });
    }

    const filter = lessonId ? { lessonId } : {};
    const results = await ragService.search(query, {
      topK: parseInt(topK),
      filter,
      includeMetadata: true
    });

    res.json({
      status: 'ok',
      data: results
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Get context for a query using RAG
 * POST /api/rag/context
 * Body: { query: "question", lessonId: "optional" }
 */
router.post('/context', async (req, res) => {
  try {
    if (!ragService || !ragService.isAvailable()) {
      return res.status(503).json({
        status: 'error',
        message: 'RAG service not available. Vector database not configured.'
      });
    }

    const { query, lessonId } = req.body;

    if (!query) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required field: query'
      });
    }

    const context = await ragService.getContextForQuery(query, lessonId);

    res.json({
      status: 'ok',
      data: context
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Populate vector database with lessons
 * POST /api/rag/populate
 */
router.post('/populate', async (req, res) => {
  try {
    if (!ragService || !ragService.isAvailable()) {
      return res.status(503).json({
        status: 'error',
        message: 'RAG service not available. Vector database not configured.'
      });
    }

    const lessonService = new LessonService();
    const result = await ragService.populateFromLessons(lessonService);

    res.json({
      status: 'ok',
      data: result,
      message: 'Vector database populated successfully'
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Check RAG service status
 * GET /api/rag/status
 */
router.get('/status', async (req, res) => {
  const available = ragService && ragService.isAvailable();
  
  res.json({
    status: 'ok',
    data: {
      available,
      vectorDBType: available ? ragService.vectorDBType : null,
      embeddingModel: available ? ragService.embeddingModel : null,
      message: available 
        ? 'RAG service is available' 
        : 'RAG service not configured. Set PINECONE_API_KEY or QDRANT_URL'
    }
  });
});

/**
 * Add specific lesson content to vector DB
 * POST /api/rag/add-lesson
 * Body: { lessonId: "leaves-change-color", ageBucket: "6-12" }
 */
router.post('/add-lesson', async (req, res) => {
  try {
    if (!ragService || !ragService.isAvailable()) {
      return res.status(503).json({
        status: 'error',
        message: 'RAG service not available'
      });
    }

    const { lessonId, ageBucket } = req.body;

    if (!lessonId) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required field: lessonId'
      });
    }

    // Load lesson
    const lessonService = new LessonService();
    const lesson = await lessonService.loadLesson(lessonId);

    // Process specific age bucket or all
    const bucketsToProcess = ageBucket 
      ? [ageBucket] 
      : Object.keys(lesson.ageVariants);

    let totalVectors = 0;
    for (const bucket of bucketsToProcess) {
      const variant = lesson.ageVariants[bucket];
      if (!variant) continue;

      const content = {
        title: lesson.title,
        description: lesson.description,
        teachingMoments: variant.teachingMoments || [],
        summary: variant.description || '',
        objectives: variant.objectives || [],
        vocabulary: variant.vocabulary?.keyTerms || []
      };

      const result = await ragService.addLessonContent(`${lessonId}-${bucket}`, content);
      totalVectors += result.vectorsCount;
    }

    res.json({
      status: 'ok',
      data: {
        lessonId,
        bucketsProcessed: bucketsToProcess.length,
        vectorsCreated: totalVectors
      },
      message: `Added ${totalVectors} vectors for ${lessonId}`
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

/**
 * Generate embedding for text (utility endpoint)
 * POST /api/rag/embed
 * Body: { text: "text to embed" }
 */
router.post('/embed', async (req, res) => {
  try {
    if (!ragService || !ragService.isAvailable()) {
      return res.status(503).json({
        status: 'error',
        message: 'RAG service not available'
      });
    }

    const { text } = req.body;

    if (!text) {
      return res.status(400).json({
        status: 'error',
        message: 'Missing required field: text'
      });
    }

    const embedding = await ragService.generateEmbedding(text);

    res.json({
      status: 'ok',
      data: {
        embedding,
        dimensions: embedding.length,
        model: ragService.embeddingModel
      }
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

module.exports = router;










