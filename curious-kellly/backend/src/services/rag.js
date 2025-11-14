/**
 * RAG (Retrieval-Augmented Generation) Service
 * Uses vector database for content retrieval
 * Supports Pinecone and Qdrant
 */

const OpenAI = require('openai');

class RAGService {
  constructor() {
    // Initialize OpenAI for embeddings
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY not found in environment variables');
    }

    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });

    this.embeddingModel = process.env.EMBEDDING_MODEL || 'text-embedding-3-small';
    this.vectorDB = null;
    this.vectorDBType = null;
    
    this.initializeVectorDB();
  }

  /**
   * Initialize vector database based on environment variables
   */
  initializeVectorDB() {
    // Check for Pinecone
    if (process.env.PINECONE_API_KEY) {
      this.initializePinecone();
      return;
    }

    // Check for Qdrant
    if (process.env.QDRANT_URL) {
      this.initializeQdrant();
      return;
    }

    console.warn('⚠️  No vector database configured. RAG features will be limited.');
    console.warn('   Set PINECONE_API_KEY or QDRANT_URL to enable vector search.');
  }

  /**
   * Initialize Pinecone
   */
  async initializePinecone() {
    try {
      const { Pinecone } = require('@pinecone-database/pinecone');
      
      const pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
        environment: process.env.PINECONE_ENVIRONMENT || 'us-east-1'
      });

      const indexName = process.env.PINECONE_INDEX || 'curious-kellly-lessons';
      this.vectorDB = pinecone.index(indexName);
      this.vectorDBType = 'pinecone';
      
      console.log('✅ Pinecone initialized:', indexName);
    } catch (error) {
      console.error('❌ Pinecone initialization error:', error);
      throw error;
    }
  }

  /**
   * Initialize Qdrant
   */
  async initializeQdrant() {
    try {
      const { QdrantClient } = require('@qdrant/js-client-rest');
      
      this.vectorDB = new QdrantClient({
        url: process.env.QDRANT_URL,
        apiKey: process.env.QDRANT_API_KEY
      });

      const collectionName = process.env.QDRANT_COLLECTION || 'curious-kellly-lessons';
      this.vectorDBType = 'qdrant';
      
      // Verify collection exists
      const collections = await this.vectorDB.getCollections();
      const collectionExists = collections.collections.some(c => c.name === collectionName);
      
      if (!collectionExists) {
        console.warn(`⚠️  Qdrant collection "${collectionName}" does not exist. Creating...`);
        await this.createQdrantCollection(collectionName);
      }
      
      console.log('✅ Qdrant initialized:', collectionName);
    } catch (error) {
      console.error('❌ Qdrant initialization error:', error);
      throw error;
    }
  }

  /**
   * Create Qdrant collection
   */
  async createQdrantCollection(collectionName) {
    await this.vectorDB.createCollection(collectionName, {
      vectors: {
        size: 1536, // text-embedding-3-small dimension
        distance: 'Cosine'
      }
    });
  }

  /**
   * Generate embedding for text
   */
  async generateEmbedding(text) {
    try {
      const response = await this.openai.embeddings.create({
        model: this.embeddingModel,
        input: text
      });

      return response.data[0].embedding;
    } catch (error) {
      console.error('Embedding generation error:', error);
      throw new Error(`Failed to generate embedding: ${error.message}`);
    }
  }

  /**
   * Add lesson content to vector database
   */
  async addLessonContent(lessonId, content) {
    if (!this.vectorDB) {
      throw new Error('Vector database not initialized');
    }

    try {
      // Generate embeddings for different parts of the lesson
      const textsToEmbed = [
        `${content.title}. ${content.description}`,
        ...content.teachingMoments.map(m => m.content),
        content.summary || ''
      ];

      const embeddings = await Promise.all(
        textsToEmbed.map(text => this.generateEmbedding(text))
      );

      // Prepare vectors for database
      const vectors = embeddings.map((embedding, index) => ({
        id: `${lessonId}-${index}`,
        values: embedding,
        metadata: {
          lessonId,
          type: index === 0 ? 'title' : index === textsToEmbed.length - 1 ? 'summary' : 'teaching-moment',
          index,
          text: textsToEmbed[index]
        }
      }));

      // Insert into vector database
      if (this.vectorDBType === 'pinecone') {
        await this.vectorDB.upsert(vectors);
      } else if (this.vectorDBType === 'qdrant') {
        const collectionName = process.env.QDRANT_COLLECTION || 'curious-kellly-lessons';
        await this.vectorDB.upsert(collectionName, {
          points: vectors
        });
      }

      console.log(`✅ Added ${vectors.length} vectors for lesson: ${lessonId}`);
      return { success: true, vectorsCount: vectors.length };
    } catch (error) {
      console.error(`Error adding lesson content: ${error.message}`);
      throw error;
    }
  }

  /**
   * Search for relevant content
   */
  async search(query, options = {}) {
    if (!this.vectorDB) {
      throw new Error('Vector database not initialized. Content search unavailable.');
    }

    try {
      const {
        topK = 5,
        filter = {},
        includeMetadata = true
      } = options;

      // Generate embedding for query
      const queryEmbedding = await this.generateEmbedding(query);

      // Search in vector database
      let results;

      if (this.vectorDBType === 'pinecone') {
        results = await this.vectorDB.query({
          vector: queryEmbedding,
          topK,
          includeMetadata,
          filter
        });
      } else if (this.vectorDBType === 'qdrant') {
        const collectionName = process.env.QDRANT_COLLECTION || 'curious-kellly-lessons';
        results = await this.vectorDB.search(collectionName, {
          vector: queryEmbedding,
          limit: topK,
          filter,
          with_payload: includeMetadata
        });
      }

      // Format results
      const formattedResults = this.formatSearchResults(results);
      
      return {
        query,
        results: formattedResults,
        count: formattedResults.length
      };
    } catch (error) {
      console.error('Vector search error:', error);
      throw new Error(`Vector search failed: ${error.message}`);
    }
  }

  /**
   * Format search results for consistent API
   */
  formatSearchResults(results) {
    if (this.vectorDBType === 'pinecone') {
      return results.matches.map(match => ({
        id: match.id,
        score: match.score,
        text: match.metadata?.text || '',
        lessonId: match.metadata?.lessonId || '',
        type: match.metadata?.type || 'unknown',
        metadata: match.metadata
      }));
    } else if (this.vectorDBType === 'qdrant') {
      return results.map(result => ({
        id: result.id,
        score: result.score,
        text: result.payload?.text || '',
        lessonId: result.payload?.lessonId || '',
        type: result.payload?.type || 'unknown',
        metadata: result.payload
      }));
    }

    return [];
  }

  /**
   * Get context for a lesson query using RAG
   */
  async getContextForQuery(query, lessonId = null) {
    try {
      const filter = lessonId ? { lessonId } : {};
      const searchResults = await this.search(query, {
        topK: 3,
        filter,
        includeMetadata: true
      });

      // Combine relevant context
      const context = searchResults.results
        .map(r => r.text)
        .join('\n\n');

      return {
        context,
        sources: searchResults.results.map(r => ({
          lessonId: r.lessonId,
          type: r.type
        })),
        query
      };
    } catch (error) {
      console.error('Context retrieval error:', error);
      // Return empty context if RAG fails
      return {
        context: '',
        sources: [],
        query,
        error: error.message
      };
    }
  }

  /**
   * Populate vector database with all lessons
   */
  async populateFromLessons(lessonService) {
    try {
      const lessons = await lessonService.getAllLessons();
      
      console.log(`📚 Populating vector DB with ${lessons.length} lessons...`);

      for (const lessonInfo of lessons) {
        try {
          const lesson = await lessonService.loadLesson(lessonInfo.id);
          
          // Process each age variant
          for (const [ageBucket, variant] of Object.entries(lesson.ageVariants)) {
            await this.addLessonContent(`${lesson.id}-${ageBucket}`, {
              title: lesson.title,
              description: lesson.description,
              teachingMoments: variant.teachingMoments || [],
              summary: variant.summary || ''
            });
          }
          
          console.log(`✅ Processed lesson: ${lesson.id}`);
        } catch (error) {
          console.error(`❌ Error processing lesson ${lessonInfo.id}:`, error);
        }
      }

      console.log('✅ Vector database population complete');
      return { success: true, lessonsProcessed: lessons.length };
    } catch (error) {
      console.error('Vector DB population error:', error);
      throw error;
    }
  }

  /**
   * Check if vector database is available
   */
  isAvailable() {
    return this.vectorDB !== null;
  }
}

module.exports = RAGService;













