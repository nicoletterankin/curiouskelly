#!/usr/bin/env node
/**
 * RAG Content Population Script
 * Populates vector database with lesson content for RAG retrieval
 * 
 * Usage:
 *   node scripts/populate-rag.js
 *   node scripts/populate-rag.js --lesson leaves-change-color
 *   node scripts/populate-rag.js --dry-run
 */

const RAGService = require('../src/services/rag');
const LessonService = require('../src/services/lessons');

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
  dryRun: args.includes('--dry-run'),
  lesson: args.find(arg => arg.startsWith('--lesson='))?.split('=')[1] || null,
  verbose: args.includes('--verbose') || args.includes('-v'),
  force: args.includes('--force')
};

/**
 * Main population function
 */
async function populateRAG() {
  console.log('🚀 RAG Content Population Tool\n');
  console.log('='.repeat(60));
  
  // Check environment
  if (options.dryRun) {
    console.log('🔍 DRY RUN MODE - No changes will be made');
  }
  
  if (options.verbose) {
    console.log('📝 Verbose mode enabled');
  }
  
  console.log('='.repeat(60));
  console.log('');

  try {
    // Initialize services
    console.log('📦 Initializing services...');
    const ragService = new RAGService();
    const lessonService = new LessonService();

    // Check if RAG is available
    if (!ragService.isAvailable()) {
      console.error('❌ RAG service not available');
      console.error('   Set PINECONE_API_KEY or QDRANT_URL in environment');
      process.exit(1);
    }

    console.log(`✅ Vector DB: ${ragService.vectorDBType}`);
    console.log(`✅ Embedding model: ${ragService.embeddingModel}`);
    console.log('');

    // Get lessons to process
    let lessons;
    if (options.lesson) {
      console.log(`📚 Processing single lesson: ${options.lesson}`);
      const lessonData = await lessonService.loadLesson(options.lesson);
      lessons = [{ id: options.lesson, data: lessonData }];
    } else {
      console.log('📚 Getting all lessons...');
      const allLessons = await lessonService.getAllLessons();
      lessons = await Promise.all(
        allLessons.map(async (l) => ({
          id: l.id,
          data: await lessonService.loadLesson(l.id)
        }))
      );
      console.log(`   Found ${lessons.length} lessons`);
    }

    console.log('');
    console.log('='.repeat(60));
    console.log('📊 Starting Population');
    console.log('='.repeat(60));
    console.log('');

    const stats = {
      lessonsProcessed: 0,
      variantsProcessed: 0,
      vectorsCreated: 0,
      errors: 0,
      startTime: Date.now()
    };

    // Process each lesson
    for (const lesson of lessons) {
      try {
        console.log(`\n📖 Processing: ${lesson.data.title}`);
        console.log(`   ID: ${lesson.id}`);

        // Process each age variant
        for (const [ageBucket, variant] of Object.entries(lesson.data.ageVariants)) {
          const variantId = `${lesson.id}-${ageBucket}`;
          console.log(`   📝 Age ${ageBucket}...`);

          if (options.dryRun) {
            console.log(`      [DRY RUN] Would process variant: ${variantId}`);
            stats.variantsProcessed++;
            continue;
          }

          // Extract content for embedding
          const content = {
            title: lesson.data.title,
            description: lesson.data.description,
            teachingMoments: variant.teachingMoments || [],
            summary: variant.description || '',
            objectives: variant.objectives || [],
            vocabulary: variant.vocabulary?.keyTerms || []
          };

          if (options.verbose) {
            console.log(`      Teaching moments: ${content.teachingMoments.length}`);
            console.log(`      Objectives: ${content.objectives.length}`);
            console.log(`      Vocabulary: ${content.vocabulary.length}`);
          }

          // Add to vector database
          const result = await ragService.addLessonContent(variantId, content);
          
          console.log(`      ✅ Created ${result.vectorsCount} vectors`);
          stats.variantsProcessed++;
          stats.vectorsCreated += result.vectorsCount;
        }

        stats.lessonsProcessed++;
        console.log(`   ✅ Lesson complete`);

      } catch (error) {
        stats.errors++;
        console.error(`   ❌ Error processing ${lesson.id}:`, error.message);
        if (options.verbose) {
          console.error(error.stack);
        }
      }
    }

    // Calculate statistics
    const duration = (Date.now() - stats.startTime) / 1000;
    
    console.log('');
    console.log('='.repeat(60));
    console.log('📈 POPULATION COMPLETE');
    console.log('='.repeat(60));
    console.log(`Lessons processed:  ${stats.lessonsProcessed}`);
    console.log(`Variants processed: ${stats.variantsProcessed}`);
    console.log(`Vectors created:    ${stats.vectorsCreated}`);
    console.log(`Errors:             ${stats.errors}`);
    console.log(`Duration:           ${duration.toFixed(2)}s`);
    console.log(`Rate:               ${(stats.vectorsCreated / duration).toFixed(1)} vectors/sec`);
    console.log('='.repeat(60));

    if (stats.errors > 0) {
      console.log('\n⚠️  Some errors occurred during population');
      process.exit(1);
    } else {
      console.log('\n🎉 All content populated successfully!');
      process.exit(0);
    }

  } catch (error) {
    console.error('\n❌ Fatal error:', error.message);
    if (options.verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

/**
 * Test vector search after population
 */
async function testSearch() {
  console.log('\n🔍 Testing vector search...\n');

  try {
    const ragService = new RAGService();
    
    const testQueries = [
      'Why do leaves change color?',
      'How does photosynthesis work?',
      'What is chlorophyll?'
    ];

    for (const query of testQueries) {
      console.log(`Query: "${query}"`);
      const results = await ragService.search(query, { topK: 3 });
      
      console.log(`   Found ${results.count} results:`);
      results.results.forEach((r, i) => {
        console.log(`   ${i + 1}. [Score: ${r.score.toFixed(3)}] ${r.lessonId}`);
        console.log(`      ${r.text.substring(0, 80)}...`);
      });
      console.log('');
    }

    console.log('✅ Search test complete');
  } catch (error) {
    console.error('❌ Search test failed:', error.message);
  }
}

// Display help
if (args.includes('--help') || args.includes('-h')) {
  console.log(`
RAG Content Population Tool

Usage:
  node scripts/populate-rag.js [options]

Options:
  --lesson=ID       Process only specified lesson
  --dry-run         Preview what would be done without making changes
  --verbose, -v     Show detailed progress
  --force           Force re-population even if content exists
  --test-search     Run search tests after population
  --help, -h        Show this help message

Examples:
  # Populate all lessons
  node scripts/populate-rag.js

  # Populate single lesson
  node scripts/populate-rag.js --lesson=leaves-change-color

  # Dry run to see what would happen
  node scripts/populate-rag.js --dry-run --verbose

  # Populate and test
  node scripts/populate-rag.js --test-search

Environment Variables:
  PINECONE_API_KEY        Pinecone API key
  PINECONE_INDEX          Pinecone index name (default: curious-kellly-lessons)
  
  OR
  
  QDRANT_URL              Qdrant server URL
  QDRANT_API_KEY          Qdrant API key (if required)
  QDRANT_COLLECTION       Qdrant collection name (default: curious-kellly-lessons)

  OPENAI_API_KEY          OpenAI API key for embeddings
  EMBEDDING_MODEL         Embedding model (default: text-embedding-3-small)
  `);
  process.exit(0);
}

// Run population
populateRAG()
  .then(() => {
    if (args.includes('--test-search')) {
      return testSearch();
    }
  })
  .catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });



