/**
 * Kelly Generation Queue
 * 
 * COMMUNITY POOLING SYSTEM:
 * - Batch generate "next week's lessons" for everyone
 * - Pool free credits from community BYOK keys
 * - KELLY_KEYS provide platform credits when community runs dry
 * - Fair distribution of generation load
 * 
 * THE FLYWHEEL:
 * 1. Student adds HeyGen key with free credits
 * 2. System uses their key to generate 1 video (credits permitting)
 * 3. That video becomes available to ALL students
 * 4. Community benefits, student feels contributing
 */

const KellyGenerationQueue = {
  // Supabase client
  supabase: null,
  
  // Queue processing state
  isProcessing: false,
  processInterval: null,
  
  // Configuration
  BATCH_SIZE: 5,
  PROCESS_INTERVAL: 30000, // 30 seconds
  MAX_RETRIES: 3,
  
  // Initialize with Supabase client
  init(supabaseClient) {
    this.supabase = supabaseClient;
    console.log('🎬 Generation Queue initialized');
  },

  // Add item to generation queue
  async enqueue(options) {
    const {
      dayNumber,
      phase,
      generationType, // 'video', 'audio', 'visual', 'transcript'
      priority = 0,
      inputData = {}
    } = options;

    if (!this.supabase) {
      console.warn('Queue: No Supabase client');
      return { success: false, error: 'Not initialized' };
    }

    try {
      const { data, error } = await this.supabase
        .from('generation_queue')
        .insert({
          day_number: dayNumber,
          phase,
          generation_type: generationType,
          priority,
          input_data: inputData,
          status: 'pending'
        })
        .select()
        .single();

      if (error) throw error;
      
      console.log(`📥 Queued: Day ${dayNumber} ${phase} (${generationType})`);
      return { success: true, id: data.id };
    } catch (e) {
      console.error('Queue enqueue error:', e);
      return { success: false, error: e.message };
    }
  },

  // Get queue status
  async getStatus() {
    if (!this.supabase) return { pending: 0, processing: 0, completed: 0 };

    try {
      const { data } = await this.supabase
        .from('generation_queue')
        .select('status')
        .order('created_at', { ascending: false })
        .limit(1000);

      const counts = { pending: 0, processing: 0, completed: 0, failed: 0 };
      (data || []).forEach(item => {
        counts[item.status] = (counts[item.status] || 0) + 1;
      });

      return counts;
    } catch (e) {
      return { pending: 0, processing: 0, completed: 0 };
    }
  },

  // Get next items to process
  async getNextBatch() {
    if (!this.supabase) return [];

    try {
      const { data } = await this.supabase
        .from('generation_queue')
        .select('*')
        .eq('status', 'pending')
        .order('priority', { ascending: false })
        .order('created_at', { ascending: true })
        .limit(this.BATCH_SIZE);

      return data || [];
    } catch (e) {
      console.error('Queue fetch error:', e);
      return [];
    }
  },

  // Process a single queue item
  async processItem(item) {
    if (!window.BYOKManager) {
      return { success: false, error: 'BYOK Manager not available' };
    }

    // Mark as processing
    await this.updateItemStatus(item.id, 'processing');

    try {
      let result;

      switch (item.generation_type) {
        case 'video':
          result = await this.generateVideo(item);
          break;
        case 'visual':
          result = await this.generateVisual(item);
          break;
        case 'audio':
          result = await this.generateAudio(item);
          break;
        default:
          result = { success: false, error: `Unknown type: ${item.generation_type}` };
      }

      if (result.success) {
        await this.updateItemStatus(item.id, 'completed', {
          output_url: result.url || result.videoId,
          output_metadata: result.metadata || {}
        });
      } else {
        const attempts = (item.attempts || 0) + 1;
        const status = attempts >= this.MAX_RETRIES ? 'failed' : 'pending';
        await this.updateItemStatus(item.id, status, {
          error_message: result.error,
          attempts
        });
      }

      return result;
    } catch (e) {
      await this.updateItemStatus(item.id, 'failed', {
        error_message: e.message
      });
      return { success: false, error: e.message };
    }
  },

  // Generate video using HeyGen BYOK
  async generateVideo(item) {
    const manager = window.BYOKManager;
    if (!manager.hasProvider('heygen')) {
      return { success: false, error: 'No HeyGen key available' };
    }

    const script = item.input_data?.script || `Welcome to Day ${item.day_number}!`;
    const avatarId = item.input_data?.avatarId || 'Kristin_pubblic_2_20240108';

    const result = await manager.generate('video', {
      script,
      avatarId
    });

    return result;
  },

  // Generate visual using Google/OpenAI BYOK
  async generateVisual(item) {
    const manager = window.BYOKManager;
    if (!manager.hasCapability('image')) {
      return { success: false, error: 'No image generation key available' };
    }

    const prompt = item.input_data?.prompt || `Educational visual for: ${item.phase}`;
    const result = await manager.generate('image', { prompt });

    return result;
  },

  // Generate audio using ElevenLabs/OpenAI BYOK
  async generateAudio(item) {
    const manager = window.BYOKManager;
    if (!manager.hasCapability('tts')) {
      return { success: false, error: 'No TTS key available' };
    }

    const text = item.input_data?.text || `Welcome to Day ${item.day_number}!`;
    const result = await manager.generate('tts', { text });

    return result;
  },

  // Update item status
  async updateItemStatus(id, status, extra = {}) {
    if (!this.supabase) return;

    try {
      const updates = {
        status,
        updated_at: new Date().toISOString(),
        ...extra
      };

      if (status === 'processing') {
        updates.started_at = new Date().toISOString();
      } else if (status === 'completed' || status === 'failed') {
        updates.completed_at = new Date().toISOString();
      }

      await this.supabase
        .from('generation_queue')
        .update(updates)
        .eq('id', id);
    } catch (e) {
      console.error('Queue update error:', e);
    }
  },

  // Start automatic processing
  startProcessing() {
    if (this.processInterval) return;

    this.processInterval = setInterval(async () => {
      if (this.isProcessing) return;
      
      this.isProcessing = true;
      try {
        const batch = await this.getNextBatch();
        for (const item of batch) {
          await this.processItem(item);
        }
      } finally {
        this.isProcessing = false;
      }
    }, this.PROCESS_INTERVAL);

    console.log('🔄 Queue processing started');
  },

  // Stop processing
  stopProcessing() {
    if (this.processInterval) {
      clearInterval(this.processInterval);
      this.processInterval = null;
    }
    console.log('⏹️ Queue processing stopped');
  },

  // Batch queue next week's lessons
  async queueNextWeek() {
    const today = new Date();
    const startDay = Math.floor((today - new Date(today.getFullYear(), 0, 0)) / (1000 * 60 * 60 * 24));
    
    const queued = [];
    
    for (let offset = 1; offset <= 7; offset++) {
      const dayNumber = ((startDay + offset - 1) % 365) + 1;
      
      // Queue video for each day
      const result = await this.enqueue({
        dayNumber,
        phase: 'hook',
        generationType: 'video',
        priority: 7 - offset, // Closer days = higher priority
        inputData: { phase: 'hook' }
      });
      
      if (result.success) {
        queued.push(dayNumber);
      }
    }
    
    console.log(`📅 Queued ${queued.length} days for next week:`, queued);
    return queued;
  },

  // Get community contribution stats
  async getCommunityStats() {
    if (!this.supabase) return null;

    try {
      const { data } = await this.supabase
        .from('generation_queue')
        .select('processed_by, generation_type, status')
        .eq('status', 'completed');

      const stats = {
        totalGenerated: data?.length || 0,
        byType: {},
        contributors: new Set()
      };

      (data || []).forEach(item => {
        stats.byType[item.generation_type] = (stats.byType[item.generation_type] || 0) + 1;
        if (item.processed_by) {
          stats.contributors.add(item.processed_by);
        }
      });

      stats.contributorCount = stats.contributors.size;
      delete stats.contributors;

      return stats;
    } catch (e) {
      return null;
    }
  }
};

// Export
if (typeof window !== 'undefined') {
  window.KellyGenerationQueue = KellyGenerationQueue;
}
