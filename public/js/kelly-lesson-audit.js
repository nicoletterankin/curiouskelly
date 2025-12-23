/**
 * Kelly Lesson Inspector - COMPREHENSIVE Asset Collection System
 * 
 * Collects and displays ALL lesson assets across ALL pipelines:
 * - Videos: All archetypes, ages, languages, phases from Supabase + local files
 * - Audio: All variants from JSON + Supabase
 * - Images: Thumbnails, infographics, visuals from multiple sources
 * - Text: All language variants, all phases, all options
 * - Variants: 12 archetypes × 6 age buckets × 3 languages × 7 phases
 */

(function() {
  'use strict';

  const LessonInspector = {
    currentAudit: null,
    mediaPlayers: {},
    supabaseClient: null,

    /**
     * Initialize Supabase client if available
     */
    initSupabase() {
      // Method 1: Use existing singleton
      if (window.supabaseClient) {
        this.supabaseClient = window.supabaseClient;
        return true;
      }
      
      // Method 2: Use getSupabase() singleton function
      if (typeof window.getSupabase === 'function') {
        const client = window.getSupabase();
        if (client) {
          this.supabaseClient = client;
          return true;
        }
      }
      
      // Method 3: Try to create from window config
      if (typeof window.supabase !== 'undefined' && window.supabase.createClient) {
        // Try multiple config sources
        const supabaseUrl = 
          window.SUPABASE_URL ||
          (window.CONFIG && window.CONFIG.SUPABASE_URL) ||
          (window.KELLY_CONFIG && window.KELLY_CONFIG.supabaseUrl) ||
          'https://tvjalxxsyryjphkforjv.supabase.co';
        
        const supabaseKey = 
          window.SUPABASE_ANON_KEY ||
          (window.CONFIG && window.CONFIG.SUPABASE_ANON_KEY) ||
          (window.KELLY_CONFIG && window.KELLY_CONFIG.supabaseKey) ||
          'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';
        
        if (supabaseUrl && supabaseKey) {
          try {
            this.supabaseClient = window.supabase.createClient(supabaseUrl, supabaseKey);
            return true;
          } catch (e) {
            console.warn('[LessonInspector] Failed to create Supabase client:', e);
          }
        }
      }
      
      return false;
    },

    /**
     * Get COMPLETE lesson audit - collects from ALL sources
     */
    async getFullAudit(dayNumber) {
      const paddedDay = String(dayNumber).padStart(3, '0');
      const audit = {
        dayNumber,
        paddedDay,
        date: null,
        sources: {
          json: null,
          api: null,
          supabase: null,
          localFiles: []
        },
        assets: {
          json: null,
          videos: {
            supabase: [],
            local: [],
            total: 0
          },
          audio: {
            json: [],
            supabase: [],
            local: [],
            api: [],
            total: 0
          },
          images: {
            thumbnails: [],
            infographics: [],
            visuals: [],
            optionCards: [],
            kellyResponses: [],
            total: 0
          },
          transcripts: [],
          alignments: []
        },
        metadata: {},
        phases: [],
        variants: {
          languages: new Set(),
          ageBuckets: new Set(),
          archetypes: new Set(),
          phases: new Set(),
          tracks: new Set()
        },
        growTrack: null,
        pipelines: {
          elevenlabs: { status: 'unknown', count: 0 },
          heygen: { status: 'unknown', count: 0 },
          iclone: { status: 'unknown', count: 0 },
          audio2face: { status: 'unknown', count: 0 },
          supabase: { status: 'unknown', count: 0 },
          cloudflare: { status: 'unknown', count: 0 },
          local: { status: 'unknown', count: 0 }
        },
        errors: [],
        warnings: []
      };

      try {
        // Initialize Supabase
        this.initSupabase();

        // Get date for this day
        if (window.KellyTime?.dayNumberToDate) {
          audit.date = window.KellyTime.dayNumberToDate(dayNumber);
        }

        // 1. Load JSON lesson file (primary source of truth) - BOTH Learn and Grow tracks
        await this.loadJsonLesson(audit, dayNumber, 'learn');
        await this.loadJsonLesson(audit, dayNumber, 'grow');

        // 2. Query Supabase for ALL video assets (both tracks)
        await this.loadSupabaseVideos(audit, dayNumber);

        // 3. Query Supabase for visual commons
        await this.loadSupabaseVisuals(audit, dayNumber);

        // 4. Check local file system for generated assets (ALL locations)
        await this.loadLocalAssets(audit, dayNumber, paddedDay);

        // 5. Check API endpoint
        await this.loadApiLesson(audit, dayNumber);
        
        // 6. Load Grow track from Supabase
        await this.loadGrowTrack(audit, dayNumber);
        
        // 7. Check API endpoints for videos/audio
        await this.loadApiAssets(audit, dayNumber);

        // 6. Aggregate variant information
        this.aggregateVariants(audit);

        // 7. Calculate pipeline status
        this.calculatePipelineStatus(audit);

      } catch (e) {
        audit.errors.push(`Audit failed: ${e.message}`);
        console.error('[LessonInspector] Audit error:', e);
      }

      return audit;
    },

    /**
     * Load JSON lesson file (Learn or Grow track)
     */
    async loadJsonLesson(audit, dayNumber, track = 'learn') {
      try {
        const jsonRes = await fetch(`/lessons/day-${dayNumber}.json`);
        if (jsonRes.ok) {
          const jsonData = await jsonRes.json();
          
          if (track === 'learn') {
            audit.sources.json = `/lessons/day-${dayNumber}.json`;
            audit.assets.json = jsonData;
            audit.metadata = this.extractMetadata(jsonData);
            audit.phases = this.extractPhases(jsonData);
            
            // Extract audio paths from JSON
            audit.assets.audio.json = this.extractAudioPaths(jsonData);
            audit.assets.audio.total += audit.assets.audio.json.length;
            
            // Extract language variants
            if (jsonData.meta?.languages) {
              jsonData.meta.languages.forEach(lang => audit.variants.languages.add(lang));
            }
            
            // Extract phase names
            if (jsonData.phases) {
              Object.keys(jsonData.phases).forEach(phase => audit.variants.phases.add(phase));
            }
          }
          
          // Extract Grow track if present
          if (jsonData.growTrack) {
            audit.assets.growTrack = jsonData.growTrack;
            audit.variants.tracks.add('grow');
          }
          
          if (jsonData.meta) {
            audit.variants.tracks.add('learn');
          }
        } else {
          if (track === 'learn') {
            audit.errors.push(`JSON file not found: /lessons/day-${dayNumber}.json`);
          }
        }
      } catch (e) {
        if (track === 'learn') {
          audit.errors.push(`Failed to load JSON: ${e.message}`);
        }
      }
    },

    /**
     * Load ALL videos from Supabase kelly_video_assets
     */
    async loadSupabaseVideos(audit, dayNumber) {
      if (!this.supabaseClient) {
        audit.warnings.push('Supabase client not available - skipping video query');
        return;
      }

      try {
        // Query ALL video assets for this day (all archetypes, ages, languages, phases)
        const { data: videos, error } = await this.supabaseClient
          .from('kelly_video_assets')
          .select(`
            id,
            lesson_day,
            phase,
            age_bucket,
            language,
            archetype,
            video_public_url,
            video_storage_path,
            video_duration_ms,
            video_file_size_bytes,
            video_resolution,
            status,
            model_used,
            lip_sync_quality_score,
            video_quality_score,
            is_approved,
            created_at
          `)
          .eq('lesson_day', dayNumber)
          .in('status', ['completed', 'generating', 'pending']);

        if (error) {
          audit.warnings.push(`Supabase video query error: ${error.message}`);
          return;
        }

        if (videos && videos.length > 0) {
          audit.assets.videos.supabase = videos;
          audit.assets.videos.total += videos.length;
          audit.pipelines.heygen.count = videos.length;
          audit.pipelines.heygen.status = 'available';
          
          // Track variants
          videos.forEach(v => {
            if (v.age_bucket) audit.variants.ageBuckets.add(v.age_bucket);
            if (v.language) audit.variants.languages.add(v.language);
            if (v.archetype) audit.variants.archetypes.add(v.archetype);
            if (v.phase) audit.variants.phases.add(v.phase);
          });
        } else {
          audit.pipelines.heygen.status = 'no_assets';
        }
      } catch (e) {
        audit.errors.push(`Supabase video query failed: ${e.message}`);
      }
    },

    /**
     * Load visual commons from Supabase
     */
    async loadSupabaseVisuals(audit, dayNumber) {
      if (!this.supabaseClient) {
        return;
      }

      try {
        const { data: visuals, error } = await this.supabaseClient
          .from('visual_commons')
          .select(`
            id,
            day_number,
            public_url,
            thumbnail_url,
            style,
            phase,
            model_used,
            age_group,
            language,
            status,
            created_at
          `)
          .eq('day_number', dayNumber)
          .eq('status', 'active')
          .limit(100);

        if (error) {
          audit.warnings.push(`Visual commons query error: ${error.message}`);
          return;
        }

        if (visuals && visuals.length > 0) {
          audit.assets.images.visuals = visuals;
          audit.assets.images.total += visuals.length;
          audit.pipelines.supabase.count = visuals.length;
          audit.pipelines.supabase.status = 'available';
        }
      } catch (e) {
        audit.warnings.push(`Visual commons query failed: ${e.message}`);
      }
    },

    /**
     * Load local file system assets - COMPREHENSIVE SEARCH
     */
    async loadLocalAssets(audit, dayNumber, paddedDay) {
      const localPaths = [
        // Thumbnails (multiple locations)
        `/generated-visuals/day-${paddedDay}/thumbnail.png`,
        `/generated-assets/day-${paddedDay}/thumbnail.png`,
        `/assets/kelly/production/thumbnails/january/lesson-${dayNumber}.webp`,
        `/images/lessons/day-${dayNumber}.jpg`,
        `/images/lessons/day-${dayNumber}.png`,
        
        // Phase images (public/kelly/phases/)
        `/kelly/phases/${paddedDay}/hook.png`,
        `/kelly/phases/${paddedDay}/q1.png`,
        `/kelly/phases/${paddedDay}/q2.png`,
        `/kelly/phases/${paddedDay}/q3.png`,
        `/kelly/phases/${paddedDay}/wisdom.png`,
        `/kelly/phases/${paddedDay}/cliff.png`,
        `/kelly/phases/${paddedDay}/outro.png`,
        
        // Infographics (multiple locations)
        `/generated-visuals/day-${paddedDay}/infographic-1.png`,
        `/generated-visuals/day-${paddedDay}/infographic-2.png`,
        `/generated-visuals/day-${paddedDay}/infographic-3.png`,
        `/generated-visuals/day-${paddedDay}/illustration.png`,
        `/kelly/infographics/${dayNumber}/infographic-brain-scan.png`,
        `/kelly/infographics/${dayNumber}/infographic-how-to-visualize.png`,
        `/kelly/infographics/${dayNumber}/infographic-olympic-athletes.png`,
        `/kelly/infographics/${dayNumber}/infographic-piano-study.png`,
        `/kelly/infographics/${dayNumber}/background-cosmic-mind.png`,
        
        // Unified factory assets (all archetypes)
        `/generated-assets/unified-factory/day-${paddedDay}/explorer/infographics/hook.png`,
        `/generated-assets/unified-factory/day-${paddedDay}/explorer/infographics/fact1.png`,
        `/generated-assets/unified-factory/day-${paddedDay}/explorer/infographics/fact2.png`,
        `/generated-assets/unified-factory/day-${paddedDay}/explorer/infographics/fact3.png`,
        `/generated-assets/unified-factory/day-${paddedDay}/explorer/infographics/wisdom.png`,
        `/generated-assets/unified-factory/day-${paddedDay}/scientist/infographics/hook.png`,
        `/generated-assets/unified-factory/day-${paddedDay}/scientist/infographics/fact1.png`,
        `/generated-assets/unified-factory/day-${paddedDay}/rebel/infographics/hook.png`,
        `/generated-assets/unified-factory/day-${paddedDay}/rebel/infographics/fact1.png`,
        
        // Videos (local file system - multiple locations)
        `/kelly/videos/${paddedDay}/welcome.mp4`,
        `/kelly/videos/${paddedDay}/kv1.mp4`,
        `/video/${dayNumber}/scientist_adult/hook.mp4`,
        `/video/${dayNumber}/scientist_adult/q1.mp4`,
        `/video/${dayNumber}/scientist_adult/q2.mp4`,
        `/video/${dayNumber}/scientist_adult/q3.mp4`,
        `/video/${dayNumber}/scientist_adult/wisdom.mp4`,
        `/video/${dayNumber}/scientist_adult/outro.mp4`,
        `/videos/summary/day-${dayNumber}.mp4`,
        `/video/351/scientist_adult/hook.mp4`, // Example pattern
        `/video/351/scientist_adult/q1.mp4`,
        `/video/351/scientist_adult/q2.mp4`,
        `/video/351/scientist_adult/q3.mp4`,
        `/video/351/scientist_adult/wisdom.mp4`,
        `/video/351/scientist_adult/outro.mp4`,
        
        // Generated videos (golden-lesson-hd structure)
        `/generated-videos/golden-lesson-hd/day_${paddedDay}_Hook_The_Scientist/final_hd.mp4`,
        `/generated-videos/golden-lesson-hd/day_${paddedDay}_Hook_The_Explorer/final_hd.mp4`,
        `/generated-videos/golden-lesson-hd/day_${paddedDay}_Hook_The_Rebel/final_hd.mp4`,
        `/generated-videos/golden-lesson-hd/day_${paddedDay}_Fact1_The_Scientist/final_hd.mp4`,
        `/generated-videos/golden-lesson-hd/day_${paddedDay}_Fact2_The_Scientist/final_hd.mp4`,
        `/generated-videos/golden-lesson-hd/day_${paddedDay}_Fact3_The_Scientist/final_hd.mp4`,
        `/generated-videos/golden-lesson-hd/day_${paddedDay}_Wisdom_The_Scientist/final_hd.mp4`,
        
        // Audio files (from unified factory)
        `/generated-assets/unified-factory/day-${paddedDay}/explorer/audio/hook_main.mp3`,
        `/generated-assets/unified-factory/day-${paddedDay}/explorer/audio/fact1_main.mp3`,
        `/generated-assets/unified-factory/day-${paddedDay}/explorer/audio/fact2_main.mp3`,
        `/generated-assets/unified-factory/day-${paddedDay}/explorer/audio/fact3_main.mp3`,
        `/generated-assets/unified-factory/day-${paddedDay}/explorer/audio/wisdom_main.mp3`,
        `/generated-assets/unified-factory/day-${paddedDay}/scientist/audio/hook_main.mp3`,
        `/generated-assets/unified-factory/day-${paddedDay}/scientist/audio/fact1_main.mp3`,
        `/generated-assets/unified-factory/day-${paddedDay}/rebel/audio/hook_main.mp3`,
        
        // Social media assets
        `/kelly/social/${dayNumber}/social-ig-carousel-1.png`,
        `/kelly/social/${dayNumber}/social-ig-carousel-2.png`,
        `/kelly/social/${dayNumber}/social-quote-card.png`,
        `/kelly/social/${dayNumber}/social-tiktok-thumb.png`,
        `/kelly/social/${dayNumber}/social-twitter-header.png`,
      ];

      const foundAssets = [];
      
      for (const path of localPaths) {
        try {
          const res = await fetch(path, { method: 'HEAD' });
          if (res.ok) {
            foundAssets.push({
              path,
              type: this.classifyAssetPath(path),
              status: 'found'
            });
            audit.sources.localFiles.push(path);
          }
        } catch (_) {
          // Asset doesn't exist, skip
        }
      }

      // Categorize found assets
      foundAssets.forEach(asset => {
        if (asset.type === 'thumbnail') {
          audit.assets.images.thumbnails.push(asset);
          audit.assets.images.total++;
        } else if (asset.type === 'infographic') {
          audit.assets.images.infographics.push(asset);
          audit.assets.images.total++;
        } else if (asset.path.includes('.mp4') || asset.path.includes('.webm')) {
          audit.assets.videos.local.push(asset);
          audit.assets.videos.total++;
        } else if (asset.path.includes('.mp3') || asset.path.includes('.wav') || asset.path.includes('.ogg')) {
          audit.assets.audio.local.push(asset);
          audit.assets.audio.total++;
        } else if (asset.type === 'illustration') {
          audit.assets.images.infographics.push(asset);
          audit.assets.images.total++;
        }
      });

      if (foundAssets.length > 0) {
        audit.pipelines.local.status = 'available';
        audit.pipelines.local.count = foundAssets.length;
      }
    },

    /**
     * Classify asset path type
     */
    classifyAssetPath(path) {
      if (path.includes('thumbnail')) return 'thumbnail';
      if (path.includes('infographic')) return 'infographic';
      if (path.includes('illustration')) return 'illustration';
      if (path.includes('option')) return 'optionCard';
      if (path.includes('kelly-response')) return 'kellyResponse';
      if (path.includes('.mp4') || path.includes('.webm')) return 'video';
      if (path.includes('.mp3') || path.includes('.wav') || path.includes('.ogg')) return 'audio';
      if (path.includes('social')) return 'social';
      return 'unknown';
    },

    /**
     * Load lesson from API endpoint
     */
    async loadApiLesson(audit, dayNumber) {
      try {
        const apiRes = await fetch(`/api/lessons/${dayNumber}?archetype=The%20Scientist&ageBucket=adult`);
        if (apiRes.ok) {
          audit.sources.api = `/api/lessons/${dayNumber}`;
          audit.pipelines.supabase.data = await apiRes.json();
          audit.pipelines.supabase.status = 'available';
        } else {
          audit.pipelines.supabase.status = 'unavailable';
        }
      } catch (e) {
        audit.warnings.push(`API error: ${e.message}`);
      }
    },

    /**
     * Extract metadata from lesson JSON
     */
    extractMetadata(json) {
      if (!json) return {};
      
      return {
        topic: typeof json.meta?.topic === 'object' ? json.meta.topic.en : json.meta?.topic,
        headline: typeof json.headline === 'object' ? json.headline.en : json.headline,
        universalTruth: typeof json.universal_truth === 'object' ? json.universal_truth.en : json.universal_truth,
        category: json.meta?.category,
        emoji: json.meta?.emoji,
        languages: json.meta?.languages || []
      };
    },

    /**
     * Extract all phases with full content (all languages)
     */
    extractPhases(json) {
      if (!json?.phases) return [];
      
      const phases = [];
      for (const [phaseName, phaseData] of Object.entries(json.phases)) {
        const phase = {
          name: phaseName,
          languages: {},
          options: [],
          audio: {},
          video: {}
        };

        // Extract multilingual content
        if (json.meta?.languages) {
          json.meta.languages.forEach(lang => {
            phase.languages[lang] = {
              script: typeof phaseData.script === 'object' 
                ? phaseData.script[lang] || phaseData.script.en
                : phaseData.script || '',
              prompt: typeof phaseData.prompt === 'object'
                ? phaseData.prompt[lang] || phaseData.prompt.en
                : phaseData.prompt || '',
              title: typeof phaseData.title === 'object'
                ? phaseData.title[lang] || phaseData.title.en
                : phaseData.title || ''
            };
          });
        }

        // Extract options (with multilingual responses)
        if (phaseData.options) {
          phase.options = phaseData.options.map(opt => {
            const option = {
              letter: opt.letter,
              quality: opt.quality,
              text: {},
              response: {}
            };

            json.meta.languages.forEach(lang => {
              option.text[lang] = typeof opt.text === 'object' 
                ? opt.text[lang] || opt.text.en
                : opt.text || '';
              
              if (opt.response) {
                option.response[lang] = typeof opt.response === 'object'
                  ? opt.response[lang] || opt.response.en
                  : opt.response || '';
              }
            });

            return option;
          });
        }

        // Extract audio/video paths
        if (phaseData.talk?.audio) {
          phase.audio = phaseData.talk.audio;
        }
        if (phaseData.talk?.video) {
          phase.video = phaseData.talk.video;
        }

        phases.push(phase);
      }
      
      return phases;
    },

    /**
     * Extract audio file paths from lesson JSON
     */
    extractAudioPaths(json) {
      const paths = [];
      if (!json.phases) return paths;

      for (const [phase, phaseData] of Object.entries(json.phases)) {
        if (phaseData.talk?.audio) {
          paths.push({
            phase,
            path: phaseData.talk.audio,
            type: 'elevenlabs'
          });
        }
      }

      return paths;
    },

    /**
     * Load assets from API endpoints
     */
    async loadApiAssets(audit, dayNumber) {
      // Check video API endpoint
      try {
        const videoRes = await fetch(`/api/kelly-video?day=${dayNumber}`);
        if (videoRes.ok) {
          const videoData = await videoRes.json();
          if (videoData.url) {
            audit.assets.videos.api = audit.assets.videos.api || [];
            audit.assets.videos.api.push({
              url: videoData.url,
              phase: videoData.phase,
              source: 'api'
            });
            audit.assets.videos.total++;
          }
        }
      } catch (e) {
        // API endpoint may not exist, skip silently
      }
      
      // Check ElevenLabs video API
      try {
        const elevenlabsRes = await fetch(`/api/elevenlabs-video?day=${dayNumber}&phase=hook&ageBucket=adult`);
        if (elevenlabsRes.ok) {
          const videoData = await elevenlabsRes.json();
          if (videoData.videoUrl) {
            audit.assets.videos.api = audit.assets.videos.api || [];
            audit.assets.videos.api.push({
              url: videoData.videoUrl,
              phase: 'hook',
              source: 'elevenlabs-api',
              cached: videoData.cached || false
            });
            audit.assets.videos.total++;
            audit.pipelines.elevenlabs.status = 'available';
            audit.pipelines.elevenlabs.count++;
          }
        }
      } catch (e) {
        // API endpoint may not exist, skip silently
      }
    },

    /**
     * Load Grow track from Supabase
     */
    async loadGrowTrack(audit, dayNumber) {
      if (!this.supabaseClient) {
        return;
      }

      try {
        const { data: growLesson, error } = await this.supabaseClient
          .from('core_lessons')
          .select('*')
          .eq('day_number', dayNumber)
          .eq('track', 'grow')
          .maybeSingle();

        if (error) {
          audit.warnings.push(`Grow track query error: ${error.message}`);
          return;
        }

        if (growLesson) {
          audit.assets.growTrackSupabase = growLesson;
          audit.variants.tracks.add('grow');
        }
      } catch (e) {
        audit.warnings.push(`Grow track query failed: ${e.message}`);
      }
    },

    /**
     * Aggregate variant information
     */
    aggregateVariants(audit) {
      // Convert Sets to Arrays for display
      audit.variants.languages = Array.from(audit.variants.languages);
      audit.variants.ageBuckets = Array.from(audit.variants.ageBuckets);
      audit.variants.archetypes = Array.from(audit.variants.archetypes);
      audit.variants.phases = Array.from(audit.variants.phases);
      audit.variants.tracks = Array.from(audit.variants.tracks);
    },

    /**
     * Calculate pipeline status
     */
    calculatePipelineStatus(audit) {
      // ElevenLabs: Check for audio files
      if (audit.assets.audio.total > 0) {
        audit.pipelines.elevenlabs.status = 'available';
        audit.pipelines.elevenlabs.count = audit.assets.audio.total;
      } else {
        audit.pipelines.elevenlabs.status = 'no_assets';
      }

      // Overall status
      const hasAnyAssets = 
        audit.assets.videos.total > 0 ||
        audit.assets.audio.total > 0 ||
        audit.assets.images.total > 0;
      
      if (!hasAnyAssets) {
        audit.warnings.push('No assets found for this lesson');
      }
    },

    /**
     * Render comprehensive inspector panel
     */
    renderInspectorPanel(audit) {
      const dateStr = audit.date 
        ? audit.date.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })
        : 'Unknown date';

      return `
        <div class="lesson-inspector-panel">
          <div class="inspector-header">
            <div>
              <h2>Day ${audit.dayNumber} • ${dateStr}</h2>
              <div class="inspector-subtitle">${audit.metadata.topic || 'Lesson Inspector'}</div>
            </div>
            <button class="inspector-close" onclick="LessonInspector.close()">×</button>
          </div>
          
          <div class="inspector-content">
            <!-- Quick Stats -->
            <section class="inspector-section stats-grid">
              <div class="stat-card">
                <div class="stat-value">${audit.assets.videos.total}</div>
                <div class="stat-label">Videos</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">${audit.assets.audio.total}</div>
                <div class="stat-label">Audio</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">${audit.assets.images.total}</div>
                <div class="stat-label">Images</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">${audit.variants.languages.length}</div>
                <div class="stat-label">Languages</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">${audit.variants.archetypes.length}</div>
                <div class="stat-label">Archetypes</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">${audit.variants.ageBuckets.length}</div>
                <div class="stat-label">Age Variants</div>
              </div>
            </section>

            <!-- Metadata -->
            <section class="inspector-section">
              <h3>📄 Lesson Metadata</h3>
              <div class="metadata-grid">
                ${audit.metadata.topic ? `<div><label>Topic:</label><div>${audit.metadata.topic}</div></div>` : ''}
                ${audit.metadata.headline ? `<div><label>Headline:</label><div>${audit.metadata.headline}</div></div>` : ''}
                ${audit.metadata.universalTruth ? `<div><label>Universal Truth:</label><div>${audit.metadata.universalTruth}</div></div>` : ''}
                ${audit.metadata.category ? `<div><label>Category:</label><div>${audit.metadata.category}</div></div>` : ''}
                ${audit.metadata.emoji ? `<div><label>Emoji:</label><div>${audit.metadata.emoji}</div></div>` : ''}
              </div>
            </section>

            <!-- Variants Summary -->
            <section class="inspector-section">
              <h3>🎭 Available Variants</h3>
              <div class="variants-grid">
                <div><strong>Tracks:</strong> ${audit.variants.tracks.join(', ') || 'None'}</div>
                <div><strong>Languages:</strong> ${audit.variants.languages.join(', ') || 'None'}</div>
                <div><strong>Age Buckets:</strong> ${audit.variants.ageBuckets.join(', ') || 'None'}</div>
                <div><strong>Archetypes:</strong> ${audit.variants.archetypes.join(', ') || 'None'}</div>
                <div><strong>Phases:</strong> ${audit.variants.phases.join(', ') || 'None'}</div>
              </div>
            </section>

            <!-- Grow Track Section -->
            ${audit.assets.growTrack || audit.assets.growTrackSupabase ? `
              <section class="inspector-section">
                <h3>🧠 Grow Track</h3>
                ${this.renderGrowTrack(audit)}
              </section>
            ` : ''}

            <!-- Videos Section -->
            ${audit.assets.videos.total > 0 ? `
              <section class="inspector-section">
                <h3>🎬 Videos (${audit.assets.videos.total})</h3>
                ${this.renderVideoSection(audit)}
              </section>
            ` : `
              <section class="inspector-section">
                <h3>🎬 Videos</h3>
                <div class="empty-state">
                  No videos found. Checked:
                  <ul style="margin-top: 8px; padding-left: 20px; font-size: 12px; color: rgba(255,255,255,0.5);">
                    <li>Supabase kelly_video_assets table</li>
                    <li>Local file system (${audit.assets.videos.local.length} paths checked)</li>
                    <li>API endpoints (/api/kelly-video, /api/elevenlabs-video)</li>
                  </ul>
                </div>
              </section>
            `}

            <!-- Audio Section -->
            ${audit.assets.audio.total > 0 ? `
              <section class="inspector-section">
                <h3>🎵 Audio (${audit.assets.audio.total})</h3>
                ${this.renderAudioSection(audit)}
              </section>
            ` : ''}

            <!-- Images Section -->
            ${audit.assets.images.total > 0 ? `
              <section class="inspector-section">
                <h3>🖼️ Images (${audit.assets.images.total})</h3>
                ${this.renderImageSection(audit)}
              </section>
            ` : ''}

            <!-- Phases Content -->
            ${audit.phases?.length > 0 ? `
              <section class="inspector-section">
                <h3>📝 Lesson Phases (${audit.phases.length})</h3>
                ${this.renderPhasesSection(audit)}
              </section>
            ` : ''}

            <!-- Pipeline Status -->
            <section class="inspector-section">
              <h3>⚙️ Pipeline Status</h3>
              ${this.renderPipelineStatus(audit)}
            </section>

            <!-- Errors & Warnings -->
            ${audit.errors.length > 0 ? `
              <section class="inspector-section inspector-errors">
                <h3>❌ Errors</h3>
                <ul>${audit.errors.map(e => `<li>${e}</li>`).join('')}</ul>
              </section>
            ` : ''}

            ${audit.warnings.length > 0 ? `
              <section class="inspector-section inspector-warnings">
                <h3>⚠️ Warnings</h3>
                <ul>${audit.warnings.map(w => `<li>${w}</li>`).join('')}</ul>
              </section>
            ` : ''}
          </div>

          <div class="inspector-actions">
            <button class="btn-primary" onclick="window.location.href='/learn.html?day=${audit.dayNumber}'">Open in Learn</button>
            <button class="btn-secondary" onclick="LessonInspector.downloadAudit(${audit.dayNumber})">Download JSON</button>
          </div>
        </div>
      `;
    },

    /**
     * Render video section with filters
     */
    renderVideoSection(audit) {
      const allVideos = [
        ...(audit.assets.videos.supabase || []),
        ...(audit.assets.videos.local || []),
        ...(audit.assets.videos.api || [])
      ];
      
      if (allVideos.length === 0) {
        return '<div class="empty-state">No videos found</div>';
      }
      
      // Group Supabase videos by phase
      const supabaseVideos = audit.assets.videos.supabase || [];

      // Group Supabase videos by phase
      const byPhase = {};
      supabaseVideos.forEach(v => {
        const phase = v.phase || 'unknown';
        if (!byPhase[phase]) byPhase[phase] = [];
        byPhase[phase].push(v);
      });
      
      let html = '';
      
      // Render Supabase videos grouped by phase
      if (Object.keys(byPhase).length > 0) {
        html += Object.entries(byPhase).map(([phase, phaseVideos]) => `
          <div class="phase-video-group">
            <h4>${phase.toUpperCase()} - Supabase (${phaseVideos.length})</h4>
            ${phaseVideos.map((video, idx) => `
              <div class="media-item video-item">
                <div class="media-info">
                  <strong>${video.archetype || 'Default'}</strong> • 
                  ${video.age_bucket || 'N/A'} • 
                  ${video.language || 'N/A'} • 
                  ${video.video_resolution || 'Unknown'} • 
                  ${video.video_duration_ms ? Math.round(video.video_duration_ms / 1000) + 's' : 'Unknown'}
                  ${video.is_approved ? ' ✓ Approved' : ''}
                </div>
                ${video.video_public_url ? `
                  <video controls class="media-player" preload="metadata" style="width: 100%; max-width: 600px;">
                    <source src="${video.video_public_url}" type="video/mp4">
                    Your browser does not support video playback.
                  </video>
                ` : '<div class="no-asset">Video URL not available</div>'}
              </div>
            `).join('')}
          </div>
        `).join('');
      }
      
      // Render local videos
      const localVideos = audit.assets.videos.local || [];
      if (localVideos.length > 0) {
        html += `
          <div class="phase-video-group">
            <h4>Local Files (${localVideos.length})</h4>
            ${localVideos.map((video, idx) => `
              <div class="media-item video-item">
                <div class="media-info">
                  <strong>Local File</strong> • ${video.path}
                </div>
                <video controls class="media-player" preload="metadata" style="width: 100%; max-width: 600px;">
                  <source src="${video.path}" type="video/mp4">
                  Your browser does not support video playback.
                </video>
              </div>
            `).join('')}
          </div>
        `;
      }
      
      // Render API videos
      const apiVideos = audit.assets.videos.api || [];
      if (apiVideos.length > 0) {
        html += `
          <div class="phase-video-group">
            <h4>API Endpoints (${apiVideos.length})</h4>
            ${apiVideos.map((video, idx) => `
              <div class="media-item video-item">
                <div class="media-info">
                  <strong>${video.source}</strong> • ${video.phase || 'N/A'} ${video.cached ? '(cached)' : ''}
                </div>
                <video controls class="media-player" preload="metadata" style="width: 100%; max-width: 600px;">
                  <source src="${video.url}" type="video/mp4">
                  Your browser does not support video playback.
                </video>
              </div>
            `).join('')}
          </div>
        `;
      }
      
      return html;
    },

    /**
     * Render audio section
     */
    renderAudioSection(audit) {
      const audioFiles = [...audit.assets.audio.json];
      
      if (audioFiles.length === 0) {
        return '<div class="empty-state">No audio files found</div>';
      }

      return audioFiles.map((audio, idx) => `
        <div class="media-item audio-item">
          <div class="media-info">
            <strong>${audio.phase}</strong> • ${audio.type}
          </div>
          <audio controls class="media-player" preload="metadata" style="width: 100%; max-width: 600px;">
            <source src="${audio.path}" type="audio/mpeg">
            Your browser does not support audio playback.
          </audio>
        </div>
      `).join('');
    },

    /**
     * Render image section
     */
    renderImageSection(audit) {
      const sections = [];

      // Thumbnails
      if (audit.assets.images.thumbnails.length > 0) {
        sections.push(`
          <div class="image-group">
            <h4>Thumbnails (${audit.assets.images.thumbnails.length})</h4>
            <div class="images-grid">
              ${audit.assets.images.thumbnails.map(img => `
                <div class="image-item">
                  <img src="${img.path}" alt="Thumbnail" loading="lazy" onclick="this.classList.toggle('expanded')">
                  <div class="image-path">${img.path}</div>
                </div>
              `).join('')}
            </div>
          </div>
        `);
      }

      // Infographics
      if (audit.assets.images.infographics.length > 0) {
        sections.push(`
          <div class="image-group">
            <h4>Infographics (${audit.assets.images.infographics.length})</h4>
            <div class="images-grid">
              ${audit.assets.images.infographics.map(img => `
                <div class="image-item">
                  <img src="${img.path}" alt="Infographic" loading="lazy" onclick="this.classList.toggle('expanded')">
                  <div class="image-path">${img.path}</div>
                </div>
              `).join('')}
            </div>
          </div>
        `);
      }

      // Visual Commons
      if (audit.assets.images.visuals.length > 0) {
        sections.push(`
          <div class="image-group">
            <h4>Visual Commons (${audit.assets.images.visuals.length})</h4>
            <div class="images-grid">
              ${audit.assets.images.visuals.map(visual => `
                <div class="image-item">
                  <img src="${visual.public_url}" alt="Visual" loading="lazy" onclick="this.classList.toggle('expanded')">
                  <div class="image-info">${visual.style || 'default'} • ${visual.phase || 'unknown'}</div>
                </div>
              `).join('')}
            </div>
          </div>
        `);
      }

      return sections.join('');
    },

    /**
     * Render phases section with all languages
     */
    renderPhasesSection(audit) {
      return audit.phases.map((phase, idx) => {
        const defaultLang = phase.languages.en || phase.languages[Object.keys(phase.languages)[0]];
        
        return `
          <div class="phase-card">
            <div class="phase-header">
              <span class="phase-name">${phase.name.toUpperCase()}</span>
              ${phase.audio ? `<button class="play-audio-btn" onclick="LessonInspector.playAudio('${phase.audio}', this)">▶ Play Audio</button>` : ''}
            </div>
            
            <!-- Language Tabs -->
            ${Object.keys(phase.languages).length > 1 ? `
              <div class="language-tabs">
                ${Object.keys(phase.languages).map(lang => `
                  <button class="lang-tab" onclick="LessonInspector.switchLanguage(this, 'phase-${idx}', '${lang}')">${lang.toUpperCase()}</button>
                `).join('')}
              </div>
            ` : ''}
            
            <!-- Phase Content (default language) -->
            <div class="phase-content" id="phase-${idx}">
              ${defaultLang.script ? `
                <div class="phase-script">
                  <strong>Script:</strong>
                  <p>${defaultLang.script}</p>
                </div>
              ` : ''}
              ${defaultLang.prompt ? `
                <div class="phase-prompt">
                  <strong>Prompt:</strong> ${defaultLang.prompt}
                </div>
              ` : ''}
              ${phase.options?.length > 0 ? `
                <div class="phase-options">
                  <strong>Options:</strong>
                  ${phase.options.map(opt => {
                    const optText = opt.text.en || opt.text[Object.keys(opt.text)[0]];
                    const optResponse = opt.response.en || opt.response[Object.keys(opt.response)[0]];
                    return `
                      <div class="option-item ${opt.quality === 'best' ? 'best' : opt.quality === 'misconception' ? 'misconception' : ''}">
                        <span class="option-letter">${opt.letter}</span>
                        <span class="option-text">${optText}</span>
                        ${optResponse ? `<div class="option-response">${optResponse}</div>` : ''}
                      </div>
                    `;
                  }).join('')}
                </div>
              ` : ''}
            </div>
          </div>
        `;
      }).join('');
    },

    /**
     * Render Grow Track section
     */
    renderGrowTrack(audit) {
      const growTrack = audit.assets.growTrack || audit.assets.growTrackSupabase;
      if (!growTrack) return '<div class="empty-state">No Grow track found</div>';

      if (audit.assets.growTrack) {
        // From JSON
        return `
          <div class="grow-track-card">
            <div class="grow-track-header">
              <span class="grow-emoji">${growTrack.emoji || '🧠'}</span>
              <h4>${typeof growTrack.title === 'object' ? growTrack.title.en : growTrack.title}</h4>
            </div>
            <div class="grow-track-content">
              <div><strong>Learning Objective:</strong> ${typeof growTrack.learning_objective === 'object' ? growTrack.learning_objective.en : growTrack.learning_objective}</div>
              <div><strong>Activity:</strong> ${typeof growTrack.activity === 'object' ? growTrack.activity.en : growTrack.activity}</div>
            </div>
          </div>
        `;
      } else {
        // From Supabase
        return `
          <div class="grow-track-card">
            <div class="grow-track-header">
              <span class="grow-emoji">${growTrack.emoji || '🧠'}</span>
              <h4>${growTrack.topic || growTrack.headline}</h4>
            </div>
            <div class="grow-track-content">
              <div><strong>Universal Truth:</strong> ${growTrack.universal_truth || 'N/A'}</div>
              <div><strong>Category:</strong> ${growTrack.category || 'N/A'}</div>
            </div>
          </div>
        `;
      }
    },

    /**
     * Render pipeline status
     */
    renderPipelineStatus(audit) {
      const pipelines = [
        { name: 'Supabase', ...audit.pipelines.supabase },
        { name: 'HeyGen', ...audit.pipelines.heygen },
        { name: 'ElevenLabs', ...audit.pipelines.elevenlabs },
        { name: 'Local Files', ...audit.pipelines.local }
      ];

      return `
        <div class="pipeline-grid">
          ${pipelines.map(pipe => `
            <div class="pipeline-item ${pipe.status}">
              <strong>${pipe.name}:</strong> ${pipe.status}
              ${pipe.count > 0 ? ` (${pipe.count} assets)` : ''}
            </div>
          `).join('')}
        </div>
      `;
    },

    /**
     * Show inspector panel for a day
     */
    async showInspector(dayNumber) {
      // Close any existing panel
      const existing = document.querySelector('.lesson-inspector-panel');
      if (existing) existing.remove();
      
      // Create loading panel
      const loadingPanel = document.createElement('div');
      loadingPanel.className = 'lesson-inspector-panel';
      loadingPanel.innerHTML = `
        <div class="inspector-header">
          <h2>Loading Day ${dayNumber}...</h2>
          <button class="inspector-close" onclick="LessonInspector.close()">×</button>
        </div>
        <div style="padding: 40px; text-align: center;">
          <div style="font-size: 18px; margin-bottom: 16px;">Gathering all assets...</div>
          <div style="font-size: 14px; color: rgba(255,255,255,0.6); margin-bottom: 24px;">Querying Supabase, checking local files, loading JSON...</div>
          <div class="loading-progress">
            <div class="progress-bar">
              <div class="progress-fill" id="audit-progress"></div>
            </div>
            <div style="font-size: 12px; color: rgba(255,255,255,0.5); margin-top: 8px;" id="audit-status">Initializing...</div>
          </div>
        </div>
      `;
      
      document.body.appendChild(loadingPanel);
      this.injectStyles();
      
      // Add escape key handler
      const escapeHandler = (e) => {
        if (e.key === 'Escape') {
          this.close();
          document.removeEventListener('keydown', escapeHandler);
        }
      };
      document.addEventListener('keydown', escapeHandler);

      try {
        // Update progress
        const updateProgress = (percent, status) => {
          const progressBar = document.getElementById('audit-progress');
          const statusText = document.getElementById('audit-status');
          if (progressBar) progressBar.style.width = `${percent}%`;
          if (statusText) statusText.textContent = status;
        };
        
        updateProgress(10, 'Loading JSON lesson...');
        const audit = await this.getFullAudit(dayNumber);
        updateProgress(100, 'Complete!');
        
        this.currentAudit = audit;
        
        // Small delay to show completion
        await new Promise(resolve => setTimeout(resolve, 300));
        
        const panelHtml = this.renderInspectorPanel(audit);
        loadingPanel.outerHTML = panelHtml;
        
        // Re-inject styles and re-attach event handlers
        this.injectStyles();
        this.attachPanelHandlers();
        
      } catch (error) {
        console.error('[LessonInspector] Error:', error);
        loadingPanel.innerHTML = `
          <div class="inspector-header">
            <h2>Error Loading Day ${dayNumber}</h2>
            <button class="inspector-close" onclick="LessonInspector.close()">×</button>
          </div>
          <div style="padding: 40px;">
            <div style="color: #ef4444; margin-bottom: 16px;">Failed to load lesson: ${error.message}</div>
            <div style="font-size: 13px; color: rgba(255,255,255,0.6);">
              <strong>Details:</strong><br>
              ${error.stack || 'No additional details available'}
            </div>
            <button onclick="LessonInspector.showInspector(${dayNumber})" style="margin-top: 16px; padding: 8px 16px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer;">
              Retry
            </button>
          </div>
        `;
      }
    },
    
    /**
     * Attach event handlers to panel
     */
    attachPanelHandlers() {
      // Close button
      const closeBtn = document.querySelector('.inspector-close');
      if (closeBtn) {
        closeBtn.onclick = () => this.close();
      }
      
      // Escape key
      const escapeHandler = (e) => {
        if (e.key === 'Escape') {
          this.close();
          document.removeEventListener('keydown', escapeHandler);
        }
      };
      document.addEventListener('keydown', escapeHandler);
      
      // Open in Learn button
      const openBtn = document.querySelector('.btn-primary');
      if (openBtn && openBtn.textContent.includes('Open in Learn')) {
        const dayNum = this.currentAudit?.dayNumber;
        if (dayNum) {
          openBtn.onclick = () => {
            window.location.href = `/learn.html?day=${dayNum}`;
          };
        }
      }
      
      // Download button
      const downloadBtn = document.querySelector('.btn-secondary');
      if (downloadBtn && downloadBtn.textContent.includes('Download')) {
        const dayNum = this.currentAudit?.dayNumber;
        if (dayNum) {
          downloadBtn.onclick = () => {
            this.downloadAudit(dayNum);
          };
        }
      }
    },

    /**
     * Play audio from a URL
     */
    playAudio(url, button) {
      document.querySelectorAll('audio.media-player').forEach(audio => {
        if (audio !== button.nextElementSibling) {
          audio.pause();
          audio.currentTime = 0;
        }
      });

      let audio = button.nextElementSibling;
      if (!audio || !audio.classList.contains('media-player')) {
        audio = document.createElement('audio');
        audio.className = 'media-player';
        audio.controls = true;
        audio.src = url;
        button.parentElement.appendChild(audio);
      }

      if (audio.paused) {
        audio.play();
        button.textContent = '⏸ Pause';
      } else {
        audio.pause();
        button.textContent = '▶ Play Audio';
      }
    },

    /**
     * Switch language in phase display
     */
    switchLanguage(button, phaseId, lang) {
      // Remove active class from all tabs
      button.parentElement.querySelectorAll('.lang-tab').forEach(tab => tab.classList.remove('active'));
      button.classList.add('active');
      
      // Update phase content (simplified - would need full implementation)
      console.log(`Switching to ${lang} for ${phaseId}`);
    },

    /**
     * Close inspector
     */
    close() {
      document.querySelectorAll('audio.media-player, video.media-player').forEach(media => {
        media.pause();
        media.currentTime = 0;
      });
      
      const panel = document.querySelector('.lesson-inspector-panel');
      if (panel) panel.remove();
      this.currentAudit = null;
    },

    /**
     * Inject CSS styles
     */
    injectStyles() {
      if (document.getElementById('lesson-inspector-styles')) return;

      const style = document.createElement('style');
      style.id = 'lesson-inspector-styles';
      style.textContent = `
        .lesson-inspector-panel {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.95);
          backdrop-filter: blur(20px);
          z-index: 10000;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }
        .inspector-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px 24px;
          border-bottom: 1px solid rgba(255,255,255,0.1);
          background: rgba(0,0,0,0.5);
        }
        .inspector-header h2 {
          margin: 0;
          font-size: 20px;
        }
        .inspector-subtitle {
          font-size: 14px;
          color: rgba(255,255,255,0.6);
          margin-top: 4px;
        }
        .inspector-close {
          background: none;
          border: none;
          color: rgba(255,255,255,0.8);
          font-size: 32px;
          cursor: pointer;
          padding: 0;
          width: 40px;
          height: 40px;
          line-height: 1;
        }
        .inspector-content {
          flex: 1;
          overflow-y: auto;
          padding: 24px;
        }
        .inspector-section {
          margin-bottom: 32px;
        }
        .inspector-section h3 {
          margin: 0 0 16px 0;
          font-size: 16px;
          color: rgba(255,255,255,0.9);
        }
        .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 16px;
          margin-bottom: 24px;
        }
        .stat-card {
          padding: 16px;
          background: rgba(255,255,255,0.05);
          border-radius: 8px;
          text-align: center;
        }
        .stat-value {
          font-size: 24px;
          font-weight: 700;
          color: var(--kelly-blue, #2563eb);
        }
        .stat-label {
          font-size: 12px;
          color: rgba(255,255,255,0.6);
          margin-top: 4px;
        }
        .metadata-grid {
          display: grid;
          gap: 12px;
        }
        .metadata-grid > div {
          display: grid;
          grid-template-columns: 120px 1fr;
          gap: 12px;
          padding: 12px;
          background: rgba(255,255,255,0.05);
          border-radius: 8px;
        }
        .metadata-grid label {
          font-weight: 600;
          color: rgba(255,255,255,0.7);
        }
        .variants-grid {
          display: grid;
          gap: 8px;
        }
        .variants-grid > div {
          padding: 8px;
          background: rgba(255,255,255,0.03);
          border-radius: 6px;
          font-size: 13px;
        }
        .grow-track-card {
          padding: 20px;
          background: rgba(59, 130, 246, 0.1);
          border-radius: 8px;
          border-left: 3px solid #3b82f6;
        }
        .grow-track-header {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 16px;
        }
        .grow-emoji {
          font-size: 24px;
        }
        .grow-track-header h4 {
          margin: 0;
          font-size: 18px;
        }
        .grow-track-content {
          display: grid;
          gap: 12px;
        }
        .grow-track-content > div {
          line-height: 1.6;
        }
        .phase-video-group {
          margin-bottom: 24px;
        }
        .phase-video-group h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
          color: rgba(255,255,255,0.8);
        }
        .media-item {
          margin-bottom: 20px;
          padding: 16px;
          background: rgba(255,255,255,0.05);
          border-radius: 8px;
        }
        .media-info {
          margin-bottom: 12px;
          font-size: 13px;
          color: rgba(255,255,255,0.7);
        }
        .media-player {
          width: 100%;
          border-radius: 8px;
        }
        .images-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 16px;
        }
        .image-item {
          position: relative;
        }
        .image-item img {
          width: 100%;
          border-radius: 8px;
          cursor: pointer;
          transition: transform 0.2s;
        }
        .image-item img.expanded {
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) scale(1.5);
          z-index: 10001;
          max-width: 90vw;
          max-height: 90vh;
        }
        .image-path, .image-info {
          margin-top: 8px;
          font-size: 12px;
          color: rgba(255,255,255,0.6);
        }
        .phase-card {
          margin-bottom: 20px;
          padding: 20px;
          background: rgba(255,255,255,0.05);
          border-radius: 8px;
          border-left: 3px solid var(--kelly-blue, #2563eb);
        }
        .phase-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }
        .phase-name {
          font-weight: 700;
          font-size: 14px;
          letter-spacing: 1px;
        }
        .language-tabs {
          display: flex;
          gap: 8px;
          margin-bottom: 16px;
        }
        .lang-tab {
          padding: 6px 12px;
          background: rgba(255,255,255,0.1);
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 12px;
          color: rgba(255,255,255,0.7);
        }
        .lang-tab.active {
          background: var(--kelly-blue, #2563eb);
          color: white;
        }
        .play-audio-btn {
          padding: 6px 12px;
          background: var(--kelly-blue, #2563eb);
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 12px;
        }
        .phase-script, .phase-prompt {
          margin-bottom: 16px;
          line-height: 1.6;
        }
        .phase-options {
          margin-top: 16px;
        }
        .option-item {
          padding: 12px;
          margin: 8px 0;
          background: rgba(255,255,255,0.03);
          border-radius: 6px;
          border-left: 3px solid rgba(255,255,255,0.2);
        }
        .option-item.best {
          border-left-color: #22c55e;
        }
        .option-item.misconception {
          border-left-color: #ef4444;
        }
        .option-letter {
          display: inline-block;
          width: 24px;
          height: 24px;
          line-height: 24px;
          text-align: center;
          background: rgba(255,255,255,0.1);
          border-radius: 4px;
          margin-right: 12px;
          font-weight: 600;
        }
        .option-response {
          margin-top: 8px;
          padding-top: 8px;
          border-top: 1px solid rgba(255,255,255,0.1);
          font-size: 13px;
          color: rgba(255,255,255,0.7);
        }
        .pipeline-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 12px;
        }
        .pipeline-item {
          padding: 12px;
          border-radius: 6px;
          font-size: 13px;
        }
        .pipeline-item.available {
          background: rgba(34, 197, 94, 0.1);
          color: #22c55e;
        }
        .pipeline-item.error, .pipeline-item.unavailable, .pipeline-item.no_assets {
          background: rgba(239, 68, 68, 0.1);
          color: #ef4444;
        }
        .empty-state {
          padding: 20px;
          text-align: center;
          color: rgba(255,255,255,0.5);
          font-style: italic;
        }
        .inspector-errors ul, .inspector-warnings ul {
          margin: 0;
          padding-left: 20px;
        }
        .inspector-errors li {
          color: #ef4444;
        }
        .inspector-warnings li {
          color: #eab308;
        }
        .inspector-actions {
          display: flex;
          gap: 12px;
          padding: 20px 24px;
          border-top: 1px solid rgba(255,255,255,0.1);
          background: rgba(0,0,0,0.5);
        }
        .btn-primary, .btn-secondary {
          flex: 1;
          padding: 12px;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-weight: 600;
          font-size: 14px;
        }
        .btn-primary {
          background: var(--kelly-blue, #2563eb);
          color: white;
        }
        .btn-secondary {
          background: rgba(255,255,255,0.1);
          color: rgba(255,255,255,0.9);
        }
        .loading-progress {
          max-width: 400px;
          margin: 0 auto;
        }
        .progress-bar {
          width: 100%;
          height: 8px;
          background: rgba(255,255,255,0.1);
          border-radius: 4px;
          overflow: hidden;
        }
        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #2563eb, #3b82f6);
          width: 0%;
          transition: width 0.3s ease;
          border-radius: 4px;
        }
      `;
      document.head.appendChild(style);
    },

    /**
     * Download audit as JSON
     */
    async downloadAudit(dayNumber) {
      const audit = await this.getFullAudit(dayNumber);
      const blob = new Blob([JSON.stringify(audit, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `lesson-${dayNumber}-comprehensive-audit.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  // Make available globally
  window.LessonAudit = LessonInspector;
  window.LessonInspector = LessonInspector;
})();
