/**
 * Kelly Unified Lesson Service - Single Source of Truth
 * 
 * Enterprise-grade unified data layer for Calendar → Panel → Player flow
 * 
 * Ensures data coherency across all components by using KellyLessonLoader
 * as the canonical data source with intelligent caching and normalization.
 * 
 * Usage:
 *   // Get metadata (for calendar)
 *   const metadata = await KellyUnifiedLessonService.getMetadata(161);
 *   
 *   // Get preview (for panel)
 *   const preview = await KellyUnifiedLessonService.getPreview(161);
 *   
 *   // Get full lesson (for player)
 *   const lesson = await KellyUnifiedLessonService.getFullLesson(161, { track: 'learn' });
 */

(function() {
  'use strict';

  const KellyUnifiedLessonService = {
    // Cache for metadata and previews (lightweight)
    metadataCache: new Map(),
    previewCache: new Map(),
    
    // Cache expiration (5 minutes)
    cacheExpiry: 5 * 60 * 1000,
    
    /**
     * Get lightweight metadata for calendar display
     * Returns: { topic, emoji, category, headline, hasLearn, hasGrow, date }
     */
    async getMetadata(dayNumber) {
      const cacheKey = `meta-${dayNumber}`;
      const cached = this.metadataCache.get(cacheKey);
      if (cached && Date.now() - cached.timestamp < this.cacheExpiry) {
        return cached.data;
      }
      
      try {
        // Try LOCAL_PACKS first (fastest)
        const pack = this._getLocalPack(dayNumber);
        if (pack?.lesson) {
          const metadata = {
            topic: pack.lesson.topic || pack.lesson.title || '',
            emoji: pack.lesson.emoji || '📚',
            category: pack.lesson.category || '',
            headline: pack.lesson.headline || pack.lesson.marketing_headline || '',
            hasLearn: !!(pack.lesson || pack.atoms?.length),
            hasGrow: !!pack.grow,
            date: this._getDateForDay(dayNumber)
          };
          
          this.metadataCache.set(cacheKey, { data: metadata, timestamp: Date.now() });
          return metadata;
        }
        
        // Fallback: Try JSON file
        const jsonData = await this._loadJsonFile(dayNumber);
        if (jsonData?.meta) {
          const metadata = {
            topic: typeof jsonData.meta.topic === 'object' ? jsonData.meta.topic.en : jsonData.meta.topic || '',
            emoji: jsonData.meta.emoji || '📚',
            category: jsonData.meta.category || '',
            headline: typeof jsonData.headline === 'object' ? jsonData.headline.en : jsonData.headline || '',
            hasLearn: !!(jsonData.lesson || jsonData.phases),
            hasGrow: !!(jsonData.grow || jsonData.growTrack),
            date: this._getDateForDay(dayNumber)
          };
          
          this.metadataCache.set(cacheKey, { data: metadata, timestamp: Date.now() });
          return metadata;
        }
        
        // Last resort: Use KellyLessonLoader (but only for metadata)
        if (window.KellyLessonLoader) {
          try {
            const result = await window.KellyLessonLoader.loadLesson(dayNumber, {
              archetype: 'The Scientist',
              age: 25,
              region: 'adult',
              track: 'learn'
            });
            
            if (result?.lesson) {
              const metadata = {
                topic: result.lesson.topic || result.lesson.title || '',
                emoji: result.lesson.emoji || '📚',
                category: result.lesson.category || '',
                headline: result.lesson.marketing_headline || '',
                hasLearn: !!(result.lesson || result.atoms?.length),
                hasGrow: false, // Grow track requires separate load
                date: this._getDateForDay(dayNumber)
              };
              
              this.metadataCache.set(cacheKey, { data: metadata, timestamp: Date.now() });
              return metadata;
            }
          } catch (e) {
            console.warn('[UnifiedService] KellyLessonLoader failed for metadata:', e);
          }
        }
        
        // Emergency fallback
        return {
          topic: `Day ${dayNumber}`,
          emoji: '📚',
          category: '',
          headline: '',
          hasLearn: false,
          hasGrow: false,
          date: this._getDateForDay(dayNumber)
        };
      } catch (e) {
        console.error('[UnifiedService] getMetadata failed:', e);
        return {
          topic: `Day ${dayNumber}`,
          emoji: '📚',
          category: '',
          headline: '',
          hasLearn: false,
          hasGrow: false,
          date: this._getDateForDay(dayNumber)
        };
      }
    },
    
    /**
     * Get full preview for panel display
     * Returns: { learn: {...}, grow: {...}, completeness, visuals, phases }
     */
    async getPreview(dayNumber, options = {}) {
      const { track = 'learn', archetype = 'The Scientist', age = 25, region = 'adult' } = options;
      const cacheKey = `preview-${dayNumber}-${track}`;
      const cached = this.previewCache.get(cacheKey);
      if (cached && Date.now() - cached.timestamp < this.cacheExpiry) {
        return cached.data;
      }
      
      try {
        // Use KellyLessonLoader for Learn track (bulletproof)
        let learnData = null;
        let growData = null;
        
        if (track === 'learn' || track === 'both') {
          if (window.KellyLessonLoader) {
            try {
              const result = await window.KellyLessonLoader.loadLesson(dayNumber, {
                archetype,
                age,
                region,
                track: 'learn'
              });
              
              if (result?.lesson || result?.atoms?.length) {
                learnData = {
                  topic: result.lesson?.topic || result.lesson?.title || '',
                  emoji: result.lesson?.emoji || '📚',
                  category: result.lesson?.category || '',
                  headline: result.lesson?.marketing_headline || '',
                  universalTruth: result.lesson?.universal_truth || '',
                  atoms: result.atoms || [],
                  phases: this._extractPhases(result.atoms || []),
                  visuals: this._extractVisuals(result.atoms || []),
                  videos: this._extractVideos(result.atoms || []),
                  completeness: this._calculateCompleteness(result)
                };
              }
            } catch (e) {
              console.warn('[UnifiedService] KellyLessonLoader failed for Learn track:', e);
            }
          }
          
          // Fallback to LOCAL_PACKS if KellyLessonLoader didn't return data
          if (!learnData) {
            const pack = this._getLocalPack(dayNumber);
            if (pack?.lesson || pack?.atoms?.length) {
              learnData = {
                topic: pack.lesson?.topic || pack.lesson?.title || '',
                emoji: pack.lesson?.emoji || '📚',
                category: pack.lesson?.category || '',
                headline: pack.lesson?.headline || '',
                universalTruth: pack.lesson?.universal_truth || '',
                atoms: pack.atoms || [],
                phases: this._extractPhases(pack.atoms || []),
                visuals: this._extractVisuals(pack.atoms || []),
                videos: this._extractVideos(pack.atoms || []),
                completeness: this._calculateCompleteness({ lesson: pack.lesson, atoms: pack.atoms })
              };
            }
          }
        }
        
        // Load Grow track
        if (track === 'grow' || track === 'both') {
          growData = await this._loadGrowTrack(dayNumber);
        }
        
        // Calculate overall completeness
        const completeness = window.LessonPreviewPopup?.calculateCompleteness?.(dayNumber) || {
          completeness: learnData ? (learnData.completeness || 0) : 0,
          status: 'missing',
          checks: {},
          stats: {}
        };
        
        const preview = {
          dayNumber,
          date: this._getDateForDay(dayNumber),
          learn: learnData || { topic: '', emoji: '📚', atoms: [], phases: [], visuals: [], videos: [] },
          grow: growData || null,
          completeness: completeness.completeness || 0,
          status: completeness.status || 'missing',
          checks: completeness.checks || {},
          stats: completeness.stats || {}
        };
        
        this.previewCache.set(cacheKey, { data: preview, timestamp: Date.now() });
        return preview;
      } catch (e) {
        console.error('[UnifiedService] getPreview failed:', e);
        return {
          dayNumber,
          date: this._getDateForDay(dayNumber),
          learn: { topic: '', emoji: '📚', atoms: [], phases: [], visuals: [], videos: [] },
          grow: null,
          completeness: 0,
          status: 'missing',
          checks: {},
          stats: {}
        };
      }
    },
    
    /**
     * Get full lesson for player
     * Delegates to KellyLessonLoader (the canonical source)
     */
    async getFullLesson(dayNumber, options = {}) {
      if (window.KellyLessonLoader) {
        return await window.KellyLessonLoader.loadLesson(dayNumber, options);
      }
      
      // Fallback if KellyLessonLoader not available
      const pack = this._getLocalPack(dayNumber);
      if (pack) {
        return {
          lesson: pack.lesson || null,
          atoms: pack.atoms || [],
          shards: [],
          source: 'local_pack'
        };
      }
      
      throw new Error(`No lesson data available for day ${dayNumber}`);
    },
    
    /**
     * Clear cache for a specific day
     */
    clearCache(dayNumber) {
      this.metadataCache.delete(`meta-${dayNumber}`);
      this.previewCache.delete(`preview-${dayNumber}-learn`);
      this.previewCache.delete(`preview-${dayNumber}-grow`);
      this.previewCache.delete(`preview-${dayNumber}-both`);
    },
    
    /**
     * Clear all caches
     */
    clearAllCaches() {
      this.metadataCache.clear();
      this.previewCache.clear();
    },
    
    // ============================================
    // PRIVATE HELPERS
    // ============================================
    
    /**
     * Get local pack for a day
     */
    _getLocalPack(dayNumber) {
      const localPacks = window.CURIOUS_KELLY?.LOCAL_PACKS || {};
      return localPacks[dayNumber] ||
             localPacks[`day-${String(dayNumber).padStart(3, '0')}`] ||
             localPacks[String(dayNumber)] ||
             null;
    },
    
    /**
     * Load JSON file for a day
     */
    async _loadJsonFile(dayNumber) {
      try {
        const response = await fetch(`/lessons/day-${dayNumber}.json`);
        if (response.ok) {
          return await response.json();
        }
      } catch (e) {
        // Silent fail
      }
      return null;
    },
    
    /**
     * Load Grow track data
     */
    async _loadGrowTrack(dayNumber) {
      // Try LOCAL_PACKS first
      const pack = this._getLocalPack(dayNumber);
      if (pack?.grow) {
        return {
          topic: pack.grow.topic || '',
          objective: pack.grow.objective || pack.grow.learning_objective || '',
          activity: pack.grow.activity || '',
          emoji: pack.grow.emoji || '🤖'
        };
      }
      
      // Try JSON file
      const jsonData = await this._loadJsonFile(dayNumber);
      if (jsonData?.grow || jsonData?.growTrack) {
        const grow = jsonData.grow || jsonData.growTrack;
        return {
          topic: typeof grow.topic === 'object' ? grow.topic.en : grow.topic || '',
          objective: typeof grow.learning_objective === 'object' ? grow.learning_objective.en : grow.learning_objective || '',
          activity: typeof grow.activity === 'object' ? grow.activity.en : grow.activity || '',
          emoji: grow.emoji || '🤖'
        };
      }
      
      return null;
    },
    
    /**
     * Extract phases from atoms
     */
    _extractPhases(atoms) {
      const phases = {};
      const phaseOrder = ['hook', 'question', 'context', 'choice', 'reflection', 'wisdom', 'action'];
      
      atoms.forEach(atom => {
        const phase = atom.phase?.toLowerCase();
        if (phase && phaseOrder.includes(phase)) {
          phases[phase] = {
            script: atom.content?.script || atom.script || atom.content?.text || '',
            visual: atom.visual_url || atom.content?.visual_url || '',
            video: atom.hd_video_url || atom.video_url || atom.content?.video_url || '',
            audio: atom.audio_url || atom.content?.audio_url || '',
            hasContent: !!(atom.content?.script || atom.script || atom.content?.text)
          };
        }
      });
      
      return phases;
    },
    
    /**
     * Extract visuals from atoms
     */
    _extractVisuals(atoms) {
      const visuals = [];
      atoms.forEach(atom => {
        if (atom.visual_url || atom.content?.visual_url) {
          visuals.push({
            url: atom.visual_url || atom.content.visual_url,
            phase: atom.phase?.toLowerCase() || '',
            description: atom.content?.visual_description || ''
          });
        }
      });
      return visuals;
    },
    
    /**
     * Extract videos from atoms
     */
    _extractVideos(atoms) {
      const videos = [];
      atoms.forEach(atom => {
        const videoUrl = atom.hd_video_url || atom.video_url || atom.content?.video_url;
        if (videoUrl) {
          videos.push({
            url: videoUrl,
            phase: atom.phase?.toLowerCase() || '',
            template: atom.template || 'default'
          });
        }
      });
      return videos;
    },
    
    /**
     * Calculate completeness score
     */
    _calculateCompleteness(result) {
      if (!result.lesson && (!result.atoms || result.atoms.length === 0)) {
        return 0;
      }
      
      let score = 0;
      const phaseCount = new Set(result.atoms?.map(a => a.phase?.toLowerCase()).filter(Boolean)).size;
      const hasVideo = result.atoms?.some(a => a.hd_video_url || a.video_url);
      const hasVisual = result.atoms?.some(a => a.visual_url);
      const hasContent = result.atoms?.some(a => a.content?.script || a.script);
      
      if (result.lesson) score += 20;
      if (phaseCount >= 7) score += 30;
      else if (phaseCount >= 4) score += 20;
      else if (phaseCount >= 1) score += 10;
      if (hasVideo) score += 20;
      if (hasVisual) score += 15;
      if (hasContent) score += 15;
      
      return Math.min(100, score);
    },
    
    /**
     * Get date for a day number
     */
    _getDateForDay(dayNumber) {
      if (window.KellyTime?.dayNumberToDate) {
        return window.KellyTime.dayNumberToDate(dayNumber);
      }
      // Fallback calculation
      const year = new Date().getFullYear();
      const isLeapYear = (year % 4 === 0 && year % 100 !== 0) || (year % 400 === 0);
      const dayOfYear = isLeapYear && dayNumber >= 60 ? dayNumber + 1 : dayNumber;
      return new Date(year, 0, dayOfYear);
    }
  };
  
  // Export to window
  window.KellyUnifiedLessonService = KellyUnifiedLessonService;
  
  console.log('[UnifiedService] ✅ Initialized - Single source of truth for lesson data');
})();

