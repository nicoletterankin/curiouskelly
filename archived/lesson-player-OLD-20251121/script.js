// The Daily Lesson - Kelly's Universal Classroom JavaScript

// Import image selector (if using modules, otherwise will be loaded via script tag)
// const ImageSelector = require('./components/image-selector.js');

class LessonPlayer {
    constructor() {
        this.currentAge = 25;
        this.currentAgeBucket = '18-35';
        this.currentLanguage = 'en';
        this.currentTopic = null;
        this.currentStep = 'welcome';
        this.currentPhase = 'welcome';
        this.isPlaying = false;
        this.imageElement = null;
        this.audioElement = null;
        this.lessonData = null;
        this.lessonManifest = null;
        this.teachingMoments = [];
        this.currentTime = 0;
        this.imageSelector = null;
        this.calendarBridge = null;
        this.calendarLessons = [];
        this.currentCalendarLesson = null;
        this.visualManifestCache = {};
        this.lessonVisualManifestUrl = null;
        
        // Unity integration
        this.unityIframe = null;
        this.unityReady = false;
        this.unityStatusElement = null;
        this.unityStatusIndicator = null;
        this.unityStatusText = null;
        
        this.init();
    }

    init() {
        this.setupElements();
        this.setupEventListeners();
        this.setupWelcomeOverlay();
        this.setupBrandingElements();
        this.setupCalendarBridge();
        this.setupUnityIntegration();
        this.loadTodayLesson();
        this.updateDateDisplay();
    }
    
    setupWelcomeOverlay() {
        // Check if this is first visit
        const hasVisited = localStorage.getItem('kelly_has_visited');
        const welcomeOverlay = document.getElementById('kelly-welcome-overlay');
        const startBtn = document.getElementById('start-lesson-btn');
        const welcomeAvatar = document.getElementById('kelly-welcome-avatar');
        
        // Set Kelly avatar in welcome
        if (welcomeAvatar) {
            welcomeAvatar.src = '../lessons/images/kelly-directors-chair-curious.png';
        }
        
        // Show welcome for first-time learners
        if (!hasVisited && welcomeOverlay) {
            setTimeout(() => {
                welcomeOverlay.classList.add('active');
            }, 500);
        }
        
        // Handle start button
        if (startBtn) {
            startBtn.addEventListener('click', () => {
                localStorage.setItem('kelly_has_visited', 'true');
                if (welcomeOverlay) {
                    welcomeOverlay.classList.remove('active');
                }
            });
        }
    }
    
    setupBrandingElements() {
        // Set Kelly brand avatar
        const brandAvatar = document.getElementById('kelly-brand-avatar');
        if (brandAvatar) {
            brandAvatar.src = '../lessons/images/kelly-directors-chair-curious.png';
        }
        
        // Initialize progress tracking
        this.updateProgressIndicator(0, 'Getting ready...');
    }
    
    updateProgressIndicator(percentage, text) {
        const progressText = document.getElementById('progress-text');
        const progressFill = document.getElementById('kelly-progress-fill');
        
        if (progressText) {
            progressText.textContent = text;
        }
        
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }
    }

    setupElements() {
        this.imageElement = document.getElementById('kelly-image');
        this.audioElement = document.getElementById('kelly-audio');
        
        // Unity elements
        this.unityIframe = document.getElementById('kelly-unity-iframe');
        this.unityStatusElement = document.getElementById('kelly-unity-status');
        this.unityStatusIndicator = document.getElementById('unity-status-indicator');
        this.unityStatusText = document.getElementById('unity-status-text');
        
        // Initialize image selector
        if (typeof ImageSelector !== 'undefined') {
            this.imageSelector = new ImageSelector();
        } else {
            console.warn('ImageSelector not loaded, using fallback');
            this.imageSelector = {
                selectImage: (state) => '../lessons/images/kelly-directors-chair-curious.png'
            };
        }
        
        // VisionOS UI elements
        this.ageSlider = document.getElementById('age-slider');
        this.ageValue = document.getElementById('age-value');
        this.ageBuckets = document.querySelectorAll('.age-bucket-floating');
        this.choiceCardsContainer = document.getElementById('choice-cards-container');
        this.questionCard = document.getElementById('question-card');
        this.questionText = document.getElementById('question-text');
        this.playButton = document.getElementById('play-pause');
        this.audioScript = document.getElementById('audio-script');
        this.progressFillStrip = document.getElementById('progress-fill-strip');
        this.loadingScreen = document.getElementById('loading-screen');
        this.languageSelector = document.getElementById('language-selector');
        this.hamburgerBtn = document.getElementById('hamburger-btn');
        this.hamburgerMenu = document.getElementById('hamburger-menu');
        this.lessonTitle = document.getElementById('lesson-title');
        this.topicDescription = document.getElementById('topic-description');
        this.learningObjectives = document.getElementById('learning-objectives');
        this.lessonDuration = document.getElementById('lesson-duration');
        this.lessonTags = document.getElementById('lesson-tags');
        this.calendarBridgeContainer = document.querySelector('[data-component="calendar-bridge"]');
        this.lessonTitle = document.getElementById('lesson-title');
        this.topicDescription = document.getElementById('topic-description');
        this.learningObjectives = document.getElementById('learning-objectives');
        
        // Initialize parallax if available
        if (typeof parallaxController !== 'undefined' && parallaxController) {
            // Add parallax to panels
            const panels = document.querySelectorAll('.glass-panel');
            panels.forEach((panel, index) => {
                parallaxController.addElement(panel, 0.05 + (index * 0.02), 8);
            });
        }
    }

    setupEventListeners() {
        // Age slider
        if (this.ageSlider) {
            this.ageSlider.addEventListener('input', (e) => {
                this.currentAge = parseInt(e.target.value);
                this.updateAgeDisplay();
                this.updateAgeBucket();
                this.loadAgeAppropriateContent();
            });
        }

        // Age buckets
        if (this.ageBuckets) {
            this.ageBuckets.forEach(bucket => {
                bucket.addEventListener('click', (e) => {
                    const ageRange = e.target.dataset.age;
                    this.setAgeFromBucket(ageRange);
                });
            });
        }

        // Language selector
        if (this.languageSelector) {
            this.languageSelector.addEventListener('change', (e) => {
                this.currentLanguage = e.target.value;
                this.loadAgeAppropriateContent();
            });
        }

        // Hamburger menu
        if (this.hamburgerBtn) {
            this.hamburgerBtn.addEventListener('click', () => {
                this.hamburgerMenu.classList.toggle('active');
            });
        }

        // Close hamburger when clicking outside
        document.addEventListener('click', (e) => {
            if (this.hamburgerMenu && 
                !this.hamburgerMenu.contains(e.target) && 
                !this.hamburgerBtn.contains(e.target)) {
                this.hamburgerMenu.classList.remove('active');
            }
        });
        
        // Audio event listeners
        if (this.audioElement) {
            this.audioElement.addEventListener('timeupdate', () => {
                this.updateProgress();
                this.checkTeachingMoments();
            });

            this.audioElement.addEventListener('ended', () => {
                console.log('🎵 Audio ended');
                this.isPlaying = false;
                this.updatePlayButton();
                this.onAudioEnded();
            });
            
            this.audioElement.addEventListener('loadedmetadata', () => {
                this.updateProgress();
            });
        }

        // Play/pause button
        if (this.playButton) {
            this.playButton.addEventListener('click', (e) => {
                e.stopPropagation();
                this.togglePlayPause();
            });
        }
    }

    setupCalendarBridge() {
        if (!this.calendarBridgeContainer || typeof CalendarBridge === 'undefined') {
            return;
        }

        this.calendarBridge = new CalendarBridge(this.calendarBridgeContainer, {
            onLessonSelected: (lesson) => {
                if (!lesson?.has_dna || !lesson?.dna_file) {
                    this.showError('Selected lesson is missing Kelly\'s DNA. Please choose another day.');
                    return;
                }
                this.setCalendarLessonMeta(lesson);
                this.loadLessonById(lesson.dna_file, lesson);
            }
        });
    }

    setupUnityIntegration() {
        if (!this.unityIframe) {
            console.warn('Unity iframe not found, using fallback image');
            return;
        }

        // Listen for messages from Unity
        window.addEventListener('message', (event) => {
            this.handleUnityMessage(event);
        });

        // Wait for iframe to load
        this.unityIframe.addEventListener('load', () => {
            this.updateUnityStatus('loading', 'Connecting to Kelly...');
            // Unity will send 'kelly-ready' when it's ready
        });

        // Initial status
        this.updateUnityStatus('loading', 'Loading Kelly...');
    }

    handleUnityMessage(event) {
        const data = event.data;
        if (!data || data.source !== 'kelly-webgl') {
            return;
        }

        switch (data.type) {
            case 'kelly-ready':
                this.unityReady = true;
                this.updateUnityStatus('ready', 'Kelly is ready');
                // Hide status after a moment
                setTimeout(() => {
                    if (this.unityStatusElement) {
                        this.unityStatusElement.classList.add('hidden');
                    }
                }, 2000);
                // Load pending lesson if any
                if (this.pendingUnityLesson) {
                    const { lessonId, jsonUrl, audioUrl, expressionsUrl } = this.pendingUnityLesson;
                    this.loadLessonToUnity(lessonId, jsonUrl, audioUrl, expressionsUrl);
                    this.pendingUnityLesson = null;
                } else if (this.lessonData && this.currentCalendarLesson) {
                    // Try to load current lesson if already loaded
                    this.loadCurrentLessonToUnity();
                }
                break;
            case 'kelly-loading':
                this.updateUnityStatus('loading', 'Loading lesson...');
                break;
            case 'kelly-playing':
                this.updateUnityStatus('playing', 'Playing');
                // Hide status while playing
                if (this.unityStatusElement) {
                    this.unityStatusElement.classList.add('hidden');
                }
                break;
            case 'kelly-stopped':
                this.updateUnityStatus('ready', 'Ready');
                break;
            case 'kelly-error':
                this.updateUnityStatus('error', data.message || 'Error occurred');
                console.error('Unity error:', data.message);
                break;
        }
    }

    updateUnityStatus(status, text) {
        if (this.unityStatusIndicator) {
            this.unityStatusIndicator.className = `status-indicator ${status}`;
        }
        if (this.unityStatusText) {
            this.unityStatusText.textContent = text;
        }
        if (this.unityStatusElement) {
            this.unityStatusElement.classList.remove('hidden');
        }
    }

    postToUnity(type, payload = {}) {
        if (!this.unityIframe?.contentWindow || !this.unityReady) {
            console.warn('Unity not ready, cannot send message:', type);
            return;
        }

        this.unityIframe.contentWindow.postMessage(
            {
                source: 'curiouskelly.com',
                destination: 'kelly-webgl',
                type,
                payload
            },
            '*'
        );
    }

    loadLessonToUnity(lessonId, jsonUrl, audioUrl, expressionsUrl = null) {
        if (!this.unityReady) {
            console.warn('Unity not ready, lesson will load when Unity is ready');
            // Store lesson data to load when ready
            this.pendingUnityLesson = { lessonId, jsonUrl, audioUrl, expressionsUrl };
            return;
        }

        this.postToUnity('kelly-load', {
            lessonId,
            jsonUrl,
            audioUrl,
            expressionsUrl,
            visualManifestUrl: this.lessonVisualManifestUrl
                ? this.toAbsoluteUrl(this.lessonVisualManifestUrl)
                : undefined,
            offsetMs: 50
        });
    }

    stopUnityLesson() {
        if (this.unityReady) {
            this.postToUnity('kelly-stop', {});
        }
    }

    updateDateDisplay() {
        // Date display removed in VisionOS design
        // Can be added to identity panel if needed
    }

    updateAgeDisplay() {
        this.ageValue.textContent = this.currentAge;
    }

    updateAgeBucket() {
        let bucket = '';
        if (this.currentAge >= 2 && this.currentAge <= 5) bucket = '2-5';
        else if (this.currentAge >= 6 && this.currentAge <= 12) bucket = '6-12';
        else if (this.currentAge >= 13 && this.currentAge <= 17) bucket = '13-17';
        else if (this.currentAge >= 18 && this.currentAge <= 35) bucket = '18-35';
        else if (this.currentAge >= 36 && this.currentAge <= 60) bucket = '36-60';
        else if (this.currentAge >= 61 && this.currentAge <= 102) bucket = '61-102';
        
        this.currentAgeBucket = bucket;
        
        // Update visual selection
        if (this.ageBuckets) {
            this.ageBuckets.forEach(bucketEl => {
                bucketEl.classList.remove('active');
                if (bucketEl.dataset.age === bucket) {
                    bucketEl.classList.add('active');
                }
            });
        }
    }

    setAgeFromBucket(ageRange) {
        const [min, max] = ageRange.split('-').map(Number);
        const midAge = Math.floor((min + max) / 2);
        this.currentAge = midAge;
        this.ageSlider.value = midAge;
        this.updateAgeDisplay();
        this.updateAgeBucket();
        this.loadAgeAppropriateContent();
    }

    async loadTodayLesson() {
        this.showLoading();
        
        try {
            // Try to load today's lesson from calendar
            const today = new Date();
            const calendarResponse = await fetch('../lessons/365_day_calendar.json');
            if (calendarResponse.ok) {
                const calendar = await calendarResponse.json();
                this.calendarLessons = calendar.lessons || [];
                if (this.calendarBridge && this.calendarLessons.length) {
                    this.calendarBridge.setLessons(this.calendarLessons);
                }
                const todayLesson = calendar.lessons.find(l => {
                    const lessonDate = new Date(l.date + ', ' + today.getFullYear());
                    return lessonDate.getDate() === today.getDate() &&
                           lessonDate.getMonth() === today.getMonth();
                });
                
                if (todayLesson && todayLesson.has_dna && todayLesson.dna_file) {
                    this.setCalendarLessonMeta(todayLesson);
                    await this.loadLessonById(todayLesson.dna_file, todayLesson);
                    return;
                }

                const firstAvailableLesson = calendar.lessons.find(l => l.has_dna && l.dna_file);
                if (firstAvailableLesson) {
                    this.setCalendarLessonMeta(firstAvailableLesson);
                    await this.loadLessonById(firstAvailableLesson.dna_file, firstAvailableLesson);
                    return;
                }
            }
            
            // Fallback: load balance lesson
            const response = await fetch('balance-lesson.json');
            if (response.ok) {
                this.lessonData = await response.json();
            } else {
                this.lessonData = await this.getSampleLesson();
            }
            
            this.setCalendarLessonMeta({
                day: 0,
                title: this.lessonData.title,
                duration: this.lessonData.metadata?.duration || { min: 5, max: 10 },
                tags: this.lessonData.metadata?.tags || ['balance', 'science'],
                has_dna: true
            });
            this.displayLesson();
        } catch (error) {
            console.error('Error loading lesson:', error);
            this.showError('Failed to load today\'s lesson. Please try again.');
        } finally {
            this.hideLoading();
        }
    }
    
    async loadLessonById(lessonId, lessonSummary = null) {
        try {
            this.updateProgressIndicator(20, 'Loading lesson...');
            
            if (lessonSummary) {
                this.setCalendarLessonMeta(lessonSummary);
            } else if (this.calendarLessons?.length) {
                const matchedLesson = this.calendarLessons.find(lesson => lesson.dna_file === lessonId);
                if (matchedLesson) {
                    this.setCalendarLessonMeta(matchedLesson);
                }
            }

            // Try to load manifest first
            this.updateProgressIndicator(40, 'Loading resources...');
            const manifestPath = `../lessons/manifests/${lessonId}-manifest.json`;
            const manifestResponse = await fetch(manifestPath);

            if (manifestResponse.ok) {
                this.lessonManifest = await manifestResponse.json();
                console.log('✓ Loaded manifest for:', lessonId);
            }

            // Load visual assets manifest (PhaseDNA v2)
            await this.loadVisualManifest(lessonId);
            
            // Load DNA file
            this.updateProgressIndicator(60, 'Preparing your lesson...');
            const dnaPath = `../lessons/${lessonId}-dna.json`;
            const dnaResponse = await fetch(dnaPath);
            
            if (dnaResponse.ok) {
                this.lessonData = await dnaResponse.json();
                this.updateProgressIndicator(80, 'Almost ready...');
                this.displayLesson();
                this.updateProgressIndicator(100, 'Ready to learn!');
            } else {
                throw new Error(`DNA file not found: ${dnaPath}`);
            }
        } catch (error) {
            console.error('Error loading lesson:', error);
            this.updateProgressIndicator(0, 'Error loading lesson');
            throw error;
        }
    }

    async getSampleLesson() {
        // Sample lesson data - in production this comes from the API
        return {
            id: 'sample-001',
            title: 'Why Do Leaves Change Color?',
            description: 'Discover the amazing science behind autumn\'s beautiful colors and learn how plants prepare for winter.',
            ageVariants: {
                '2-5': {
                    title: 'Pretty Leaves!',
                    description: 'Let\'s learn about the beautiful colors of leaves!',
                    video: 'kelly_leaves_2-5.mp4',
                    script: 'Hi little friends! Do you see the pretty leaves outside? They change colors in fall! Let\'s learn why!',
                    objectives: ['See different leaf colors', 'Learn about fall', 'Have fun with Kelly']
                },
                '6-12': {
                    title: 'The Science of Fall Colors',
                    description: 'Explore how and why leaves change color in autumn.',
                    video: 'kelly_leaves_6-12.mp4',
                    script: 'Hello young scientists! Today we\'re going to discover the amazing science behind why leaves change color in the fall. It\'s all about chemistry and nature!',
                    objectives: ['Understand chlorophyll and photosynthesis', 'Learn about carotenoids and anthocyanins', 'Connect science to seasonal changes']
                },
                '13-17': {
                    title: 'Photosynthesis and Seasonal Changes',
                    description: 'Dive deep into the biochemical processes that create autumn\'s spectacular display.',
                    video: 'kelly_leaves_13-17.mp4',
                    script: 'Welcome to today\'s lesson on photosynthesis and seasonal changes. We\'ll explore the complex biochemical processes that create the stunning autumn colors we see each year.',
                    objectives: ['Analyze chlorophyll breakdown processes', 'Understand pigment chemistry', 'Connect biology to environmental factors']
                },
                '18-35': {
                    title: 'The Biochemistry of Autumn',
                    description: 'Explore the molecular mechanisms behind leaf color changes and their ecological significance.',
                    video: 'kelly_leaves_18-35.mp4',
                    script: 'Today we\'ll examine the fascinating biochemistry of autumn leaf color changes. This process involves complex molecular interactions that reveal much about plant biology and environmental adaptation.',
                    objectives: ['Master chlorophyll degradation pathways', 'Understand pigment synthesis regulation', 'Analyze ecological and evolutionary implications']
                },
                '36-60': {
                    title: 'Seasonal Biology and Environmental Science',
                    description: 'Investigate the broader implications of seasonal changes in plant biology and climate science.',
                    video: 'kelly_leaves_36-60.mp4',
                    script: 'Let\'s explore how seasonal changes in plants reflect broader environmental patterns and what they tell us about climate, ecology, and the interconnectedness of natural systems.',
                    objectives: ['Connect plant biology to climate science', 'Understand ecosystem dynamics', 'Apply knowledge to environmental stewardship']
                },
                '61-102': {
                    title: 'The Wisdom of Seasonal Cycles',
                    description: 'Reflect on the deeper meanings of seasonal change and the lessons nature teaches us about life cycles.',
                    video: 'kelly_leaves_61-102.mp4',
                    script: 'Today we\'ll contemplate the profound wisdom embedded in seasonal cycles. These natural processes offer insights into life, change, and our place in the greater scheme of things.',
                    objectives: ['Appreciate natural wisdom and cycles', 'Connect science to life philosophy', 'Share knowledge across generations']
                }
            },
            interactions: [
                {
                    step: 'welcome',
                    question: 'What do you think causes leaves to change color?',
                    choices: [
                        { text: 'Magic!', nextStep: 'teaching', response: 'That\'s a wonderful way to think about it! Let\'s discover the science behind the magic.' },
                        { text: 'The weather gets cold', nextStep: 'teaching', response: 'You\'re on the right track! Temperature plays a role, but there\'s more to the story.' },
                        { text: 'I don\'t know', nextStep: 'teaching', response: 'That\'s perfectly fine! Learning is all about discovering new things together.' }
                    ]
                },
                {
                    step: 'teaching',
                    question: 'Which part of the leaf do you think is most important for its color?',
                    choices: [
                        { text: 'The outside skin', nextStep: 'practice', response: 'Good thinking! The surface is important, but the real magic happens inside.' },
                        { text: 'The inside parts', nextStep: 'practice', response: 'Exactly! The internal structures contain the color-making chemicals.' },
                        { text: 'The stem', nextStep: 'practice', response: 'The stem is important for transport, but the colors are made in the leaf itself.' }
                    ]
                }
            ]
        };
    }

    displayLesson() {
        if (!this.lessonData || !this.lessonData.ageVariants) {
            this.updateLessonMeta();
            return;
        }

        const variant = this.lessonData.ageVariants[this.currentAgeBucket];

        if (!variant) {
            this.teachingMoments = [];
            this.showInteraction();
            return;
        }
        this.updateLessonMeta(variant);

        // Load age-appropriate video
        this.loadAgeAppropriateContent();
    }

    loadAgeAppropriateContent() {
        if (!this.lessonData || !this.lessonData.ageVariants) {
            this.updateLessonMeta();
            return;
        }
        
        const variant = this.lessonData.ageVariants[this.currentAgeBucket];
        this.updateLessonMeta(variant);

        if (!variant) {
            this.teachingMoments = [];
            this.showInteraction();
            return;
        }
        
        // Update Kelly image based on current phase
        this.updateKellyImage();
        
        // Load audio for this age variant and phase
        this.loadAudioForPhase(this.currentPhase);
        
        // Load lesson to Unity if available
        this.loadCurrentLessonToUnity();
        
        // Store teaching moments for this variant
        this.teachingMoments = variant.teachingMoments || [];
        
        // Show appropriate interaction for current step
        this.showInteraction();
    }

    loadCurrentLessonToUnity() {
        if (!this.lessonData || !this.currentCalendarLesson) {
            return;
        }

        const lessonId = this.currentCalendarLesson.dna_file || this.lessonData.id;
        const phase = this.currentPhase || 'welcome';
        
        // Construct URLs based on lesson structure
        // Audio URL: ../lessons/audio/{lessonId}/{ageBucket}-{language}-{phase}.mp3
        const audioUrl = `../lessons/audio/${lessonId}/${this.currentAgeBucket}-${this.currentLanguage}-${phase}.mp3`;
        
        // JSON URL (viseme data): ../lessons/audio/{lessonId}/{ageBucket}-{language}-{phase}.a2f.json
        const jsonUrl = `../lessons/audio/${lessonId}/${this.currentAgeBucket}-${this.currentLanguage}-${phase}.a2f.json`;
        
        // Expressions URL (optional): ../lessons/audio/{lessonId}/{ageBucket}-{language}-{phase}.expressions.json
        const expressionsUrl = `../lessons/audio/${lessonId}/${this.currentAgeBucket}-${this.currentLanguage}-${phase}.expressions.json`;
        
        // Load to Unity
        this.loadLessonToUnity(lessonId, jsonUrl, audioUrl, expressionsUrl);
    }

    async loadVisualManifest(lessonId) {
        if (this.visualManifestCache[lessonId]) {
            this.lessonVisualManifestUrl = this.visualManifestCache[lessonId].manifestPath;
            return this.visualManifestCache[lessonId].assets;
        }

        const manifestPath = `../lessons/manifests/${lessonId}-visual-assets.json`;
        try {
            const response = await fetch(manifestPath);
            if (!response.ok) {
                console.warn(`Visual manifest not found for ${lessonId}`);
                return null;
            }
            const payload = await response.json();
            const assetMap = {};
            (payload.assets || []).forEach(asset => {
                assetMap[asset.id] = asset;
            });
            this.visualManifestCache[lessonId] = {
                assets: assetMap,
                manifestPath
            };
            this.lessonVisualManifestUrl = manifestPath;
            console.log(`✓ Loaded visual manifest for ${lessonId} (${Object.keys(assetMap).length} assets)`);
            return assetMap;
        } catch (error) {
            console.error('Failed to load visual manifest:', error);
            return null;
        }
    }

    toAbsoluteUrl(relativePath) {
        if (!relativePath) return null;
        const link = document.createElement('a');
        link.href = relativePath;
        return link.href;
    }

    setCalendarLessonMeta(meta) {
        if (!meta) return;
        this.currentCalendarLesson = meta;
        if (this.calendarBridge) {
            this.calendarBridge.setActiveLesson(meta);
        }
        const variant = this.lessonData?.ageVariants?.[this.currentAgeBucket];
        this.updateLessonMeta(variant, meta);
    }

    updateLessonMeta(variant = {}, calendarMeta = this.currentCalendarLesson) {
        if (this.lessonTitle) {
            this.lessonTitle.textContent = variant.title || 'Kelly is loading today\'s topic...';
        }

        if (this.topicDescription) {
            this.topicDescription.textContent = variant.description || 'Daily description will appear here once the lesson manifest loads.';
        }

        if (this.learningObjectives) {
            this.learningObjectives.innerHTML = '';
            const objectives = Array.isArray(variant.objectives) ? variant.objectives : [];

            if (!objectives.length) {
                const placeholder = document.createElement('li');
                placeholder.className = 'objective-placeholder';
                placeholder.textContent = 'Kelly will list today\'s objectives once the lesson is ready.';
                this.learningObjectives.appendChild(placeholder);
                return;
            }

            objectives.forEach(objective => {
                const li = document.createElement('li');
                li.className = 'objective-item';
                li.textContent = objective;
                this.learningObjectives.appendChild(li);
            });
        }

        if (this.lessonDuration) {
            const durationText = this.getDurationText(calendarMeta?.duration);
            this.lessonDuration.textContent = durationText;
        }

        if (this.lessonTags) {
            this.renderTagPills(calendarMeta?.tags);
        }
    }

    getDurationText(duration) {
        if (!duration) return '-- min';
        const min = duration.min;
        const max = duration.max;
        if (min && max) return `${min}-${max} min`;
        if (min || max) return `${min || max} min`;
        return '-- min';
    }

    renderTagPills(tags) {
        if (!this.lessonTags) return;
        this.lessonTags.innerHTML = '';
        if (!Array.isArray(tags) || !tags.length) {
            const placeholder = document.createElement('span');
            placeholder.className = 'tag-pill tag-pill-placeholder';
            placeholder.textContent = 'Tags pending';
            this.lessonTags.appendChild(placeholder);
            return;
        }

        tags.slice(0, 6).forEach(tag => {
            const pill = document.createElement('span');
            pill.className = 'tag-pill';
            pill.textContent = tag;
            this.lessonTags.appendChild(pill);
        });
    }
    
    updateKellyImage() {
        if (!this.imageElement || !this.imageSelector) return;
        
        // Try to get image from manifest first
        let imagePath = null;
        if (this.lessonManifest && this.lessonManifest.images) {
            const expression = this.imageSelector.getExpressionName({
                phase: this.currentPhase || this.currentStep,
                interactionType: this.getCurrentInteractionType(),
                sentiment: this.getCurrentSentiment()
            });
            
            const manifestImagePath = this.lessonManifest.images[expression];
            if (manifestImagePath) {
                imagePath = `../lessons/${manifestImagePath}`;
            }
        }
        
        // Fallback to image selector
        if (!imagePath) {
            const state = {
                phase: this.currentPhase || this.currentStep,
                interactionType: this.getCurrentInteractionType(),
                sentiment: this.getCurrentSentiment()
            };
            imagePath = this.imageSelector.selectImage(state);
        }
        
        // Update image with smooth transition
        if (this.imageElement.src && this.imageElement.src !== '' && this.imageElement.src !== imagePath) {
            // Fade out
            this.imageElement.style.opacity = '0';
            
            setTimeout(() => {
                this.imageElement.src = imagePath;
                // Fade in
                setTimeout(() => {
                    this.imageElement.style.opacity = '1';
                }, 50);
            }, 150);
        } else {
            // First load or same image
            this.imageElement.src = imagePath;
            this.imageElement.style.opacity = '1';
        }
        
        // Handle image load errors
        this.imageElement.onerror = () => {
            console.warn(`Failed to load image: ${imagePath}`);
            // Fallback to default
            this.imageElement.src = '../lessons/images/kelly-directors-chair-curious.png';
        };
    }
    
    getCurrentInteractionType() {
        // Determine interaction type based on current step/phase
        if (this.currentPhase === 'welcome') return 'question';
        if (this.currentPhase === 'wisdom' || this.currentPhase === 'wisdomMoment') return 'wisdom';
        if (this.currentStep === 'teaching' || this.currentPhase === 'mainContent') return 'explanation';
        return null;
    }
    
    getCurrentSentiment() {
        // Determine sentiment based on learner response or phase
        // This can be enhanced based on actual learner responses
        if (this.currentPhase === 'wisdom') return 'reflective';
        return 'neutral';
    }
    
    loadAudioForPhase(phase) {
        if (!this.lessonData || !this.audioElement) return;
        
        // Special handling for Balance lesson prototype
        if (this.lessonData.id === 'balance-lesson') {
             const audioPath = `videos/audio/kelly_balance_${this.currentAgeBucket}.mp3`;
             this.audioElement.src = audioPath;
             this.audioElement.load();
             console.log(`✓ Loading local balance audio: ${audioPath}`);
             return;
        }

        // Try to load from manifest first
        if (this.lessonManifest && this.lessonManifest.audio) {
            const audioPath = this.lessonManifest.audio[this.currentAgeBucket]?.[this.currentLanguage]?.[phase];
            if (audioPath) {
                this.audioElement.src = `../lessons/${audioPath}`;
                this.audioElement.load();
                console.log(`✓ Loading audio: ${audioPath}`);
                return;
            }
        }
        
        // Fallback: construct path from lesson structure
        const lessonId = this.lessonData.id || 'lesson';
        const audioPath = `../lessons/audio/${lessonId}/${this.currentAgeBucket}-${this.currentLanguage}-${phase}.mp3`;
        this.audioElement.src = audioPath;
        this.audioElement.load();
        console.log(`✓ Loading audio (fallback): ${audioPath}`);
    }
    
    onAudioEnded() {
        // Move to next phase or complete lesson
        const phases = ['welcome', 'mainContent', 'wisdomMoment'];
        const currentIndex = phases.indexOf(this.currentPhase);
        
        if (currentIndex < phases.length - 1) {
            // Move to next phase
            this.currentPhase = phases[currentIndex + 1];
            this.currentStep = this.currentPhase;
            this.updateProgressSteps();
            this.loadAgeAppropriateContent();
        } else {
            // Lesson complete
            this.currentPhase = 'complete';
            console.log('✅ Lesson completed!');
        }
    }
    
    checkTeachingMoments() {
        if (!this.audioElement || !this.teachingMoments.length) return;
        
        const currentTime = Math.floor(this.audioElement.currentTime);
        
        this.teachingMoments.forEach(moment => {
            // Check if moment time is within 1 second of current time
            if (Math.abs(moment.timestamp - currentTime) <= 1) {
                // Show teaching moment if not already shown
                this.showTeachingMoment(moment);
            }
        });
    }
    
    showTeachingMoment(moment) {
        // Create or update teaching moment indicator
        let indicator = document.getElementById('teaching-moment-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'teaching-moment-indicator';
            indicator.className = 'teaching-moment';
            document.body.appendChild(indicator);
        }
        
        // Update content
        indicator.innerHTML = `
            <div class="teaching-moment-content">
                <div class="teaching-moment-icon">✨</div>
                <div class="teaching-moment-type">${this.getMomentTypeLabel(moment.type)}</div>
                <div class="teaching-moment-text">${moment.content}</div>
            </div>
        `;
        
        // Show indicator
        indicator.style.display = 'flex';
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            indicator.style.display = 'none';
        }, 5000);
        
        console.log(`📚 Teaching moment: ${moment.type} at ${moment.timestamp}s`);
    }
    
    getMomentTypeLabel(type) {
        const labels = {
            'explanation': 'Explanation',
            'question': 'Question Time',
            'demonstration': 'Demonstration',
            'story': 'Story',
            'wisdom': 'Wisdom'
        };
        return labels[type] || 'Teaching Moment';
    }

    // Removed showVideoPlaceholder - using images instead

    showInteraction() {
        if (!this.lessonData) return;

        // Determine phase type and render appropriate template
        if (this.currentStep === 'welcome' || this.currentPhase === 'welcome') {
            this.renderWelcomePhase();
        } else if (this.currentStep === 'wisdom' || this.currentPhase === 'wisdom' || this.currentPhase === 'wisdomMoment') {
            this.renderWisdomPhase();
        } else {
            this.renderQuestionPhase();
        }
    }

    renderWelcomePhase() {
        const interaction = this.lessonData.interactions?.find(i => i.step === 'welcome');
        const variant = this.lessonData.ageVariants?.[this.currentAgeBucket];
        const languageContent = variant?.language?.[this.currentLanguage] || variant?.language?.en;
        const ageAdaptation = interaction?.ageAdaptations?.[this.currentAgeBucket];
        
        // Get welcome text - prefer language content, then age adaptation, then variant script
        let welcomeText = languageContent?.welcome || 
                         ageAdaptation?.question ||
                         variant?.script || 
                         interaction?.question ||
                         'Welcome! Let\'s learn together today.';
        
        // Format if it's a key
        welcomeText = this.formatText(welcomeText);
        
        // Update question card
        if (this.questionCard) {
            this.questionCard.className = 'question-card glass-panel-medium welcome-template';
            this.questionCard.innerHTML = `
                <div class="welcome-message">${welcomeText}</div>
            `;
        }
        
        // Hide choice cards for welcome
        if (this.choiceCardsContainer) {
            this.choiceCardsContainer.innerHTML = '';
            this.choiceCardsContainer.style.display = 'none';
        }
        
        // Update audio strip
        const scriptText = variant?.script || 
                          languageContent?.welcome || 
                          welcomeText;
        this.updateAudioStrip(scriptText);
    }

    renderQuestionPhase() {
        // Find interaction matching current step
        // DNA files use step: "welcome", "teaching", "practice", "wisdom"
        const interaction = this.lessonData.interactions?.find(i => {
            if (i.step === this.currentStep) return true;
            // Map legacy phase names
            if (this.currentStep === 'teaching' && i.step === 'teaching') return true;
            if (this.currentStep === 'practice' && i.step === 'practice') return true;
            // Handle sequential questions (q1, q2, q3)
            const questionSteps = ['teaching', 'practice'];
            const stepIndex = questionSteps.indexOf(this.currentStep);
            if (stepIndex >= 0) {
                // Find interactions that are not welcome/wisdom
                const questionInteractions = this.lessonData.interactions?.filter(
                    i => i.step !== 'welcome' && i.step !== 'wisdom'
                ) || [];
                return questionInteractions[stepIndex] === i;
            }
            return false;
        });
        
        if (!interaction) {
            console.warn('No interaction found for step:', this.currentStep);
            // Try to find any non-welcome/wisdom interaction as fallback
            const fallbackInteraction = this.lessonData.interactions?.find(
                i => i.step !== 'welcome' && i.step !== 'wisdom'
            );
            if (fallbackInteraction) {
                return this.renderQuestionPhaseWithInteraction(fallbackInteraction);
            }
            return;
        }

        this.renderQuestionPhaseWithInteraction(interaction);
    }

    renderQuestionPhaseWithInteraction(interaction) {
        const variant = this.lessonData.ageVariants?.[this.currentAgeBucket];
        const ageAdaptation = interaction.ageAdaptations?.[this.currentAgeBucket];
        
        // Get question text - prefer age-adapted, fallback to base question
        // Note: question might be a key (underscore_separated) or actual text
        let question = ageAdaptation?.question || interaction.question || 'No question available';
        
        // Format question if it's a key (replace underscores with spaces, capitalize)
        if (question.includes('_') && !question.includes(' ')) {
            question = question.split('_').map(word => 
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
        }
        
        // Get choices - prefer age-adapted, fallback to base choices
        const choices = ageAdaptation?.choices || interaction.choices || [];
        
        // Format choice text if needed
        const formattedChoices = choices.map(choice => ({
            ...choice,
            text: this.formatText(choice.text || choice.label || 'Option'),
            response: choice.response || 'Thank you for your answer!'
        }));

        // Update question card
        if (this.questionCard) {
            this.questionCard.className = 'question-card glass-panel-medium hover-lift';
            this.questionCard.innerHTML = `
                <button id="play-pause" class="play-button glass-panel-light">
                    <span class="play-icon">${this.isPlaying ? '⏸' : '▶'}</span>
                </button>
                <div class="question-text">${question}</div>
            `;
            
            // Re-attach play button listener
            const playBtn = this.questionCard.querySelector('#play-pause');
            if (playBtn) {
                playBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.togglePlayPause();
                });
            }
            this.playButton = playBtn;
        }

        // Render choice cards (max 2, side by side)
        if (this.choiceCardsContainer) {
            this.choiceCardsContainer.style.display = 'flex';
            this.choiceCardsContainer.innerHTML = '';
            
            const options = formattedChoices.slice(0, 2);
            options.forEach((choice, index) => {
                const card = document.createElement('div');
                card.className = 'choice-card glass-panel-medium hover-lift';
                card.textContent = choice.text;
                card.dataset.choiceIndex = index;
                card.addEventListener('click', () => {
                    this.handleChoice(choice);
                });
                this.choiceCardsContainer.appendChild(card);
            });
        }
        
        // Update audio strip with script
        const scriptText = variant?.script || 
                          variant?.language?.[this.currentLanguage]?.mainContent || 
                          question;
        this.updateAudioStrip(scriptText);
    }

    formatText(text) {
        // Format underscore-separated keys into readable text
        if (typeof text !== 'string') return String(text);
        if (text.includes('_') && !text.includes(' ')) {
            return text.split('_').map(word => 
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
        }
        return text;
    }

    renderWisdomPhase() {
        const interaction = this.lessonData.interactions?.find(i => i.step === 'wisdom');
        const variant = this.lessonData.ageVariants?.[this.currentAgeBucket];
        const languageContent = variant?.language?.[this.currentLanguage] || variant?.language?.en;
        const ageAdaptation = interaction?.ageAdaptations?.[this.currentAgeBucket];
        
        // Get wisdom text - prefer language content, then age adaptation
        let wisdomText = languageContent?.wisdomMoment || 
                        ageAdaptation?.question ||
                        variant?.wisdomMoment ||
                        interaction?.question ||
                        'Thank you for learning with me today!';
        
        // Format if it's a key
        wisdomText = this.formatText(wisdomText);
        
        // Update question card
        if (this.questionCard) {
            this.questionCard.className = 'question-card glass-panel-medium wisdom-template';
            this.questionCard.innerHTML = `
                <div class="wisdom-message">${wisdomText}</div>
                <div class="completion-indicator">✨</div>
                <button class="next-lesson-btn">Continue Learning</button>
            `;
            
            // Add click handler for next lesson button
            const nextBtn = this.questionCard.querySelector('.next-lesson-btn');
            if (nextBtn) {
                nextBtn.addEventListener('click', () => {
                    // Load next lesson or return to calendar
                    console.log('Next lesson clicked');
                    // TODO: Implement next lesson navigation
                });
            }
        }
        
        // Hide choice cards for wisdom
        if (this.choiceCardsContainer) {
            this.choiceCardsContainer.innerHTML = '';
            this.choiceCardsContainer.style.display = 'none';
        }
        
        // Update audio strip
        const scriptText = variant?.script || 
                          languageContent?.wisdomMoment || 
                          wisdomText;
        this.updateAudioStrip(scriptText);
    }

    updateAudioStrip(scriptText) {
        if (this.audioScript && scriptText) {
            this.audioScript.textContent = scriptText;
        }
    }

    handleChoice(choice) {
        // Mark selected choice
        const choiceCards = document.querySelectorAll('.choice-card');
        choiceCards.forEach(card => {
            card.classList.remove('selected');
            const cardText = card.textContent.trim();
            const choiceText = (choice.text || choice.label || '').trim();
            if (cardText === choiceText || card.dataset.choiceIndex === String(choiceCards.length - 1)) {
                card.classList.add('selected');
            }
        });
        
        // Format and show Kelly's response (update audio strip)
        let response = choice.response || 'Thank you for your answer!';
        response = this.formatText(response);
        this.updateAudioStrip(response);
        
        // Update image to celebrating/positive sentiment
        if (this.imageSelector) {
            const state = {
                phase: this.currentPhase,
                interactionType: 'response',
                sentiment: 'positive'
            };
            const imagePath = this.imageSelector.selectImage(state);
            this.updateImageWithTransition(imagePath);
        }
        
        // Move to next step based on choice.nextStep
        // DNA files use: "teaching", "practice", "wisdom"
        const nextStep = choice.nextStep || 'teaching';
        this.currentStep = nextStep;
        
        // Map nextStep to currentPhase
        if (nextStep === 'wisdom') {
            this.currentPhase = 'wisdom';
        } else if (nextStep === 'practice') {
            this.currentPhase = 'practice';
        } else {
            this.currentPhase = 'mainContent';
        }
        
        // Show next interaction after delay
        setTimeout(() => {
            this.showInteraction();
            this.updateKellyImage();
            this.loadAudioForPhase(this.currentPhase);
        }, 2000);
    }
    
    updateImageWithTransition(imagePath) {
        if (!this.imageElement) return;
        
        this.imageElement.style.opacity = '0';
        setTimeout(() => {
            this.imageElement.src = imagePath;
            setTimeout(() => {
                this.imageElement.style.opacity = '1';
            }, 50);
        }, 150);
    }

    showKellyResponse(response) {
        // Update audio strip with Kelly's response
        if (response) {
            this.updateAudioStrip(response);
        }
        console.log('Kelly says:', response);
    }

    updateProgressSteps() {
        // Progress steps removed in VisionOS design
        // Phase progression handled by templates
    }

    togglePlayPause() {
        if (!this.audioElement) return;
        
        if (this.isPlaying) {
            this.audioElement.pause();
            this.isPlaying = false;
        } else {
            this.audioElement.play().catch(e => {
                console.log('Audio play failed:', e);
            });
            this.isPlaying = true;
        }
        this.updatePlayButton();
    }

    updatePlayButton() {
        if (this.playButton) {
            const icon = this.playButton.querySelector('.play-icon');
            if (icon) {
                icon.textContent = this.isPlaying ? '⏸' : '▶';
            }
        }
    }

    updateProgress() {
        if (!this.audioElement) return;
        
        const progress = ((this.audioElement.currentTime / this.audioElement.duration) * 100) || 0;
        
        // Update audio strip progress
        if (this.progressFillStrip) {
            this.progressFillStrip.style.width = progress + '%';
        }
        
        // Update time display
        const currentTimeEl = document.getElementById('current-time');
        const totalTimeEl = document.getElementById('total-time');
        if (currentTimeEl) {
            currentTimeEl.textContent = this.formatTime(this.audioElement.currentTime);
        }
        if (totalTimeEl) {
            totalTimeEl.textContent = this.formatTime(this.audioElement.duration);
        }
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    // Removed onVideoEnded - using onAudioEnded instead

    joinLiveClass() {
        // In production, this would connect to live class functionality
        alert('Joining live class! (Feature coming soon)');
        
        // Simulate live learner count
        this.simulateLiveLearners();
    }

    simulateLiveLearners() {
        // Simulate live learner count
        const baseCount = Math.floor(Math.random() * 50) + 10;
        this.liveLearners.textContent = baseCount;
        
        // Update count periodically
        setInterval(() => {
            const change = Math.floor(Math.random() * 6) - 3; // -3 to +3
            const newCount = Math.max(0, parseInt(this.liveLearners.textContent) + change);
            this.liveLearners.textContent = newCount;
        }, 5000);
    }

    showLoading() {
        this.loadingScreen.style.display = 'flex';
    }

    hideLoading() {
        this.loadingScreen.style.display = 'none';
    }

    showError(message) {
        alert(`Error: ${message}`);
    }
}

// Initialize the lesson player when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new LessonPlayer();
});

// Service Worker registration for offline functionality (disabled for now)
// if ('serviceWorker' in navigator) {
//     window.addEventListener('load', () => {
//         navigator.serviceWorker.register('/sw.js')
//             .then(registration => {
//                 console.log('SW registered: ', registration);
//             })
//             .catch(registrationError => {
//                 console.log('SW registration failed: ', registrationError);
//             });
//     });
// }

