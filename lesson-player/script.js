// The Daily Lesson - Kelly's Universal Classroom JavaScript

class LessonPlayer {
    constructor() {
        this.currentAge = 25;
        this.currentAgeBucket = '18-35';
        this.currentTopic = null;
        this.currentStep = 'welcome';
        this.isPlaying = false;
        this.videoElement = null;
        this.audioElement = null;
        this.lessonData = null;
        this.teachingMoments = [];
        this.currentTime = 0;
        
        this.init();
    }

    init() {
        this.setupElements();
        this.setupEventListeners();
        this.loadTodayLesson();
        this.updateDateDisplay();
    }

    setupElements() {
        this.videoElement = document.getElementById('kelly-video');
        
        // Create audio element if it doesn't exist
        const videoContainer = this.videoElement.parentElement;
        if (!document.getElementById('kelly-audio')) {
            this.audioElement = document.createElement('audio');
            this.audioElement.id = 'kelly-audio';
            this.audioElement.preload = 'auto';
            videoContainer.appendChild(this.audioElement);
        } else {
            this.audioElement = document.getElementById('kelly-audio');
        }
        
        this.ageSlider = document.getElementById('age-slider');
        this.ageValue = document.getElementById('age-value');
        this.ageBuckets = document.querySelectorAll('.age-bucket');
        this.choiceContainer = document.getElementById('choice-container');
        this.loadingScreen = document.getElementById('loading-screen');
        
        // Lesson elements
        this.lessonTitle = document.getElementById('lesson-title');
        this.topicDescription = document.getElementById('topic-description');
        this.learningObjectives = document.getElementById('learning-objectives');
        this.liveLearners = document.getElementById('live-learners');
    }

    setupEventListeners() {
        // Age slider
        this.ageSlider.addEventListener('input', (e) => {
            this.currentAge = parseInt(e.target.value);
            this.updateAgeDisplay();
            this.updateAgeBucket();
            this.loadAgeAppropriateContent();
        });

        // Age buckets
        this.ageBuckets.forEach(bucket => {
            bucket.addEventListener('click', (e) => {
                const ageRange = e.target.dataset.age;
                this.setAgeFromBucket(ageRange);
            });
        });

        // Video controls
        this.videoElement.addEventListener('click', () => {
            this.togglePlayPause();
        });

        this.videoElement.addEventListener('timeupdate', () => {
            this.updateProgress();
        });

        this.videoElement.addEventListener('ended', () => {
            this.onVideoEnded();
        });
        
        // Add audio event listeners if audio element exists
        if (this.audioElement) {
            this.audioElement.addEventListener('timeupdate', () => {
                this.updateProgress();
            });

            this.audioElement.addEventListener('ended', () => {
                console.log('🎵 Audio ended');
                this.isPlaying = false;
                document.getElementById('play-pause').textContent = '▶️';
            });
        }

        // Play/pause button
        document.getElementById('play-pause').addEventListener('click', (e) => {
            e.stopPropagation();
            this.togglePlayPause();
        });

        // Join live class
        document.getElementById('join-live').addEventListener('click', () => {
            this.joinLiveClass();
        });
    }

    updateDateDisplay() {
        const today = new Date();
        const options = { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        };
        document.getElementById('current-date').textContent = today.toLocaleDateString('en-US', options);
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
        this.ageBuckets.forEach(bucketEl => {
            bucketEl.classList.remove('active');
            if (bucketEl.dataset.age === bucket) {
                bucketEl.classList.add('active');
            }
        });
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
            // For now, load a sample lesson
            // In production, this would fetch from the API
            this.lessonData = await this.getSampleLesson();
            this.displayLesson();
        } catch (error) {
            console.error('Error loading lesson:', error);
            this.showError('Failed to load today\'s lesson. Please try again.');
        } finally {
            this.hideLoading();
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
        const variant = this.lessonData.ageVariants[this.currentAgeBucket];
        
        this.lessonTitle.textContent = variant.title;
        this.topicDescription.textContent = variant.description;
        
        // Update learning objectives
        this.learningObjectives.innerHTML = '';
        variant.objectives.forEach(objective => {
            const li = document.createElement('li');
            li.textContent = objective;
            this.learningObjectives.appendChild(li);
        });

        // Load age-appropriate video
        this.loadAgeAppropriateContent();
    }

    loadAgeAppropriateContent() {
        if (!this.lessonData) return;
        
        const variant = this.lessonData.ageVariants[this.currentAgeBucket];
        
        // Check if video file exists, otherwise show placeholder
        const videoPath = `videos/${variant.video}`;
        
        // Try to load video, but handle gracefully if it doesn't exist
        this.videoElement.src = '';
        this.videoElement.src = videoPath;
        
        // Add error handler for missing videos
        this.videoElement.onerror = () => {
            this.showVideoPlaceholder(variant);
        };
        
        // Load audio for this age variant
        const audioPath = `videos/audio/kelly_leaves_${this.currentAgeBucket}.mp3`;
        if (this.audioElement) {
            this.audioElement.src = audioPath;
            this.audioElement.load();
            
            // Set up audio event listeners
            this.audioElement.addEventListener('timeupdate', () => {
                this.checkTeachingMoments();
            });
            
            this.audioElement.addEventListener('ended', () => {
                console.log('📺 Audio finished playing');
            });
        }
        
        // Store teaching moments for this variant
        this.teachingMoments = variant.teachingMoments || [];
        
        // Update lesson content
        this.lessonTitle.textContent = variant.title;
        this.topicDescription.textContent = variant.description;
        
        // Update learning objectives
        this.learningObjectives.innerHTML = '';
        variant.objectives.forEach(objective => {
            const li = document.createElement('li');
            li.textContent = objective;
            this.learningObjectives.appendChild(li);
        });

        // Show appropriate interaction for current step
        this.showInteraction();
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

    showVideoPlaceholder(variant) {
        // Hide video element
        this.videoElement.style.display = 'none';
        
        // Create placeholder div
        const placeholder = document.createElement('div');
        placeholder.className = 'video-placeholder';
        placeholder.innerHTML = `
            <div class="placeholder-content">
                <div class="placeholder-icon">🎓</div>
                <h3>Kelly's Video Coming Soon</h3>
                <p>${variant.title}</p>
                <p class="placeholder-script">"${variant.script}"</p>
            </div>
        `;
        
        // Insert after video container
        const container = this.videoElement.parentElement;
        if (!container.querySelector('.video-placeholder')) {
            container.appendChild(placeholder);
        }
        
        console.log(`📝 Kelly's script for ${this.currentAgeBucket}: "${variant.script}"`);
    }

    showInteraction() {
        const interaction = this.lessonData.interactions.find(i => i.step === this.currentStep);
        if (!interaction) return;

        this.choiceContainer.innerHTML = '';
        
        // Add question
        const questionEl = document.createElement('h3');
        questionEl.textContent = interaction.question;
        questionEl.className = 'interaction-question';
        this.choiceContainer.appendChild(questionEl);

        // Add choice buttons
        interaction.choices.forEach((choice, index) => {
            const button = document.createElement('button');
            button.textContent = choice.text;
            button.className = 'choice-btn';
            button.addEventListener('click', () => {
                this.handleChoice(choice);
            });
            this.choiceContainer.appendChild(button);
        });
    }

    handleChoice(choice) {
        // Show Kelly's response
        this.showKellyResponse(choice.response);
        
        // Move to next step
        this.currentStep = choice.nextStep;
        this.updateProgressSteps();
        
        // Show next interaction or continue lesson
        setTimeout(() => {
            this.showInteraction();
        }, 2000);
    }

    showKellyResponse(response) {
        // In production, this would trigger Kelly speaking the response
        console.log('Kelly says:', response);
        
        // For now, show a simple alert
        // In production, this would be integrated with the video/TTS system
        alert(`Kelly: ${response}`);
    }

    updateProgressSteps() {
        const steps = document.querySelectorAll('.step');
        steps.forEach(step => {
            step.classList.remove('active');
            if (step.dataset.step === this.currentStep) {
                step.classList.add('active');
            }
        });
    }

    togglePlayPause() {
        // Try audio first, fall back to video
        const element = this.audioElement || this.videoElement;
        
        if (this.isPlaying) {
            if (element) element.pause();
            document.getElementById('play-pause').textContent = '▶️';
            this.isPlaying = false;
        } else {
            if (element) {
                element.play().catch(e => {
                    console.log('Audio play failed:', e);
                });
            }
            document.getElementById('play-pause').textContent = '⏸️';
            this.isPlaying = true;
        }
    }

    updateProgress() {
        // Use audio element if available, otherwise video element
        const element = this.audioElement || this.videoElement;
        if (!element) return;
        
        const progress = ((element.currentTime / element.duration) * 100) || 0;
        document.querySelector('.progress-fill').style.width = progress + '%';
        
        // Update time display
        document.getElementById('current-time').textContent = this.formatTime(element.currentTime);
        document.getElementById('total-time').textContent = this.formatTime(element.duration);
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    onVideoEnded() {
        this.isPlaying = false;
        document.getElementById('play-pause').textContent = '▶️';
        
        // Move to next step or show completion
        if (this.currentStep === 'welcome') {
            this.currentStep = 'teaching';
            this.updateProgressSteps();
            this.showInteraction();
        }
    }

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

