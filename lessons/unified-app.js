// Unified App Logic
// Handles the "Claude-like" Product Experience
// Integrates CalendarApp Logic for Data & Lesson Playback

class UnifiedApp {
    constructor() {
        this.state = {
            isSidebarOpen: window.innerWidth > 768,
            userStreak: 0,
            lastVisit: null,
            completedLessons: [],
            settings: {
                age: '6-12',
                language: 'en'
            },
            currentLesson: null,
            calendarData: null,
            
            // Lesson Playback State
            currentPhase: 'welcome', // welcome, q1, q2, q3, wisdom
            lessonDNA: null,
            
            // Kelly State
            kellyImages: [
                '../lesson-player/0.png',  // Zoom 0: Most zoomed in (Conversation)
                '../lesson-player/1.png',  // Zoom 1
                '../lesson-player/2.png',  // Zoom 2
                '../lesson-player/3.png',  // Zoom 3: Mid-shot (Idle)
                '../lesson-player/4.jpeg', // Zoom 4
                '../lesson-player/5.png',  // Zoom 5
                '../lesson-player/6.png'   // Zoom 6: Most zoomed out
            ],
            currentZoom: 3
        };

        this.init();
    }

    async init() {
        this.cacheDOM();
        this.bindEvents();
        this.loadProgress();
        this.renderState();
        
        // Load Real Data
        await this.loadCalendarData();
        
        // Initial animation entry
        setTimeout(() => {
            const active = document.querySelector('.interaction-state.active');
            if(active) active.classList.add('animate-in');
        }, 500);
    }

    cacheDOM() {
        this.dom = {
            sidebar: document.getElementById('app-sidebar'),
            sidebarToggle: document.getElementById('sidebar-toggle'),
            mobileMenuBtn: document.getElementById('mobile-menu-btn'),
            kellyViewport: document.getElementById('kelly-viewport'),
            kellyActor: document.getElementById('kelly-actor'),
            interactionContainer: document.getElementById('interaction-container'),
            
            // Overlays
            calendarOverlay: document.getElementById('calendar-overlay'),
            
            // State Containers
            stateGreeting: document.getElementById('state-greeting'),
            stateLesson: document.getElementById('state-lesson'),
            
            // Dynamic Elements
            greetingTitle: document.querySelector('.greeting-title'),
            greetingSubtitle: document.querySelector('.greeting-subtitle'),
            streakCount: document.querySelector('.streak-badge'),
            sidebarNavList: document.querySelector('.nav-section:nth-child(3)'), // "Recent Lessons" container
            
            // Buttons
            btnStartLesson: document.getElementById('start-lesson-btn'),
            btnOpenCalendar: document.getElementById('open-calendar-btn'),
            btnCloseOverlay: document.querySelector('.close-overlay-btn')
        };
    }

    bindEvents() {
        // Sidebar Toggles
        this.dom.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        this.dom.mobileMenuBtn.addEventListener('click', () => this.toggleSidebar(true));
        
        // Navigation Actions
        this.dom.btnStartLesson.addEventListener('click', () => this.startLesson());
        this.dom.btnOpenCalendar.addEventListener('click', () => this.openCalendar());
        this.dom.btnCloseOverlay.addEventListener('click', () => this.closeOverlay());

        // Resize Listener
        window.addEventListener('resize', () => {
            if (window.innerWidth <= 768 && this.state.isSidebarOpen) {
                this.toggleSidebar(false); 
            }
        });
    }

    // =================================================================
    // DATA & PERSISTENCE
    // =================================================================

    loadProgress() {
        try {
            const saved = localStorage.getItem('kelly_progress');
            if (saved) {
                const parsed = JSON.parse(saved);
                // Merge saved state, preserving defaults
                this.state = { ...this.state, ...parsed };
                this.validateStreak();
            }
            this.updateStreakUI();
        } catch (e) {
            console.error('Failed to load progress:', e);
        }
    }

    saveProgress() {
        try {
            // Only save persistent fields
            const toSave = {
                userStreak: this.state.userStreak,
                lastVisit: this.state.lastVisit,
                completedLessons: this.state.completedLessons,
                settings: this.state.settings
            };
            localStorage.setItem('kelly_progress', JSON.stringify(toSave));
        } catch (e) {
            console.error('Failed to save progress:', e);
        }
    }

    validateStreak() {
        if (!this.state.lastVisit) return;
        const last = new Date(this.state.lastVisit);
        const now = new Date();
        last.setHours(0,0,0,0);
        const today = new Date(now);
        today.setHours(0,0,0,0);
        
        const diffTime = Math.abs(today - last);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        if (diffDays > 1) {
            this.state.userStreak = 0;
        }
    }

    updateStreakUI() {
        if (this.dom.streakCount) {
            this.dom.streakCount.textContent = `🔥 ${this.state.userStreak} Day Streak`;
        }
    }

    async loadCalendarData() {
        try {
            const response = await fetch('365_day_calendar.json');
            this.state.calendarData = await response.json();
            this.loadTodayLesson();
        } catch (error) {
            console.error('Failed to load calendar data:', error);
        }
    }

    loadTodayLesson() {
        const today = new Date();
        // Find lesson for today (mocking date logic for prototype if needed)
        // In real app: use today.getDate() / getMonth()
        // For prototype: defaulting to Day 45 if no match or simply picking one
        
        const lesson = this.state.calendarData.lessons.find(l => {
             const d = new Date(l.date + ', ' + today.getFullYear());
             return d.getDate() === today.getDate() && d.getMonth() === today.getMonth();
        }) || this.state.calendarData.lessons.find(l => l.day === 45); // Fallback to Day 45 (The Sun)

        if (lesson) {
            this.state.currentLesson = lesson;
            this.updateGreetingUI(lesson);
        }
    }

    updateGreetingUI(lesson) {
        // Update "Good morning" text
        this.dom.greetingTitle.textContent = `Good morning!`;
        this.dom.greetingSubtitle.innerHTML = `I'm ready to teach you about <strong>${lesson.title}</strong> today.`;
        
        // Check if completed
        const isCompleted = this.state.completedLessons.includes(lesson.day);
        if(isCompleted) {
            this.dom.btnStartLesson.innerHTML = `<span class="btn-icon">↺</span> Review Lesson`;
        } else {
            this.dom.btnStartLesson.innerHTML = `<span class="btn-icon">▶</span> Start Daily Lesson`;
        }
    }

    // =================================================================
    // UI & INTERACTION FLOW
    // =================================================================

    toggleSidebar(forceState = null) {
        if (forceState !== null) {
            this.state.isSidebarOpen = forceState;
        } else {
            this.state.isSidebarOpen = !this.state.isSidebarOpen;
        }

        if (this.state.isSidebarOpen) {
            this.dom.sidebar.classList.remove('collapsed');
            this.dom.sidebar.classList.add('open');
            document.body.classList.add('sidebar-open');
        } else {
            this.dom.sidebar.classList.add('collapsed');
            this.dom.sidebar.classList.remove('open');
            document.body.classList.remove('sidebar-open');
        }
    }

    startLesson() {
        // 1. Fade out greeting
        this.dom.stateGreeting.style.opacity = '0';
        this.dom.stateGreeting.style.transform = 'translateY(-20px)';
        
        // 2. Trigger Kelly Animation (Zoom in)
        this.setKellyZoom(0); // Close-up
        
        // 3. Load Lesson Logic
        setTimeout(() => {
            this.dom.stateGreeting.classList.remove('active');
            this.dom.stateLesson.classList.add('active');
            
            // Start the Lesson Flow
            this.runLessonSequence();
        }, 400);
    }

    async runLessonSequence() {
        this.state.currentPhase = 'welcome';
        
        // Try load DNA
        if(this.state.currentLesson.has_dna) {
            try {
                const dnaRes = await fetch(`${this.state.currentLesson.dna_file}-dna.json`);
                this.state.lessonDNA = await dnaRes.json();
            } catch(e) {
                console.warn("DNA not found, using fallback");
            }
        }
        
        this.renderLessonPhase();
    }

    renderLessonPhase() {
        const dna = this.state.lessonDNA;
        const phase = this.state.currentPhase;
        const age = this.state.settings.age; // e.g. "6-12" - In real app mapping might be needed if keys differ
        const lang = this.state.settings.language;

        let contentHTML = '';
        let showChoices = false;

        // --- WELCOME PHASE ---
        if (phase === 'welcome') {
            const welcomeText = dna?.ageVariants?.[age]?.script || 
                              dna?.interactions?.find(i=>i.step==='welcome')?.question ||
                              "Welcome! Let's explore this topic together.";
            
            contentHTML = `
                <h2 class="lesson-phase-title">Welcome</h2>
                <p class="lesson-text">${welcomeText}</p>
                <button class="btn-primary-large" onclick="window.app.advancePhase('teaching')">
                    Let's Begin
                </button>
            `;
        }
        
        // --- TEACHING / QUESTION PHASE ---
        else if (phase === 'teaching' || phase === 'q1' || phase === 'q2' || phase === 'q3') {
            // Find interaction
            let interaction = null;
            if(dna?.interactions) {
                // Simplistic finding logic for prototype
                // In production use robust step mapping
                interaction = dna.interactions.find(i => i.step !== 'welcome' && i.step !== 'wisdom');
            }

            const question = interaction?.ageAdaptations?.[age]?.question || 
                           interaction?.question || 
                           "What do you think about this?";
            
            const choices = interaction?.ageAdaptations?.[age]?.choices || 
                          interaction?.choices || [];

            contentHTML = `
                <h3 class="lesson-question">${question}</h3>
                <div class="options-grid">
                    ${choices.map((c, idx) => `
                        <button class="option-card" onclick="window.app.handleChoice(${idx})">
                            ${c.text || c.label}
                        </button>
                    `).join('')}
                </div>
            `;
        }
        
        // --- WISDOM PHASE ---
        else if (phase === 'wisdom') {
            const wisdomText = dna?.ageVariants?.[age]?.wisdomMoment || 
                             "You've done great today! Keep staying curious.";
            
            this.completeLesson(); // Mark complete

            contentHTML = `
                <div class="wisdom-badge">✨ Wisdom Moment</div>
                <p class="lesson-text large">${wisdomText}</p>
                <button class="btn-primary-large" onclick="window.app.finishLesson()">
                    Complete Lesson
                </button>
            `;
        }

        // Inject into DOM
        this.dom.stateLesson.innerHTML = `
            <div class="lesson-card glass-panel animate-in">
                <div class="lesson-header">
                    <div class="phase-dots">
                        <div class="dot ${phase === 'welcome' ? 'active' : ''}"></div>
                        <div class="dot ${phase.includes('q') || phase === 'teaching' ? 'active' : ''}"></div>
                        <div class="dot ${phase === 'wisdom' ? 'active' : ''}"></div>
                    </div>
                    <button class="close-btn" onclick="window.app.finishLesson()">×</button>
                </div>
                <div class="lesson-content-body">
                    ${contentHTML}
                </div>
            </div>
        `;
    }

    advancePhase(nextPhase) {
        this.state.currentPhase = nextPhase;
        this.renderLessonPhase();
    }

    handleChoice(idx) {
        // Logic to show feedback could go here
        // For now, jump straight to wisdom
        this.advancePhase('wisdom');
    }

    completeLesson() {
        if (!this.state.completedLessons.includes(this.state.currentLesson.day)) {
            this.state.completedLessons.push(this.state.currentLesson.day);
            
            // Update Streak
            const now = new Date();
            const today = new Date(); today.setHours(0,0,0,0);
            let last = this.state.lastVisit ? new Date(this.state.lastVisit) : null;
            if(last) last.setHours(0,0,0,0);

            if(!last || today > last) {
                this.state.userStreak++;
                this.state.lastVisit = now.toISOString();
            }
            
            this.saveProgress();
            this.updateStreakUI();
        }
    }

    finishLesson() {
        // Reset UI to Greeting
        this.dom.stateLesson.classList.remove('active');
        this.dom.stateLesson.innerHTML = ''; // Clear
        
        this.dom.stateGreeting.style.opacity = '1';
        this.dom.stateGreeting.style.transform = 'translateY(0)';
        this.dom.stateGreeting.classList.add('active');
        
        // Reset Kelly Zoom
        this.setKellyZoom(3); // Back to idle
        
        // Update button to "Review"
        this.updateGreetingUI(this.state.currentLesson);
    }

    // =================================================================
    // KELLY PRESENCE
    // =================================================================

    setKellyZoom(level) {
        this.state.currentZoom = level;
        const newSrc = this.state.kellyImages[level];
        
        // Simple src swap (CSS handles breathing)
        // In production: Cross-fade between two img tags for smoothness
        this.dom.kellyActor.src = newSrc;
    }

    // =================================================================
    // OVERLAYS
    // =================================================================

    openCalendar() {
        this.dom.calendarOverlay.classList.add('active');
        this.renderCalendarGrid();
    }

    closeOverlay() {
        this.dom.calendarOverlay.classList.remove('active');
    }

    renderCalendarGrid() {
        // Simple month render for the overlay
        // Could reuse full CalendarApp logic here
        const container = this.dom.calendarOverlay.querySelector('.calendar-grid-mock');
        if(container && !container.hasChildNodes()) {
            // Just mocking a grid for the visual prototype
            let html = '';
            for(let i=1; i<=30; i++) {
                const isCompleted = this.state.completedLessons.includes(i);
                const isToday = i === this.state.currentLesson?.day;
                
                html += `
                    <div class="day-cell ${isCompleted ? 'completed' : ''} ${isToday ? 'today' : ''}">
                        ${i}
                        ${isCompleted ? '<span class="check">✓</span>' : ''}
                    </div>
                `;
            }
            container.innerHTML = html;
        }
    }

    renderState() {
        // Initialize sidebar state
        if (window.innerWidth <= 768) {
            this.toggleSidebar(false);
        } else {
            this.toggleSidebar(true);
        }
    }
}

// Init
document.addEventListener('DOMContentLoaded', () => {
    window.app = new UnifiedApp();
});
