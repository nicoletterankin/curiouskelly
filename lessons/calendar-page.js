// Calendar Page JavaScript - Layout #1: Side Panel Slide-Over with Google Maps-like Kelly Navigation
class CalendarApp {
    constructor() {
        this.calendarData = null;
        this.currentYear = 2025;
        this.currentMonth = 0;
        this.currentWeek = 0;
        this.currentView = 'today';
        this.currentTab = 'calendar';
        this.selectedDay = null;
        this.currentLesson = null;
        this.currentPhase = 'welcome';
        this.currentAge = '6-12';
        this.currentLanguage = 'en';
        this.phaseHistory = [];
        
        // Zoom state (panel state removed - no longer needed)
        this.currentZoomLevel = 3;
        
        this.init();
    }

    async init() {
        // Setup Kelly image system with all images
        this.setupKellyImageSystem();
        
        // Setup pan/drag navigation
        this.setupPanNavigation();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load calendar data and then render
        await this.loadCalendarData();
        
        // Render initial view after data is loaded
        this.renderView();
    }

    setupKellyImageSystem() {
        // 6-step zoom system using lesson-player images
        this.kellyImages = [
            '../lesson-player/0.png',  // Zoom 0: Most zoomed in
            '../lesson-player/1.png',  // Zoom 1
            '../lesson-player/2.png',  // Zoom 2
            '../lesson-player/3.png',  // Zoom 3
            '../lesson-player/4.jpeg', // Zoom 4
            '../lesson-player/5.png',  // Zoom 5
            '../lesson-player/6.png'   // Zoom 6: Most zoomed out
        ];

        this.zoomLevels = [0, 1, 2, 3, 4, 5, 6];
        this.currentZoomLevel = 3; // Start at middle zoom level
        
        // Load initial image
        this.updateKellyImage();
    }

    setupPanNavigation() {
        // Remove pan hint (no dragging needed)
        const panHint = document.getElementById('pan-hint');
        if (panHint) panHint.style.display = 'none';

        // Mouse wheel zoom only
        const container = document.getElementById('kelly-container');
        container.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 1 : -1;
            if (delta > 0) {
                this.zoomOut();
            } else {
                this.zoomIn();
            }
        });
    }

    updateKellyImage() {
        const kellyImage = document.getElementById('kelly-image');
        const kellyStatus = document.getElementById('kelly-status');
        const kellyFallback = document.getElementById('kelly-fallback');
        
        // Select image based on zoom level
        const imageSrc = this.kellyImages[this.currentZoomLevel];
        
        if (!imageSrc) {
            console.warn('No image found for zoom level:', this.currentZoomLevel);
            kellyImage.style.display = 'none';
            kellyFallback.style.display = 'block';
            kellyFallback.classList.add('show');
            return;
        }
        
        // Fade out current
        kellyImage.style.opacity = '0';
        kellyImage.style.transition = 'opacity 0.4s ease-in-out';
        
        // Load new image
        const newImage = new Image();
        newImage.onload = () => {
            kellyImage.src = imageSrc;
            kellyImage.style.display = 'block';
            kellyFallback.style.display = 'none';
            kellyFallback.classList.remove('show');
            
            setTimeout(() => {
                kellyImage.style.opacity = '1';
                kellyStatus.textContent = 'Ready to learn';
            }, 50);
        };
        
        newImage.onerror = () => {
            console.error('Image failed to load:', imageSrc);
            kellyImage.style.display = 'none';
            kellyFallback.style.display = 'block';
            kellyFallback.classList.add('show');
            kellyStatus.textContent = 'Image not found';
        };
        
        newImage.src = imageSrc;
    }

    setKellyZoomLevel(level) {
        if (level >= 0 && level < this.kellyImages.length) {
            this.currentZoomLevel = level;
            this.updateKellyImage();
            this.updateZoomControls();
        }
    }

    zoomIn() {
        if (this.currentZoomLevel > 0) {
            this.setKellyZoomLevel(this.currentZoomLevel - 1);
        }
    }

    zoomOut() {
        if (this.currentZoomLevel < this.kellyImages.length - 1) {
            this.setKellyZoomLevel(this.currentZoomLevel + 1);
        }
    }

    updateZoomControls() {
        const zoomInBtn = document.getElementById('zoom-in-btn');
        const zoomOutBtn = document.getElementById('zoom-out-btn');
        const zoomIndicator = document.getElementById('zoom-indicator');
        
        if (zoomInBtn) zoomInBtn.disabled = this.currentZoomLevel === 0;
        if (zoomOutBtn) zoomOutBtn.disabled = this.currentZoomLevel === this.kellyImages.length - 1;
        if (zoomIndicator) {
            zoomIndicator.textContent = `Level ${this.currentZoomLevel + 1}/${this.kellyImages.length}`;
        }
    }

    async loadCalendarData() {
        try {
            console.log('Loading calendar data...');
            const response = await fetch('365_day_calendar.json');
            if (!response.ok) {
                throw new Error(`Failed to load calendar: ${response.status} ${response.statusText}`);
            }
            this.calendarData = await response.json();
            console.log('Calendar data loaded:', this.calendarData.lessons?.length || 0, 'lessons');
            
            // Hide CORS warning if data loaded successfully
            const corsWarning = document.getElementById('cors-warning');
            if (corsWarning) {
                corsWarning.classList.remove('show');
            }
            
            const today = new Date();
            this.todayDay = today.getDate();
            this.currentMonth = today.getMonth();
            this.currentYear = today.getFullYear();
            
            const todayLesson = this.calendarData.lessons.find(l => {
                const lessonDate = new Date(l.date + ', ' + this.currentYear);
                return lessonDate.getDate() === this.todayDay && 
                       lessonDate.getMonth() === this.currentMonth;
            });
            
            if (todayLesson) {
                console.log('Found today\'s lesson:', todayLesson.title, 'Day', todayLesson.day);
                this.selectedDay = todayLesson.day;
                await this.loadLesson(todayLesson.day);
            } else {
                console.log('No lesson found for today, defaulting to day 1');
                this.selectedDay = 1;
                // Load day 1 lesson if available
                const day1Lesson = this.calendarData.lessons.find(l => l.day === 1);
                if (day1Lesson) {
                    await this.loadLesson(1);
                }
            }
        } catch (error) {
            console.error('Error loading calendar data:', error);
            this.calendarData = { lessons: [] };
            
            // Check if it's a CORS error
            if (error.message.includes('Failed to fetch') || error.message.includes('CORS')) {
                // Show CORS warning banner
                const corsWarning = document.getElementById('cors-warning');
                if (corsWarning) {
                    corsWarning.classList.add('show');
                }
            }
            
            // Show error in UI
            const card = document.getElementById('today-card');
            if (card) {
                card.innerHTML = `<div class="empty-state" style="color: #ef4444;">Error loading calendar: ${error.message}<br><br>Please run a local web server (see warning banner above).</div>`;
            }
        }
    }

    setupEventListeners() {
        // Panel toggle
        document.getElementById('panel-toggle').addEventListener('click', () => {
            this.togglePanel();
        });

        document.getElementById('panel-close').addEventListener('click', () => {
            this.closePanel();
        });

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Calendar view selector
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchCalendarView(e.target.dataset.view);
            });
        });

        // Month navigation
        document.getElementById('prev-month').addEventListener('click', () => {
            this.currentMonth--;
            if (this.currentMonth < 0) {
                this.currentMonth = 11;
                this.currentYear--;
            }
            this.renderView();
        });

        document.getElementById('next-month').addEventListener('click', () => {
            this.currentMonth++;
            if (this.currentMonth > 11) {
                this.currentMonth = 0;
                this.currentYear++;
            }
            this.renderView();
        });

        // Week navigation
        document.getElementById('prev-week').addEventListener('click', () => {
            this.currentWeek--;
            this.renderView();
        });

        document.getElementById('next-week').addEventListener('click', () => {
            this.currentWeek++;
            this.renderView();
        });

        // Phase navigation
        document.getElementById('prev-phase').addEventListener('click', () => {
            this.goToPreviousPhase();
        });

        document.getElementById('next-phase').addEventListener('click', () => {
            this.goToNextPhase();
        });

        // Settings
        document.getElementById('age-selector').addEventListener('change', (e) => {
            this.currentAge = e.target.value;
            this.updateLessonContent();
        });

        document.getElementById('language-selector').addEventListener('change', (e) => {
            this.currentLanguage = e.target.value;
            this.updateLessonContent();
        });

        // Zoom controls
        document.getElementById('zoom-in-btn').addEventListener('click', () => {
            this.zoomIn();
        });

        document.getElementById('zoom-out-btn').addEventListener('click', () => {
            this.zoomOut();
        });

        // Keyboard zoom controls
        document.addEventListener('keydown', (e) => {
            if (e.key === '+' || e.key === '=') {
                e.preventDefault();
                this.zoomIn();
            } else if (e.key === '-' || e.key === '_') {
                e.preventDefault();
                this.zoomOut();
            }
        });
    }

    togglePanel() {
        const panel = document.getElementById('side-panel');
        const toggle = document.getElementById('panel-toggle');
        panel.classList.toggle('open');
        toggle.classList.toggle('panel-open');
    }

    closePanel() {
        const panel = document.getElementById('side-panel');
        const toggle = document.getElementById('panel-toggle');
        panel.classList.remove('open');
        toggle.classList.remove('panel-open');
    }

    switchTab(tab) {
        this.currentTab = tab;
        
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tab);
        });

        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `tab-${tab}`);
        });

        if (!document.getElementById('side-panel').classList.contains('open')) {
            this.togglePanel();
        }

        if (tab === 'calendar') {
            this.renderView();
        } else if (tab === 'lesson') {
            this.renderLessonTab();
        }
    }

    switchCalendarView(view) {
        this.currentView = view;
        
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });

        document.querySelectorAll('.calendar-view').forEach(v => {
            v.classList.toggle('active', v.id === `view-${view}`);
        });

        this.renderView();
    }

    renderView() {
        // Don't render if calendar data isn't loaded yet
        if (!this.calendarData || !this.calendarData.lessons) {
            console.log('Calendar data not loaded yet, skipping render');
            return;
        }

        switch (this.currentView) {
            case 'today':
                this.renderToday();
                break;
            case 'year':
                this.renderYearView();
                break;
            case 'month':
                this.renderMonthView();
                break;
            case 'week':
                this.renderWeekView();
                break;
        }
    }

    renderToday() {
        if (!this.calendarData || !this.calendarData.lessons) {
            const card = document.getElementById('today-card');
            card.innerHTML = '<div class="empty-state">Loading calendar...</div>';
            return;
        }

        const lesson = this.calendarData.lessons.find(l => l.day === this.selectedDay);
        if (!lesson) {
            const card = document.getElementById('today-card');
            card.innerHTML = '<div class="empty-state">No lesson found for selected day</div>';
            return;
        }

        const card = document.getElementById('today-card');
        const isToday = this.isLessonToday(lesson);
        
        card.innerHTML = `
            <div class="day-label">Day ${lesson.day} - ${lesson.date}${isToday ? ' (Today)' : ''}</div>
            ${lesson.has_dna ? '<span class="dna-badge">🧬 DNA</span>' : ''}
            <div class="lesson-title">${lesson.title}</div>
            <div class="lesson-objective" style="font-size: 12px; color: #666; margin-top: 8px;">${lesson.learning_objective || ''}</div>
            <button class="option-btn" style="margin-top: 12px; text-align: center;" onclick="calendarApp.selectDay(${lesson.day})">
                Open Lesson
            </button>
        `;
    }

    renderYearView() {
        const container = document.getElementById('year-grid');
        const monthNames = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December'];
        
        container.innerHTML = monthNames.map((month, index) => {
            const monthLessons = this.getLessonsForMonth(index);
            const dnaCount = monthLessons.filter(l => l.has_dna).length;
            
            return `
                <div class="month-card" onclick="calendarApp.openMonth(${index})">
                    <h4>${month}</h4>
                    <div class="month-stats">
                        <span>📚 ${monthLessons.length}</span>
                        ${dnaCount > 0 ? `<span>🧬 ${dnaCount}</span>` : ''}
                    </div>
                </div>
            `;
        }).join('');
    }

    openMonth(monthIndex) {
        this.currentMonth = monthIndex;
        this.renderView();
        this.switchCalendarView('month');
    }

    renderMonthView() {
        const container = document.getElementById('month-calendar');
        const header = document.getElementById('current-month-name');
        const monthNames = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December'];
        header.textContent = `${monthNames[this.currentMonth]} ${this.currentYear}`;

        const monthLessons = this.getLessonsForMonth(this.currentMonth);
        const dayMap = {};
        monthLessons.forEach(lesson => {
            const lessonDate = new Date(lesson.date + ', ' + this.currentYear);
            dayMap[lessonDate.getDate()] = lesson;
        });

        const firstDay = new Date(this.currentYear, this.currentMonth, 1).getDay();
        const daysInMonth = new Date(this.currentYear, this.currentMonth + 1, 0).getDate();
        const today = new Date();
        const dayHeaders = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];

        let html = dayHeaders.map(day => `<div class="day-header">${day}</div>`).join('');

        for (let i = 0; i < firstDay; i++) {
            html += '<div class="day-cell empty"></div>';
        }

        for (let day = 1; day <= daysInMonth; day++) {
            const lesson = dayMap[day];
            const isToday = today.getDate() === day && 
                           today.getMonth() === this.currentMonth && 
                           today.getFullYear() === this.currentYear;
            const classes = ['day-cell'];
            if (isToday) classes.push('today');
            if (lesson?.has_dna) classes.push('has-dna');

            html += `
                <div class="${classes.join(' ')}" 
                     onclick="calendarApp.selectDay(${lesson?.day || day})"
                     title="${lesson?.title || 'No lesson'}">
                    ${day}
                </div>
            `;
        }

        container.innerHTML = html;
    }

    renderWeekView() {
        const container = document.getElementById('week-list');
        const header = document.getElementById('current-week-range');
        
        const weekStart = this.getWeekStartDate();
        header.textContent = `Week of ${weekStart.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;

        const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
        let html = '';

        for (let i = 0; i < 7; i++) {
            const date = new Date(weekStart);
            date.setDate(date.getDate() + i);
            
            const lesson = this.calendarData.lessons.find(l => {
                const lessonDate = new Date(l.date + ', ' + this.currentYear);
                return lessonDate.getDate() === date.getDate() &&
                       lessonDate.getMonth() === date.getMonth();
            });

            if (lesson) {
                const isToday = this.isLessonToday(lesson);
                html += `
                    <div class="week-day-card ${lesson.has_dna ? 'has-dna' : ''}" 
                         onclick="calendarApp.selectDay(${lesson.day})">
                        <div class="week-day-header">${dayNames[date.getDay()]}</div>
                        <div class="week-day-title">${lesson.title}</div>
                        ${isToday ? '<div style="color: #10b981; font-size: 9px; margin-top: 4px;">Today</div>' : ''}
                    </div>
                `;
            }
        }

        container.innerHTML = html;
    }

    getWeekStartDate() {
        const date = new Date(this.currentYear, 0, 1);
        date.setDate(date.getDate() + (this.currentWeek * 7));
        const day = date.getDay();
        const diff = date.getDate() - day;
        date.setDate(diff);
        return date;
    }

    async selectDay(day) {
        this.selectedDay = day;
        await this.loadLesson(day);
        this.switchTab('lesson');
    }

    async loadLesson(day) {
        const lesson = this.calendarData.lessons.find(l => l.day === day);
        if (!lesson) return;

        this.currentLesson = lesson;
        this.currentPhase = 'welcome';
        this.phaseHistory = [];

        if (lesson.has_dna && lesson.dna_file) {
            try {
                // DNA files are in the same directory with -dna.json suffix
                const dnaFileName = `${lesson.dna_file}-dna.json`;
                const dnaResponse = await fetch(dnaFileName);
                if (dnaResponse.ok) {
                    this.lessonDNA = await dnaResponse.json();
                    console.log('Loaded DNA lesson:', dnaFileName);
                } else {
                    console.warn('DNA file not found:', dnaFileName, dnaResponse.status);
                    this.lessonDNA = null;
                }
            } catch (error) {
                console.error('Error loading DNA lesson:', error);
                this.lessonDNA = null;
            }
        } else {
            this.lessonDNA = null;
        }

        this.renderLessonTab();
        this.updateControls();
    }

    renderLessonTab() {
        if (!this.currentLesson) {
            document.getElementById('lesson-title').textContent = 'No lesson selected';
            document.getElementById('lesson-day').textContent = '';
            document.getElementById('phase-content').innerHTML = '<div class="empty-phase"><p>Select a lesson to begin</p></div>';
            return;
        }

        document.getElementById('lesson-title').textContent = this.currentLesson.title;
        document.getElementById('lesson-day').textContent = `Day ${this.currentLesson.day} - ${this.currentLesson.date}`;

        this.updatePhaseProgress();
        this.renderPhase();
    }

    updatePhaseProgress() {
        const phases = ['welcome', 'q1', 'q2', 'q3', 'wisdom'];
        phases.forEach((phase, index) => {
            const indicator = document.querySelector(`.phase-indicator[data-phase="${phase}"]`);
            if (!indicator) return;

            indicator.classList.remove('active', 'completed');
            
            const currentIndex = phases.indexOf(this.currentPhase);
            if (index < currentIndex) {
                indicator.classList.add('completed');
            } else if (index === currentIndex) {
                indicator.classList.add('active');
            }
        });
    }

    renderPhase() {
        const content = document.getElementById('phase-content');
        
        if (!this.lessonDNA) {
            content.innerHTML = `
                <div class="phase-question">${this.currentLesson.learning_objective || this.currentLesson.title}</div>
                <div class="empty-phase">
                    <p>This lesson doesn't have interactive DNA content yet.</p>
                    <p style="font-size: 12px; color: #666; margin-top: 8px;">Lesson: ${this.currentLesson.title}</p>
                </div>
            `;
            return;
        }

        if (!this.lessonDNA.interactions || !Array.isArray(this.lessonDNA.interactions)) {
            console.warn('No interactions array found in DNA:', this.lessonDNA);
            content.innerHTML = `
                <div class="phase-question">${this.currentLesson.learning_objective || this.currentLesson.title}</div>
                <div class="empty-phase">
                    <p>This lesson's DNA file doesn't have interactions yet.</p>
                </div>
            `;
            return;
        }

        let interaction = null;
        
        if (this.currentPhase === 'welcome') {
            interaction = this.lessonDNA.interactions.find(i => i.step === 'welcome' || i.phase === 'welcome');
        } else if (this.currentPhase === 'wisdom') {
            interaction = this.lessonDNA.interactions.find(i => i.step === 'wisdom' || i.phase === 'wisdom');
        } else {
            // Find question interactions (not welcome/wisdom)
            const questionInteractions = this.lessonDNA.interactions.filter(i => {
                const step = i.step || i.phase || '';
                return step !== 'welcome' && step !== 'wisdom';
            });
            const qIndex = ['q1', 'q2', 'q3'].indexOf(this.currentPhase);
            if (qIndex >= 0 && qIndex < questionInteractions.length) {
                interaction = questionInteractions[qIndex];
            }
        }

        if (!interaction) {
            console.warn('No interaction found for phase:', this.currentPhase, 'Available interactions:', this.lessonDNA.interactions.map(i => i.step || i.phase));
            content.innerHTML = `
                <div class="empty-phase">
                    <p>Phase content not available for "${this.currentPhase}"</p>
                    <p style="font-size: 12px; color: #666; margin-top: 8px;">Available phases: ${this.lessonDNA.interactions.map(i => i.step || i.phase || 'unknown').join(', ')}</p>
                </div>
            `;
            return;
        }

        // Get age-adapted content
        const ageAdaptation = interaction.ageAdaptations?.[this.currentAge];
        const question = ageAdaptation?.question || interaction.question || 'No question available';
        const choices = ageAdaptation?.choices || interaction.choices || [];

        // Get age variant for language content
        const ageVariant = this.lessonDNA.ageVariants?.[this.currentAge];
        const languageContent = ageVariant?.language?.[this.currentLanguage] || ageVariant?.language?.en;

        if (this.currentPhase === 'welcome') {
            // Welcome phase - show welcome message
            const welcomeText = languageContent?.welcome || 
                              ageVariant?.script || 
                              interaction.welcomeText ||
                              'Welcome! Let\'s learn together today.';
            
            content.innerHTML = `
                <div class="wisdom-content">
                    <div class="wisdom-text">${welcomeText}</div>
                </div>
            `;
        } else if (this.currentPhase === 'wisdom') {
            // Wisdom phase
            const wisdomText = languageContent?.wisdomMoment || 
                             ageVariant?.wisdomMoment ||
                             interaction.wisdomText ||
                             'Thank you for learning with me today!';
            
            content.innerHTML = `
                <div class="wisdom-content">
                    <div class="wisdom-text">${wisdomText}</div>
                </div>
            `;
        } else {
            // Question phases - show question and choices
            if (!choices || choices.length === 0) {
                content.innerHTML = `
                    <div class="phase-question">${question}</div>
                    <div class="empty-phase">
                        <p>No answer choices available for this question.</p>
                    </div>
                `;
                return;
            }

            // Show up to 2 choices (as per original design)
            const options = choices.slice(0, 2);
            
            content.innerHTML = `
                <div class="phase-question">${question}</div>
                <div class="phase-options">
                    ${options.map((choice, index) => `
                        <button class="option-btn" data-choice-index="${index}" onclick="calendarApp.selectOption(${index})">
                            <span class="option-text">${choice.text || choice.label || 'Option ' + (index + 1)}</span>
                        </button>
                    `).join('')}
                </div>
            `;
        }
    }

    selectOption(choiceIndex) {
        const interaction = this.getCurrentInteraction();
        if (!interaction) return;

        const ageAdaptation = interaction.ageAdaptations?.[this.currentAge];
        const choices = ageAdaptation?.choices || interaction.choices;
        const choice = choices[choiceIndex];
        
        if (!choice) return;

        document.querySelectorAll('.option-btn').forEach(btn => {
            btn.classList.remove('selected');
        });
        document.querySelector(`.option-btn[data-choice-index="${choiceIndex}"]`).classList.add('selected');

        const kellyStatus = document.getElementById('kelly-status');
        kellyStatus.textContent = 'Responding...';

        console.log('Selected option:', choice.text);
        console.log('Response:', choice.response);
        
        setTimeout(() => {
            kellyStatus.textContent = 'Ready for next phase';
            this.phaseHistory.push({
                phase: this.currentPhase,
                choice: choiceIndex,
                response: choice.response
            });
            this.updateControls();
        }, 2000);
    }

    getCurrentInteraction() {
        if (!this.lessonDNA || !this.lessonDNA.interactions) return null;

        if (this.currentPhase === 'welcome') {
            return this.lessonDNA.interactions.find(i => i.step === 'welcome');
        } else if (this.currentPhase === 'wisdom') {
            return this.lessonDNA.interactions.find(i => i.step === 'wisdom');
        } else {
            const questionInteractions = this.lessonDNA.interactions.filter(i => 
                i.step !== 'welcome' && i.step !== 'wisdom'
            );
            const qIndex = ['q1', 'q2', 'q3'].indexOf(this.currentPhase);
            if (qIndex >= 0 && qIndex < questionInteractions.length) {
                return questionInteractions[qIndex];
            }
        }
        return null;
    }

    goToPreviousPhase() {
        const phases = ['welcome', 'q1', 'q2', 'q3', 'wisdom'];
        const currentIndex = phases.indexOf(this.currentPhase);
        if (currentIndex > 0) {
            this.currentPhase = phases[currentIndex - 1];
            this.renderPhase();
            this.updatePhaseProgress();
            this.updateControls();
        }
    }

    goToNextPhase() {
        const phases = ['welcome', 'q1', 'q2', 'q3', 'wisdom'];
        const currentIndex = phases.indexOf(this.currentPhase);
        if (currentIndex < phases.length - 1) {
            this.currentPhase = phases[currentIndex + 1];
            this.renderPhase();
            this.updatePhaseProgress();
            this.updateControls();
        }
    }

    updateControls() {
        const phases = ['welcome', 'q1', 'q2', 'q3', 'wisdom'];
        const currentIndex = phases.indexOf(this.currentPhase);
        
        const prevBtn = document.getElementById('prev-phase');
        const nextBtn = document.getElementById('next-phase');
        const phaseName = document.getElementById('current-phase-name');

        prevBtn.disabled = currentIndex === 0;
        nextBtn.disabled = currentIndex === phases.length - 1;

        const phaseNames = {
            'welcome': 'Welcome',
            'q1': 'Question 1',
            'q2': 'Question 2',
            'q3': 'Question 3',
            'wisdom': 'Wisdom'
        };

        phaseName.textContent = phaseNames[this.currentPhase] || 'No lesson active';
    }

    updateLessonContent() {
        if (this.currentLesson) {
            this.renderPhase();
        }

        const info = document.getElementById('current-lesson-info');
        if (this.currentLesson) {
            info.innerHTML = `
                <p><strong>${this.currentLesson.title}</strong></p>
                <p>Age: ${this.currentAge}</p>
                <p>Language: ${this.currentLanguage.toUpperCase()}</p>
            `;
        }
    }

    getLessonsForMonth(monthIndex) {
        const monthNames = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December'];
        const monthName = monthNames[monthIndex];
        
        return this.calendarData.lessons.filter(lesson => {
            return lesson.date.startsWith(monthName);
        });
    }

    isLessonToday(lesson) {
        const today = new Date();
        const lessonDate = new Date(lesson.date + ', ' + this.currentYear);
        return lessonDate.getDate() === today.getDate() &&
               lessonDate.getMonth() === today.getMonth() &&
               lessonDate.getFullYear() === today.getFullYear();
    }
}

// Initialize app
let calendarApp;
document.addEventListener('DOMContentLoaded', () => {
    calendarApp = new CalendarApp();
});
