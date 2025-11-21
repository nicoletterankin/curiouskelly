class CalendarBridge {
    constructor(container, options = {}) {
        this.container = container;
        this.onLessonSelected = options.onLessonSelected || (() => {});
        this.drawerOpen = false;
        this.lessons = Array.isArray(options.initialLessons) ? options.initialLessons : [];
        this.enrichedLessons = [];
        this.activeLessonDay = options.initialLesson?.day || null;
        this.today = new Date();
        this.currentYear = this.today.getFullYear();
        this.state = {
            loading: !this.lessons.length,
            error: null
        };

        if (this.lessons.length) {
            this.decorateLessons();
        } else {
            this.loadLessons();
        }

        this.render();
    }

    async loadLessons() {
        try {
            this.state.loading = true;
            this.render();

            const response = await fetch('../lessons/365_day_calendar.json', { cache: 'no-cache' });
            if (!response.ok) {
                throw new Error(`Failed to load calendar (${response.status})`);
            }

            const data = await response.json();
            this.setLessons(data.lessons || []);
        } catch (error) {
            console.error('CalendarBridge load error:', error);
            this.state.loading = false;
            this.state.error = 'Unable to load calendar data.';
            this.render();
        }
    }

    setLessons(lessons = []) {
        this.lessons = Array.isArray(lessons) ? lessons : [];
        this.state.loading = false;
        this.state.error = null;
        this.decorateLessons();
        this.render();
    }

    decorateLessons() {
        this.enrichedLessons = this.lessons.map((lesson) => {
            return {
                ...lesson,
                dateObj: this.parseLessonDate(lesson.date)
            };
        });
    }

    parseLessonDate(dateString) {
        if (!dateString) return null;
        const parsed = new Date(`${dateString}, ${this.currentYear}`);
        return Number.isNaN(parsed) ? null : parsed;
    }

    setActiveLesson(lessonSummary) {
        if (!lessonSummary) return;
        const resolvedLesson = this.findLessonByDay(lessonSummary.day) || lessonSummary;
        this.activeLessonDay = resolvedLesson?.day ?? null;
        this.currentActiveLesson = resolvedLesson;
        this.render();
    }

    findLessonByDay(day) {
        if (!day || !this.enrichedLessons.length) return null;
        return this.enrichedLessons.find((lesson) => Number(lesson.day) === Number(day));
    }

    toggleDrawer() {
        this.drawerOpen = !this.drawerOpen;
        this.render();
    }

    handleLessonClick(dayValue) {
        const lesson = this.findLessonByDay(dayValue);
        if (!lesson || !lesson.has_dna || !lesson.dna_file) {
            return;
        }

        this.activeLessonDay = lesson.day;
        this.currentActiveLesson = lesson;
        this.drawerOpen = false;
        this.render();

        if (typeof this.onLessonSelected === 'function') {
            this.onLessonSelected(lesson);
        }
    }

    getLessonSubset(count = 6) {
        if (!this.enrichedLessons.length) {
            return [];
        }

        const todayIndex = this.enrichedLessons.findIndex((lesson) => {
            if (!lesson.dateObj) return false;
            return this.isSameDay(lesson.dateObj, this.today);
        });

        const startIndex = todayIndex === -1 ? 0 : Math.max(0, todayIndex - 1);
        const subset = this.enrichedLessons.slice(startIndex, startIndex + count);
        return subset.length ? subset : this.enrichedLessons.slice(0, count);
    }

    isSameDay(dateA, dateB) {
        if (!dateA || !dateB) return false;
        return (
            dateA.getDate() === dateB.getDate() &&
            dateA.getMonth() === dateB.getMonth() &&
            dateA.getFullYear() === dateB.getFullYear()
        );
    }

    formatDate(dateObj) {
        if (!dateObj) return 'Date TBA';
        return dateObj.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    }

    renderLessonItem(lesson) {
        const disabled = !lesson.has_dna || !lesson.dna_file;
        const isActive = Number(lesson.day) === Number(this.activeLessonDay);

        return `
            <li class="bridge-lesson ${disabled ? 'disabled' : ''} ${isActive ? 'active' : ''}"
                data-day="${lesson.day}"
                data-disabled="${disabled}">
                <div class="bridge-lesson-details">
                    <span class="bridge-lesson-title">${this.escapeHtml(lesson.title || `Day ${lesson.day}`)}</span>
                    <span class="bridge-lesson-date">${this.formatDate(lesson.dateObj)}</span>
                </div>
                <div class="bridge-lesson-meta">
                    <span class="bridge-dna-badge">${disabled ? 'PENDING' : 'DNA'}</span>
                </div>
            </li>
        `;
    }

    renderBody() {
        if (this.state.error) {
            return `<p class="bridge-error">${this.escapeHtml(this.state.error)}</p>`;
        }

        if (this.state.loading) {
            return `<div class="calendar-bridge-loading">Loading topics…</div>`;
        }

        const subset = this.getLessonSubset();
        if (!subset.length) {
            return `<p class="bridge-open-note">No lessons available yet. Please add topics to the calendar.</p>`;
        }

        const listMarkup = subset.map((lesson) => this.renderLessonItem(lesson)).join('');
        return `
            <ul class="bridge-lesson-list">
                ${listMarkup}
            </ul>
            <p class="bridge-open-note">Select a DNA-ready topic to jump directly into the player.</p>
        `;
    }

    render() {
        if (!this.container) return;

        const activeLesson = this.getActiveLessonSummary();
        this.container.innerHTML = `
            <div class="calendar-bridge-header">
                <div>
                    <p class="meta-label">Calendar bridge</p>
                    <p class="calendar-bridge-summary-title">
                        ${activeLesson.title}
                    </p>
                    <p class="bridge-open-note">${activeLesson.subtitle}</p>
                </div>
                <button class="bridge-toggle" type="button" aria-expanded="${this.drawerOpen}">
                    ${this.drawerOpen ? 'Close' : 'Browse'}
                </button>
            </div>
            <div class="calendar-bridge-body ${this.drawerOpen ? 'open' : ''}">
                ${this.renderBody()}
            </div>
        `;

        this.attachEvents();
    }

    attachEvents() {
        const toggleBtn = this.container.querySelector('.bridge-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleDrawer());
        }

        const lessonItems = this.container.querySelectorAll('.bridge-lesson');
        lessonItems.forEach((item) => {
            const disabled = item.dataset.disabled === 'true';
            if (disabled) return;

            item.addEventListener('click', () => {
                const dayValue = Number(item.dataset.day);
                this.handleLessonClick(dayValue);
            });
        });
    }

    getActiveLessonSummary() {
        const lesson = this.findLessonByDay(this.activeLessonDay) || this.currentActiveLesson;
        if (!lesson) {
            return {
                title: 'Select a topic',
                subtitle: 'Browse Kelly’s daily lessons to jump right in.'
            };
        }

        return {
            title: lesson.title || `Day ${lesson.day}`,
            subtitle: `Day ${lesson.day} · ${this.formatDate(lesson.dateObj)}`
        };
    }

    escapeHtml(value) {
        if (typeof value !== 'string') return value;
        return value
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
}

if (typeof window !== 'undefined') {
    window.CalendarBridge = CalendarBridge;
}








