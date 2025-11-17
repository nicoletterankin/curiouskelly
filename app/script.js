import StateManager from './state-manager.js';
import SessionClient from './session-client.js';
import UnityBridge from './unity-bridge.js';

const PHASE_ALIASES = {
  welcome: 'welcome',
  teaching: 'teaching',
  practice: 'practice',
  wisdom: 'wisdom',
  q1: 'teaching',
  q2: 'practice',
  q3: 'practice',
  q4: 'wisdom',
};

class UnifiedLessonApp {
  constructor() {
    this.sessionClient = new SessionClient();
    this.unityBridge = new UnityBridge();
    this.stateManager = new StateManager({
      age: 25,
      ageBucket: '18-35',
      language: 'en',
      currentView: 'today',
      calendarLessons: [],
      todayLesson: null,
      selectedLesson: null,
      selectedDay: null,
      monthOffset: 0,
      lessonData: null,
      currentPhase: 'welcome',
      isPlaying: false,
      streak: 0,
      sessionId: null,
      sessionLessonId: null,
      sessionState: null,
      sessionCompleted: false,
    });

    this.bucketRanges = {
      '2-5': [2, 5],
      '6-12': [6, 12],
      '13-17': [13, 17],
      '18-35': [18, 35],
      '36-60': [36, 60],
      '61-102': [61, 102],
    };

    this.elements = {};
  }

  init() {
    this.cacheElements();
    this.bindEvents();
    this.setupStateSubscriptions();
    this.updateDateLabel();
    this.updateAgeDisplay(this.stateManager.getState().age);
    this.highlightBucket(this.stateManager.getState().ageBucket);
    this.loadCalendarData();
    this.setupUnityBridgeTransports();
  }

  cacheElements() {
    this.elements.ageSlider = document.getElementById('age-slider');
    this.elements.ageValue = document.getElementById('age-value');
    this.elements.ageBuckets = document.querySelectorAll('.age-bucket-floating');
    this.elements.languageSelector = document.getElementById('language-selector');
    this.elements.calendarDate = document.getElementById('calendar-date');
    this.elements.viewPills = document.querySelectorAll('.view-pill');
    this.elements.calendarViews = document.querySelectorAll('.calendar-view');
    this.elements.todayLessonTitle = document.getElementById('today-lesson-title');
    this.elements.todayLessonObjective = document.getElementById('today-lesson-objective');
    this.elements.todayLessonTags = document.getElementById('today-lesson-tags');
    this.elements.todayLessonDuration = document.getElementById('today-lesson-duration');
    this.elements.todayLessonDay = document.getElementById('today-lesson-day');
    this.elements.lessonDayChip = document.getElementById('lesson-day-chip');
    this.elements.lessonTitle = document.getElementById('lesson-title');
    this.elements.topicDescription = document.getElementById('topic-description');
    this.elements.learningObjectives = document.getElementById('learning-objectives');
    this.elements.lessonDuration = document.getElementById('lesson-duration');
    this.elements.lessonTags = document.getElementById('lesson-tags');
    this.elements.choiceContainer = document.getElementById('choice-cards-container');
    this.elements.questionText = document.getElementById('question-text');
    this.elements.phasePill = document.getElementById('current-phase-pill');
    this.elements.audioScript = document.getElementById('audio-script');
    this.elements.playButton = document.getElementById('play-pause');
    this.elements.playIcon = document.getElementById('play-icon');
    this.elements.weekStrip = document.getElementById('week-strip');
    this.elements.monthGrid = document.getElementById('month-grid');
    this.elements.currentMonthLabel = document.getElementById('current-month-label');
    this.elements.prevMonth = document.getElementById('prev-month');
    this.elements.nextMonth = document.getElementById('next-month');
    this.elements.resumeBtn = document.getElementById('resume-btn');
    this.elements.startTodayBtn = document.getElementById('start-today-btn');
    this.elements.openCalendarBtn = document.getElementById('open-calendar-btn');
    this.elements.panelMenuBtn = document.getElementById('panel-menu-btn');
    this.elements.panelMenu = document.getElementById('panel-menu');
    this.elements.streakCount = document.getElementById('streak-count');
    this.elements.sessionStatus = document.getElementById('session-status');
    this.elements.unityOverlay = document.getElementById('unity-overlay');
    this.elements.unityRibbon = document.getElementById('unity-lock-ribbon');
    this.elements.unityIframe = document.getElementById('unity-iframe');
    this.elements.unityStatusLabel = document.getElementById('unity-status-label');
  }

  bindEvents() {
    this.elements.ageSlider?.addEventListener('input', (event) => {
      const value = Number(event.target.value);
      const bucket = this.getBucketForAge(value);
      this.stateManager.setState({ age: value, ageBucket: bucket });
      this.updateAgeDisplay(value);
      this.highlightBucket(bucket);
    });

    this.elements.ageBuckets?.forEach((bucket) => {
      bucket.addEventListener('click', () => {
        const bucketId = bucket.dataset.age;
        if (!bucketId) return;
        const [min, max] = this.bucketRanges[bucketId];
        const midpoint = Math.round((min + max) / 2);
        this.elements.ageSlider.value = midpoint;
        this.stateManager.setState({ age: midpoint, ageBucket: bucketId });
        this.updateAgeDisplay(midpoint);
        this.highlightBucket(bucketId);
      });
    });

    this.elements.languageSelector?.addEventListener('change', (event) => {
      this.stateManager.setState({ language: event.target.value });
      this.elements.sessionStatus.textContent = `Language set to ${event.target.value.toUpperCase()}`;
    });

    this.elements.viewPills?.forEach((pill) => {
      pill.addEventListener('click', () => this.changeView(pill.dataset.view));
    });

    this.elements.prevMonth?.addEventListener('click', () => this.shiftMonth(-1));
    this.elements.nextMonth?.addEventListener('click', () => this.shiftMonth(1));

    this.elements.playButton?.addEventListener('click', () => this.togglePlay());

    this.elements.resumeBtn?.addEventListener('click', () => this.scrollToLesson());
    this.elements.startTodayBtn?.addEventListener('click', () => this.scrollToTodayLesson());
    this.elements.openCalendarBtn?.addEventListener('click', () => {
      window.open('../lessons/calendar-page.html', '_blank');
    });

    this.elements.panelMenuBtn?.addEventListener('click', (event) => {
      event.stopPropagation();
      this.elements.panelMenu?.classList.toggle('show');
    });

    document.addEventListener('click', (event) => {
      if (
        !this.elements.panelMenu?.contains(event.target) &&
        !this.elements.panelMenuBtn?.contains(event.target)
      ) {
        this.elements.panelMenu?.classList.remove('show');
      }
    });
  }

  setupUnityBridgeTransports() {
    const docBtn = document.getElementById('unity-doc-btn');
    docBtn?.addEventListener('click', () => {
      window.open('UNITY_BRIDGE_PLAN.md', '_blank', 'noopener');
    });

    if (this.elements.unityIframe) {
      const src = this.elements.unityIframe.dataset.src;
      if (src) {
        this.elements.unityIframe.src = src;
      }
      this.elements.unityIframe.addEventListener('load', () => {
        const origin = this.elements.unityIframe.dataset.targetOrigin || '*';
        if (this.elements.unityIframe.contentWindow) {
          this.unityBridge.connectToIframe(this.elements.unityIframe.contentWindow, origin);
        }
      });
      const wsUrl = this.elements.unityIframe.dataset.wsUrl;
      if (wsUrl) {
        this.unityBridge.connectWebSocket(wsUrl);
      }
    }

    this.unityBridge.onStatusChange = (status) => this.setUnityStatus(status);
    this.unityBridge.onTelemetry = (telemetry) => this.setUnityTelemetry(telemetry);
    this.unityBridge.onConnectionChange = (_channel, state) => {
      if (state === 'connected') {
        this.showUnityStream(true);
      } else if (state === 'disconnected') {
        this.showUnityStream(this.unityBridge.hasActiveTransport());
      }
    };

    this.showUnityStream(false);
  }

  setupStateSubscriptions() {
    this.stateManager.subscribe((state, prev) => {
      if (state.selectedLesson !== prev.selectedLesson && state.selectedLesson) {
        this.renderTodayCard(state.selectedLesson);
        this.renderLessonOverviewFromSummary(state.selectedLesson);
        this.highlightSelectedCells();
        this.loadLessonDNA(state.selectedLesson);
      }

      if (
        state.lessonData !== prev.lessonData ||
        state.ageBucket !== prev.ageBucket ||
        state.language !== prev.language
      ) {
        this.updateLessonMetaFromVariant(state);
        this.renderPhase(state);
      } else if (state.currentPhase !== prev.currentPhase) {
        this.renderPhase(state);
      }

      if (
        state.calendarLessons !== prev.calendarLessons ||
        state.monthOffset !== prev.monthOffset
      ) {
        this.renderMonthGrid();
      }

      if (
        state.calendarLessons !== prev.calendarLessons ||
        state.selectedLesson !== prev.selectedLesson
      ) {
        this.renderWeekStrip();
      }

      if (state.selectedDay !== prev.selectedDay) {
        this.highlightSelectedCells();
      }

      if (state.currentView !== prev.currentView) {
        this.updateViewVisibility(state.currentView);
      }

      if (state.streak !== prev.streak) {
        this.updateStreakPlaceholder();
      }
    });
  }

  async loadCalendarData() {
    try {
      const response = await fetch('../lessons/365_day_calendar.json');
      if (!response.ok) throw new Error('Failed to load calendar data');
      const data = await response.json();
      const lessons = data.lessons || [];
      const todayLesson = this.findTodayLesson(lessons);
      const fallbackLesson = lessons.find((lesson) => lesson.has_dna && lesson.dna_file);
      const initialLesson = todayLesson?.has_dna ? todayLesson : fallbackLesson || todayLesson || null;

      this.stateManager.setState({
        calendarLessons: lessons,
        todayLesson: todayLesson || fallbackLesson || null,
        selectedLesson: initialLesson,
        selectedDay: initialLesson?.day ?? null,
        streak: initialLesson?.day ?? 0,
      });
      this.refreshHistoryStreak();
    } catch (error) {
      console.error(error);
      this.showCalendarError();
    }
  }

  async loadLessonDNA(lessonSummary) {
    if (!lessonSummary?.dna_file) {
      this.stateManager.setState({ lessonData: null });
      return;
    }
    try {
      const response = await fetch(`../lessons/${lessonSummary.dna_file}-dna.json`);
      if (!response.ok) throw new Error(`Unable to load DNA for ${lessonSummary.dna_file}`);
      const lessonData = await response.json();
      this.stateManager.setState({
        lessonData,
        currentPhase: 'welcome',
      });
      await this.establishSession(lessonSummary);
      this.elements.sessionStatus.textContent = `Synced "${lessonSummary.title}" DNA`;
    } catch (error) {
      console.error(error);
      this.elements.sessionStatus.textContent = 'DNA missing for this lesson';
      this.stateManager.setState({ lessonData: null });
    }
  }

  updateDateLabel() {
    const formatter = new Intl.DateTimeFormat('en-US', {
      weekday: 'long',
      month: 'short',
      day: 'numeric',
    });
    if (this.elements.calendarDate) {
      this.elements.calendarDate.textContent = formatter.format(new Date());
    }
  }

  updateAgeDisplay(value) {
    if (this.elements.ageValue) {
      this.elements.ageValue.textContent = value;
    }
  }

  highlightBucket(bucketId) {
    this.elements.ageBuckets?.forEach((bucket) => {
      bucket.classList.toggle('active', bucket.dataset.age === bucketId);
    });
  }

  updateLessonMetaFromVariant(state) {
    const variant = this.getVariant(state);
    const languagePack = this.getVariantLanguage(state, variant);
    if (!variant) return;

    const title =
      languagePack?.title ||
      state.selectedLesson?.title ||
      variant.title ||
      'Kelly is preparing the lesson';
    this.elements.lessonTitle.textContent = title;
    this.elements.lessonDayChip.textContent = `Day ${state.selectedLesson?.day ?? '--'}`;

    const description =
      languagePack?.mainContent ||
      variant.description ||
      state.selectedLesson?.learning_essence ||
      state.selectedLesson?.learning_objective ||
      'Lesson essence will appear once synced.';
    this.elements.topicDescription.textContent = description;

    const objectives =
      variant.objectives?.length
        ? variant.objectives
        : state.selectedLesson?.learning_objective
        ? [state.selectedLesson.learning_objective]
        : [];
    this.elements.learningObjectives.innerHTML = '';
    if (!objectives.length) {
      const placeholder = document.createElement('li');
      placeholder.textContent = 'Objective will appear once synced.';
      this.elements.learningObjectives.appendChild(placeholder);
    } else {
      objectives.forEach((objective) => {
        const item = document.createElement('li');
        item.textContent = this.formatText(objective);
        this.elements.learningObjectives.appendChild(item);
      });
    }

    const duration = state.lessonData?.metadata?.duration || state.selectedLesson?.duration;
    this.elements.lessonDuration.textContent = this.formatDuration(duration);
    const tags = state.lessonData?.metadata?.tags?.length
      ? state.lessonData.metadata.tags
      : state.selectedLesson?.tags;
    this.populateTags(this.elements.lessonTags, tags);
  }

  renderPhase(state) {
    const normalized = this.normalizePhase(state.currentPhase);
    this.elements.phasePill.textContent = this.formatText(normalized);
    this.unityBridge.emit('phase-progress', {
      phase: normalized,
      sessionId: state.sessionId,
      lessonId: state.sessionLessonId,
    });

    if (!state.lessonData) {
      this.elements.questionText.textContent = 'Kelly is syncing today’s DNA...';
      this.setAudioScript('Loading...');
      this.hideChoices();
      return;
    }

    if (normalized === 'welcome') {
      this.renderWelcomePhase(state);
    } else if (normalized === 'wisdom') {
      this.renderWisdomPhase(state);
      this.completeSessionIfNeeded();
    } else {
      this.renderQuestionPhase(state, normalized);
    }
  }

  renderWelcomePhase(state) {
    const variant = this.getVariant(state);
    const languagePack = this.getVariantLanguage(state, variant);
    const welcomeText =
      languagePack?.welcome ||
      variant?.welcome ||
      `Kelly is welcoming you to ${state.selectedLesson?.title ?? 'today’s lesson'}.`;
    this.elements.questionText.textContent = welcomeText;
    const scriptText = languagePack?.mainContent || variant?.script || welcomeText;
    this.setAudioScript(`Kelly: ${scriptText}`);
    this.hideChoices();
  }

  renderWisdomPhase(state) {
    const variant = this.getVariant(state);
    const languagePack = this.getVariantLanguage(state, variant);
    const wisdomText =
      languagePack?.wisdomMoment ||
      variant?.wisdomMoment ||
      'You completed today’s lesson. Amazing work!';

    const container = document.createElement('div');
    container.className = 'wisdom-block';
    const message = document.createElement('div');
    message.className = 'wisdom-message';
    message.textContent = wisdomText;
    const action = document.createElement('button');
    action.className = 'primary-btn wisdom-action';
    action.textContent = 'Continue learning';
    action.addEventListener('click', () => this.scrollToLesson());
    container.appendChild(message);
    container.appendChild(action);

    this.elements.questionText.innerHTML = '';
    this.elements.questionText.appendChild(container);
    this.setAudioScript(`Kelly: ${wisdomText}`);
    this.hideChoices();
  }

  renderQuestionPhase(state, phase) {
    const interaction = this.findInteraction(state.lessonData, phase);
    if (!interaction) {
      this.elements.questionText.textContent = 'Kelly is composing the next prompt.';
      this.hideChoices();
      return;
    }
    const adaptedInteraction = this.applyAgeAdaptation(interaction, state.ageBucket);
    const questionText = this.formatText(adaptedInteraction.question || interaction.question);
    this.elements.questionText.textContent = questionText;

    const variant = this.getVariant(state);
    const languagePack = this.getVariantLanguage(state, variant);
    const prompt =
      languagePack?.interactionPrompts?.[0] ||
      languagePack?.interactionPrompts?.[1] ||
      'Share your thoughts when you are ready.';
    this.setAudioScript(`Kelly: ${prompt}`);
    this.renderChoiceButtons(adaptedInteraction.choices || interaction.choices || []);
  }

  renderChoiceButtons(choices) {
    if (!this.elements.choiceContainer) return;
    if (!choices.length) {
      this.hideChoices();
      return;
    }
    this.elements.choiceContainer.classList.remove('hidden');
    this.elements.choiceContainer.innerHTML = '';
    choices.forEach((choice) => {
      const button = document.createElement('button');
      button.className = 'choice-card glass-panel-light hover-lift';
      button.innerHTML = `
        <span class="choice-label">${this.formatText(choice.text)}</span>
        <p>${this.formatText(choice.response || 'Kelly will share feedback')}</p>
      `;
      button.addEventListener('click', () => this.handleChoiceSelection(button, choice));
      this.elements.choiceContainer.appendChild(button);
    });
  }

  handleChoiceSelection(button, choice) {
    this.elements.choiceContainer
      ?.querySelectorAll('.choice-card')
      .forEach((card) => card.classList.toggle('selected', card === button));

    const responseText = this.formatText(
      choice.response || 'Wonderful insight! Let me take you deeper.'
    );
    this.setAudioScript(`Kelly: ${responseText}`);
    const nextPhase = this.normalizePhase(choice.nextStep) || 'wisdom';
    const currentPhase = this.stateManager.getState().currentPhase;
    this.unityBridge.emit('choice-selected', {
      choiceId: choice.id || choice.text || 'choice',
      currentPhase,
      nextPhase,
      sessionId: this.stateManager.getState().sessionId,
    });
    this.syncPhaseProgress(nextPhase, { completedPhase: currentPhase });
    setTimeout(() => {
      this.stateManager.setState({ currentPhase: nextPhase });
    }, 700);
  }

  hideChoices() {
    if (!this.elements.choiceContainer) return;
    this.elements.choiceContainer.classList.add('hidden');
    this.elements.choiceContainer.innerHTML = '';
  }

  setAudioScript(text) {
    if (this.elements.audioScript) {
      this.elements.audioScript.textContent = text;
    }
  }

  renderTodayCard(lesson) {
    if (!lesson) return;
    this.elements.todayLessonTitle.textContent = lesson.title || 'Lesson title unavailable';
    this.elements.todayLessonObjective.textContent =
      lesson.learning_objective ||
      lesson.learning_essence ||
      'Kelly will surface your focus once the calendar syncs.';
    this.elements.todayLessonDay.textContent = `Day ${lesson.day ?? '--'}`;
    this.elements.todayLessonDuration.textContent = this.formatDuration(lesson.duration);
    this.populateTags(this.elements.todayLessonTags, lesson.tags);
  }

  renderLessonOverviewFromSummary(lesson) {
    if (!lesson) return;
    this.elements.lessonTitle.textContent = lesson.title || 'Lesson title unavailable';
    this.elements.lessonDayChip.textContent = `Day ${lesson.day ?? '--'}`;
    this.elements.topicDescription.textContent =
      lesson.learning_essence ||
      lesson.learning_objective ||
      'Lesson essence will flow in once the session syncs.';
    this.elements.lessonDuration.textContent = this.formatDuration(lesson.duration);
    this.populateTags(this.elements.lessonTags, lesson.tags);

    this.elements.learningObjectives.innerHTML = '';
    const objective = lesson.learning_objective || 'Kelly will provide guidance for this lesson.';
    const listItem = document.createElement('li');
    listItem.textContent = objective;
    this.elements.learningObjectives.appendChild(listItem);

    this.elements.questionText.textContent = `What stands out to you about "${lesson.title}" today?`;
    this.setAudioScript(`Kelly: I'm ready to guide you through ${lesson.title}.`);
    this.elements.sessionStatus.textContent = `Ready for ${lesson.title}`;
  }

  renderWeekStrip() {
    if (!this.elements.weekStrip) return;
    const state = this.stateManager.getState();
    const referenceDate = state.selectedLesson
      ? this.getLessonDate(state.selectedLesson)
      : new Date();
    const startOfWeek = new Date(referenceDate);
    const dayOfWeek = startOfWeek.getDay();
    const diff = (dayOfWeek + 6) % 7;
    startOfWeek.setDate(referenceDate.getDate() - diff);

    this.elements.weekStrip.innerHTML = '';
    for (let i = 0; i < 7; i++) {
      const current = new Date(startOfWeek);
      current.setDate(startOfWeek.getDate() + i);
      const lesson = this.findLessonByDate(current, state.calendarLessons);
      const weekCard = document.createElement('button');
      weekCard.className = 'week-card';
      if (lesson?.has_dna) weekCard.classList.add('has-dna');
      if (lesson?.day === state.selectedDay) weekCard.classList.add('selected');
      weekCard.dataset.day = lesson?.day ?? '';
      weekCard.innerHTML = `
        <span class="meta-label subtle">${current.toLocaleDateString('en-US', { weekday: 'short' })}</span>
        <strong>${current.getDate()}</strong>
        <span>${lesson?.has_dna ? '🧬' : '—'}</span>
      `;
      if (lesson?.day) {
        weekCard.addEventListener('click', () => this.selectLessonByDay(lesson.day));
      } else {
        weekCard.disabled = true;
      }
      this.elements.weekStrip.appendChild(weekCard);
    }
  }

  renderMonthGrid() {
    if (!this.elements.monthGrid) return;
    const state = this.stateManager.getState();
    const today = new Date();
    const viewDate = new Date(today.getFullYear(), today.getMonth() + state.monthOffset, 1);
    const month = viewDate.getMonth();
    const year = viewDate.getFullYear();

    this.elements.currentMonthLabel.textContent = viewDate.toLocaleDateString('en-US', {
      month: 'long',
      year: 'numeric',
    });

    const daysInMonth = new Date(year, month + 1, 0).getDate();
    this.elements.monthGrid.innerHTML = '';

    for (let day = 1; day <= daysInMonth; day++) {
      const cell = document.createElement('button');
      cell.className = 'day-cell';
      cell.textContent = day;

      const lesson = state.calendarLessons.find((item) => {
        if (!item?.date) return false;
        const lessonDate = this.getLessonDate(item, year);
        return lessonDate.getMonth() === month && lessonDate.getDate() === day;
      });

      if (lesson?.has_dna) cell.classList.add('has-dna');
      const isToday =
        day === today.getDate() && month === today.getMonth() && year === today.getFullYear();
      if (isToday) cell.classList.add('today');
      if (lesson?.day === state.selectedDay) cell.classList.add('selected');

      if (lesson?.day) {
        cell.dataset.day = lesson.day;
        cell.addEventListener('click', () => this.selectLessonByDay(lesson.day));
      } else {
        cell.disabled = true;
      }

      this.elements.monthGrid.appendChild(cell);
    }
  }

  highlightSelectedCells() {
    const state = this.stateManager.getState();
    const selectedDay = state.selectedDay;
    if (!selectedDay) return;

    this.elements.monthGrid?.querySelectorAll('.day-cell').forEach((cell) => {
      cell.classList.toggle('selected', Number(cell.dataset.day) === selectedDay);
    });

    this.elements.weekStrip?.querySelectorAll('.week-card').forEach((card) => {
      card.classList.toggle('selected', Number(card.dataset.day) === selectedDay);
    });
  }

  selectLessonByDay(day) {
    const state = this.stateManager.getState();
    const lesson = state.calendarLessons.find((item) => item.day === day);
    if (!lesson) return;
    this.stateManager.setState({
      selectedLesson: lesson,
      selectedDay: day,
      currentPhase: 'welcome',
    });
  }

  changeView(view) {
    if (!view) return;
    const state = this.stateManager.getState();
    if (view === state.currentView) return;
    this.stateManager.setState({ currentView: view });
  }

  updateViewVisibility(view) {
    this.elements.viewPills.forEach((pill) => {
      pill.classList.toggle('active', pill.dataset.view === view);
    });
    this.elements.calendarViews.forEach((panel) => {
      panel.classList.toggle('active', panel.dataset.viewTarget === view);
    });
  }

  shiftMonth(direction) {
    const state = this.stateManager.getState();
    this.stateManager.setState({ monthOffset: state.monthOffset + direction });
  }

  togglePlay() {
    const state = this.stateManager.getState();
    const nextState = !state.isPlaying;
    this.stateManager.setState({ isPlaying: nextState });
    this.elements.playIcon.textContent = nextState ? '❚❚' : '▶';
    this.setAudioScript(
      nextState ? 'Kelly is narrating…' : 'Kelly paused. Tap play to resume.'
    );
  }

  scrollToLesson() {
    document.getElementById('question-card')?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    });
  }

  scrollToTodayLesson() {
    const state = this.stateManager.getState();
    if (state.todayLesson?.day) {
      this.selectLessonByDay(state.todayLesson.day);
    }
    this.scrollToLesson();
  }

  populateTags(container, tags = []) {
    if (!container) return;
    container.innerHTML = '';
    if (!tags?.length) {
      const placeholder = document.createElement('span');
      placeholder.className = 'tag-pill tag-pill-placeholder';
      placeholder.textContent = 'Awaiting tags';
      container.appendChild(placeholder);
      return;
    }
    tags.forEach((tag) => {
      const pill = document.createElement('span');
      pill.className = 'tag-pill';
      pill.textContent = this.formatText(tag);
      container.appendChild(pill);
    });
  }

  formatDuration(duration) {
    if (!duration) return '-- min';
    if (duration.min && duration.max) {
      return `${duration.min}-${duration.max} min`;
    }
    return duration.min ? `${duration.min} min` : '-- min';
  }

  findLessonByDate(date, lessons = []) {
    return lessons.find((lesson) => {
      const lessonDate = this.getLessonDate(lesson, date.getFullYear());
      return (
        lessonDate.getDate() === date.getDate() && lessonDate.getMonth() === date.getMonth()
      );
    });
  }

  getLessonDate(lesson, year = new Date().getFullYear()) {
    if (!lesson?.date) return new Date(year, 0, 1);
    return new Date(`${lesson.date} ${year}`);
  }

  findTodayLesson(lessons) {
    const today = new Date();
    return lessons.find((lesson) => {
      const lessonDate = new Date(`${lesson.date}, ${today.getFullYear()}`);
      return (
        lessonDate.getDate() === today.getDate() && lessonDate.getMonth() === today.getMonth()
      );
    });
  }

  updateStreakPlaceholder() {
    const state = this.stateManager.getState();
    const streak = state.streak || 0;
    const label = streak === 1 ? 'day' : 'days';
    this.elements.streakCount.textContent = `${streak} ${label}`;
  }

  showCalendarError() {
    this.elements.todayLessonTitle.textContent = 'Unable to load calendar';
    this.elements.todayLessonObjective.textContent = 'Please ensure the local server is running.';
  }

  getBucketForAge(value) {
    const match = Object.entries(this.bucketRanges).find(
      ([, range]) => value >= range[0] && value <= range[1]
    );
    return match ? match[0] : '18-35';
  }

  normalizePhase(step) {
    if (!step) return 'welcome';
    const key = String(step).toLowerCase();
    return PHASE_ALIASES[key] || key;
  }

  getVariant(state) {
    return state.lessonData?.ageVariants?.[state.ageBucket] || null;
  }

  getVariantLanguage(state, variant = null) {
    const source = variant || this.getVariant(state);
    return (
      source?.language?.[state.language] ||
      source?.language?.en ||
      source?.language?.es ||
      null
    );
  }

  findInteraction(lessonData, phase) {
    if (!lessonData?.interactions) return null;
    return (
      lessonData.interactions.find((interaction) => {
        const step = this.normalizePhase(interaction.step || interaction.phase);
        return step === phase;
      }) || null
    );
  }

  applyAgeAdaptation(interaction, ageBucket) {
    if (!interaction?.ageAdaptations?.[ageBucket]) return interaction;
    const adaptation = interaction.ageAdaptations[ageBucket];
    return {
      ...interaction,
      ...adaptation,
      choices: adaptation.choices?.length ? adaptation.choices : interaction.choices,
    };
  }

  formatText(text) {
    if (!text) return '';
    if (typeof text !== 'string') return text;
    const spaced = text.replace(/_/g, ' ');
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
  }

  getLessonBackendId(lessonSummary) {
    if (!lessonSummary) return null;
    return lessonSummary.dna_file || lessonSummary.lesson_id || lessonSummary.id || null;
  }

  async establishSession(lessonSummary) {
    const state = this.stateManager.getState();
    const lessonId = this.getLessonBackendId(lessonSummary);
    if (!lessonId) return;

    const stored = this.sessionClient.getStoredSession();
    const currentSessionId = state.sessionId || stored?.sessionId;

    if (currentSessionId && (state.sessionLessonId === lessonId || stored?.lessonId === lessonId)) {
      const resumed = await this.sessionClient.getSession(currentSessionId);
      if (resumed) {
        this.stateManager.setState({
          sessionId: resumed.sessionId,
          sessionLessonId: lessonId,
          sessionState: resumed.state,
          sessionCompleted: resumed.state?.isCompleted ?? false,
          currentPhase: resumed.progress?.currentPhase || state.currentPhase,
        });
        this.elements.sessionStatus.textContent = resumed.state?.isCompleted
          ? 'Previous session completed'
          : 'Session resumed';
        this.unityBridge.emit('session-start', {
          mode: 'resume',
          sessionId: resumed.sessionId,
          lessonId,
          phase: resumed.progress?.currentPhase || 'welcome',
        });
        await this.refreshHistoryStreak();
        return;
      }
    }

    const session = await this.sessionClient.startSession(state.age, lessonId);
    if (session) {
      this.stateManager.setState({
        sessionId: session.sessionId,
        sessionLessonId: lessonId,
        sessionState: session.state,
        sessionCompleted: false,
        currentPhase: session.progress?.currentPhase || 'welcome',
      });
      this.elements.sessionStatus.textContent = `Session started • ${lessonSummary.title}`;
      this.unityBridge.emit('session-start', {
        mode: 'new',
        sessionId: session.sessionId,
        lessonId,
        phase: session.progress?.currentPhase || 'welcome',
      });
    } else {
      this.elements.sessionStatus.textContent = 'Offline mode: unable to reach session service.';
    }
  }

  async syncPhaseProgress(nextPhase, options = {}) {
    const state = this.stateManager.getState();
    if (!state.sessionId) return;
    const payload = {
      currentPhase: nextPhase,
    };
    if (options.completedPhase) {
      payload.completedPhase = options.completedPhase;
    }
    if (options.interactionCompleted) {
      payload.interactionCompleted = options.interactionCompleted;
    }
    await this.sessionClient.updateProgress(state.sessionId, payload);
    this.unityBridge.emit('phase-progress', {
      phase: nextPhase,
      sessionId: state.sessionId,
      completedPhase: options.completedPhase,
    });
  }

  async completeSessionIfNeeded() {
    const state = this.stateManager.getState();
    if (!state.sessionId || state.sessionCompleted) return;
    const result = await this.sessionClient.completeSession(state.sessionId);
    if (result) {
      this.stateManager.setState({
        sessionCompleted: true,
        sessionState: result.state,
        sessionId: null,
        sessionLessonId: null,
      });
      this.elements.sessionStatus.textContent = 'Lesson completed! Great job!';
      this.unityBridge.emit('session-complete', {
        lessonId: result.lessonId,
        durationMin: result.durationMin,
      });
      await this.refreshHistoryStreak();
    }
  }

  async refreshHistoryStreak() {
    const history = await this.sessionClient.fetchHistory();
    const streak = this.calculateHistoryStreak(history);
    this.stateManager.setState({ streak });
  }

  calculateHistoryStreak(history = []) {
    if (!history.length) return 0;
    const daySet = new Set(
      history.map((entry) => new Date(entry.completedAt).toISOString().slice(0, 10))
    );
    const cursor = new Date();
    let streak = 0;

    while (true) {
      const key = cursor.toISOString().slice(0, 10);
      if (daySet.has(key)) {
        streak += 1;
        cursor.setDate(cursor.getDate() - 1);
      } else {
        break;
      }
    }
    return streak;
  }

  showUnityStream(isActive) {
    if (this.elements.unityIframe) {
      this.elements.unityIframe.classList.toggle('active', Boolean(isActive));
    }
    this.elements.unityOverlay?.classList.toggle('hidden', Boolean(isActive));
    this.elements.unityRibbon?.classList.toggle('hidden', Boolean(isActive));
  }

  setUnityStatus(text) {
    if (this.elements.unityStatusLabel) {
      this.elements.unityStatusLabel.textContent = text || 'Awaiting bridge handshake…';
    }
  }

  setUnityTelemetry(telemetry = {}) {
    if (!telemetry || (!telemetry.fps && !telemetry.pose && !telemetry.latency)) {
      return;
    }
    const fps = telemetry.fps ?? telemetry.frameRate;
    const pose = telemetry.pose || telemetry.state;
    const latency = telemetry.latency ?? telemetry.ms;
    const chunks = [];
    if (typeof fps !== 'undefined') {
      chunks.push(`${Math.round(fps)} fps`);
    }
    if (typeof latency !== 'undefined') {
      chunks.push(`${latency} ms`);
    }
    if (pose) {
      chunks.push(pose);
    }
    if (chunks.length) {
      this.setUnityStatus(`Streaming ${chunks.join(' • ')}`);
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const app = new UnifiedLessonApp();
  app.init();
});

