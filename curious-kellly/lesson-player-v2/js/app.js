/**
 * Curious Kelly OS - "No UI" Operating System
 * Manages Modes: Attract, Dashboard, Lesson, and Apps (Modals)
 */
'use strict';

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm';

// Configuration - Supabase credentials (from public/config.js)
const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';
// Set to your backend URL (e.g. from Railway) or default for dev
const API_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:3000/api' 
    : 'https://curiouskelly-production.up.railway.app/api'; // Connected to Railway Production

class KellyOS {
    constructor() {
        this.supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
        
        // State
        this.state = {
            mode: 'dashboard', 
            user: null,
            age: 25,
            ageBucket: '18-35',
            language: 'en',
            lessonPhase: 'welcome',
            isPlaying: false,
            unityReady: false,
            currentLesson: null,
            lessonId: null,
            activeModal: null,
            syllabusData: null, // Cache for the full calendar
            // Earn to Learn state
            earnings: {
                referralCode: null,
                commissionTier: 'new_learner',
                commissionRate: 0.10,
                pendingEarnings: 0,
                availableEarnings: 0,
                lifetimeEarnings: 0,
                totalReferrals: 0
            }
        };

        // DOM Cache
        this.dom = {
            // Layers
            backgroundLayer: document.getElementById('layer-background'),
            interfaceLayer: document.getElementById('layer-os-interface'),
            modalLayer: document.getElementById('layer-modal-stack'),
            
            // Nav & Triggers
            osTrigger: document.getElementById('btn-menu-drawer'), // The Single Hamburger
            drawer: document.getElementById('menu-drawer'),
            mobileTabs: document.querySelectorAll('.tab-item'),
            
            // Modes
            modes: {
                dashboard: document.getElementById('mode-dashboard'), // Now mostly hidden/virtual
                lesson: document.getElementById('mode-lesson')
            },
            
            // Modals
            modals: {
                tuition: document.getElementById('modal-tuition'),
                reader: document.getElementById('modal-reader')
            },
            
            // Drawer Elements (The "Bam" Content)
            drawerTitle: document.getElementById('drawer-lesson-title'),
            drawerDate: document.getElementById('drawer-date'),
            drawerStartBtn: document.getElementById('btn-drawer-start'),
            
            // Lesson Elements
            audio: document.getElementById('kelly-audio'),
            questionText: document.getElementById('question-text'),
            choiceContainer: document.getElementById('choice-cards-container'),
            playButton: document.getElementById('play-pause'),
            
            // Unity
            unityIframe: document.getElementById('kelly-unity-iframe'),
            unityContainer: document.getElementById('kelly-unity-container'),
            unityStatus: document.getElementById('kelly-unity-status'),
            kellyImage: document.getElementById('kelly-image'),
            
            // Controls
            ageSlider: document.getElementById('age-slider'),
            ageValue: document.getElementById('age-value'),
            ageBuckets: document.querySelectorAll('.age-bucket-floating'),
            languageSelector: document.getElementById('language-selector')
        };

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupUnity();
        this.checkSession();
        
        // Initial State: Clean Desktop (Dashboard logic runs but UI is hidden)
        this.switchMode('dashboard');
        
        // Pre-fetch data
        this.fetchDailyLesson();
        
        // Bind Checkout Buttons
        this.bindCheckoutButtons();
        
        // Earn to Learn: Capture referral and load earnings
        this.captureReferral();
        this.setupShareListeners();
    }

    // --- Mode Management ---

    switchMode(newMode) {
        if (this.state.mode === newMode && newMode !== 'lesson') return;
        console.log(`Switching OS Mode: ${this.state.mode} -> ${newMode}`);
        
        this.state.mode = newMode;
        
        // Toggle Body Class for UI Visibility
        if (newMode === 'lesson') {
            document.body.classList.add('ui-active');
            this.sendToUnity('kelly-focus-user', {});
            if (!this.state.currentLesson) this.fetchDailyLesson();
        } else {
            document.body.classList.remove('ui-active');
            this.sendToUnity('kelly-idle', {});
        }

        // Update UI Panels
        Object.entries(this.dom.modes).forEach(([key, el]) => {
            if (el) {
                if (key === newMode) el.classList.add('active');
                else el.classList.remove('active');
            }
        });
        
        // Close Drawer on mode switch if entering lesson
        if (newMode === 'lesson') this.toggleDrawer(false);
    }

    // --- Modal & Drawer Management ---

    toggleDrawer(forceState) {
        const isOpen = this.dom.drawer.classList.contains('open');
        const newState = forceState !== undefined ? forceState : !isOpen;
        
        if (newState) {
            this.dom.drawer.classList.add('open');
            // document.body.classList.add('ui-active'); 
        } else {
            this.dom.drawer.classList.remove('open');
            if (this.state.mode !== 'lesson') {
                document.body.classList.remove('ui-active');
            }
        }
    }

    openModal(modalId, content = null) {
        const modal = this.dom.modals[modalId];
        if (!modal) return;

        this.toggleDrawer(false);

        if (content && modalId === 'reader') {
            document.getElementById('reader-title').textContent = content.title;
            document.getElementById('reader-content').innerHTML = content.body;
        }

        modal.classList.add('open');
        this.state.activeModal = modalId;
    }

    closeAllModals() {
        Object.values(this.dom.modals).forEach(el => el.classList.remove('open'));
        this.state.activeModal = null;
    }

    // --- "Apps" Implementation ---

    openContentModal(type, title, bodyText) {
        let richContent = '';
        
        if (type === 'Careers') {
            richContent = `
                <div class="reader-body">
                    <p class="lead">Join our mission to democratize high-quality education through AI.</p>
                    <div class="job-list" style="margin-top:30px;">
                        <div class="job-item" style="border:1px solid rgba(255,255,255,0.1); padding:20px; border-radius:12px; margin-bottom:15px;">
                            <h4 style="margin:0 0 10px 0">Senior Full Stack Engineer</h4>
                            <p style="color:#aaa; font-size:0.9em;">Remote • Full Time</p>
                            <button class="btn-primary-glass" style="margin-top:10px; width:auto;">Apply</button>
                        </div>
                    </div>
                </div>`;
        } else if (type === 'Syllabus') {
            // Use a CSS Grid similar to marketing page
            richContent = `
                <div class="reader-body">
                     <p class="lead">Your 365-day journey. Interactive and adaptive.</p>
                     
                     <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 15px; margin-top: 30px;">
                        ${this.generateMockCalendarGrid()}
                     </div>
                </div>`;
        } else {
            richContent = `<p>${bodyText}</p>`;
        }

        this.openModal('reader', { title, body: richContent });
    }

    generateMockCalendarGrid() {
        // Generate a grid that looks like the marketing calendar
        const topics = [
            { d: 1, t: "The Sun", c: "CORE" },
            { d: 2, t: "Photosynthesis", c: "" },
            { d: 3, t: "Water Cycle", c: "" },
            { d: 4, t: "Gravity", c: "CORE" },
            { d: 5, t: "Motion", c: "" },
            { d: 6, t: "Elements", c: "" },
            { d: 7, t: "Atoms", c: "CORE" },
            { d: 8, t: "Molecules", c: "" },
            { d: 9, t: "Cells", c: "" },
            { d: 10, t: "DNA", c: "CORE" },
            { d: 11, t: "Evolution", c: "" },
            { d: 12, t: "Ecosystems", c: "" }
        ];
        
        return topics.map(t => `
            <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 15px; border-radius: 12px; display: flex; flex-direction: column; gap: 8px;">
                <div style="font-size: 0.7em; color: #aaa; text-transform: uppercase; display: flex; justify-content: space-between;">
                    <span>Day ${t.d}</span>
                    ${t.c ? '<span style="color:#d97757; border:1px solid #d97757; padding:0 4px; border-radius:4px;">CORE</span>' : ''}
                </div>
                <div style="font-weight: 500; font-size: 0.95em;">${t.t}</div>
            </div>
        `).join('');
    }

    // --- Checkout Logic ---

    bindCheckoutButtons() {
        document.addEventListener('click', (e) => {
            if (e.target.closest('.btn-checkout')) {
                const btn = e.target.closest('.btn-checkout');
                const plan = btn.dataset.plan;
                this.handleCheckout(btn, plan);
            }
        });
    }

    async handleCheckout(btn, plan) {
        // 1. Auth Check (Simple)
        if (!this.state.user && plan !== 'gift') {
             // Prompt login logic would go here, for now we proceed or use dummy
        }

        const originalText = btn.innerText;
        btn.innerText = "Processing...";
        btn.disabled = true;

        try {
            // 2. Prepare Payload
            const payload = {
                plan,
                customerEmail: this.state.user?.email || 'guest@example.com', // Fallback for guests/gifts
                // Optional fields for Gifts
                recipientEmail: plan === 'gift' ? 'recipient@example.com' : undefined, // UI needs input for this
                gifterName: this.state.user?.user_metadata?.full_name || 'A Friend'
            };

            // 3. Call Backend
            const response = await fetch(`${API_URL}/checkout/create-session`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) throw new Error(data.error || 'Checkout failed');

            // 4. Redirect to Stripe
            if (data.url) {
                window.location.href = data.url;
            } else {
                throw new Error("No checkout URL returned");
            }

        } catch (error) {
            console.error('Checkout Error:', error);
            btn.innerText = "Error - Try Again";
            btn.style.background = "#ef4444";
            
            setTimeout(() => {
                btn.innerText = originalText;
                btn.disabled = false;
                btn.style.background = ""; 
            }, 3000);
            
            alert(`Checkout failed: ${error.message}. (Backend might not be active yet)`);
        }
    }

    // Deprecated: Mock function, kept for reference or fallbacks
    handleTuitionSelectMock(planName) {
        // ... old mock logic ...
    }

    openSettingsModal() {
        const currentArchetype = this.state.currentArchetype || 'The Explorer';
        const archetypeInfo = this.getArchetypeInfo(currentArchetype);
        const manualChoice = localStorage.getItem('kelly_archetype');
        
        // Build archetype options
        const archetypeOptions = this.getAllArchetypes().map(arch => {
            const info = this.getArchetypeInfo(arch);
            const isSelected = arch === currentArchetype;
            return `<option value="${arch}" ${isSelected ? 'selected' : ''}>${info.emoji} ${arch.replace('The ', '')} - ${info.desc}</option>`;
        }).join('');
        
        const html = `
            <div class="settings-panel" style="display: flex; flex-direction: column; gap: 30px;">
                
                <!-- Age / Persona -->
                <div class="setting-group">
                    <label style="display: block; margin-bottom: 15px; font-weight: 500; color: #ccc;">Your Age</label>
                    <div class="age-selector-modal" style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 16px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span id="modal-age-label">${archetypeInfo.emoji} ${currentArchetype}</span>
                            <span id="modal-age-val" style="color: #d97757;">${this.state.age}</span>
                        </div>
                        <input type="range" min="2" max="102" value="${this.state.age}" style="width: 100%;" id="modal-age-slider">
                        <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #666; margin-top: 5px;">
                            <span>Toddler</span>
                            <span>Elder</span>
                        </div>
                    </div>
                </div>
                
                <!-- Learning Style (Archetype) -->
                <div class="setting-group">
                    <label style="display: block; margin-bottom: 15px; font-weight: 500; color: #ccc;">Learning Style</label>
                    <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 16px;">
                        <p style="font-size: 0.85rem; color: #888; margin-bottom: 12px;">
                            ${manualChoice ? 'You\'ve chosen your style. Clear to use age-based defaults.' : 'Auto-selected based on your age. Choose manually to override.'}
                        </p>
                        <select id="archetype-selector" style="width: 100%; padding: 12px; border-radius: 8px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15); color: #fff; font-size: 1rem; cursor: pointer;">
                            <option value="">🎯 Auto (based on age)</option>
                            ${archetypeOptions}
                        </select>
                        <div id="archetype-preview" style="margin-top: 12px; padding: 12px; background: rgba(255,255,255,0.03); border-radius: 8px; display: ${manualChoice ? 'block' : 'none'};">
                            <div style="font-size: 0.9rem; color: #ccc;">${archetypeInfo.desc}</div>
                        </div>
                    </div>
                </div>

                <!-- Language -->
                <div class="setting-group">
                    <label style="display: block; margin-bottom: 15px; font-weight: 500; color: #ccc;">Language</label>
                    <div class="segmented-control" style="display: flex; background: rgba(255,255,255,0.05); border-radius: 12px; padding: 4px;">
                        <button class="seg-btn ${this.state.language === 'en' ? 'active' : ''}" data-lang="en" style="flex: 1; padding: 10px; border: none; background: ${this.state.language === 'en' ? 'rgba(255,255,255,0.1)' : 'transparent'}; color: #fff; border-radius: 8px; cursor: pointer;">English</button>
                        <button class="seg-btn ${this.state.language === 'es' ? 'active' : ''}" data-lang="es" style="flex: 1; padding: 10px; border: none; background: ${this.state.language === 'es' ? 'rgba(255,255,255,0.1)' : 'transparent'}; color: #fff; border-radius: 8px; cursor: pointer;">Español</button>
                        <button class="seg-btn ${this.state.language === 'fr' ? 'active' : ''}" data-lang="fr" style="flex: 1; padding: 10px; border: none; background: ${this.state.language === 'fr' ? 'rgba(255,255,255,0.1)' : 'transparent'}; color: #fff; border-radius: 8px; cursor: pointer;">Français</button>
                    </div>
                </div>

                <!-- Danger Zone -->
                <div class="setting-group" style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <button id="btn-reset-progress" class="btn-secondary-glass" style="width: 100%; color: #ef4444; border-color: rgba(239, 68, 68, 0.3);">Reset Progress</button>
                </div>
            </div>
        `;

        this.openModal('reader', { title: 'Settings', body: html });

        // Attach Listeners after render
        setTimeout(() => {
            const slider = document.getElementById('modal-age-slider');
            const valDisplay = document.getElementById('modal-age-val');
            const labelDisplay = document.getElementById('modal-age-label');
            const archetypeSelector = document.getElementById('archetype-selector');
            const archetypePreview = document.getElementById('archetype-preview');
            
            if (slider) {
                slider.addEventListener('input', (e) => {
                    const val = parseInt(e.target.value);
                    valDisplay.textContent = val;
                    
                    // Only auto-update archetype if no manual selection
                    if (!localStorage.getItem('kelly_archetype')) {
                        const archetype = this.getArchetypeForAge(val);
                        const info = this.getArchetypeInfo(archetype);
                        labelDisplay.textContent = `${info.emoji} ${archetype}`;
                        
                        // Update selector to show the auto-selected value
                        if (archetypeSelector) {
                            archetypeSelector.value = '';  // Show "Auto" selected
                        }
                    }
                    
                    // Update State
                    this.handleAgeChange(val);
                });
            }
            
            // Archetype manual selector
            if (archetypeSelector) {
                archetypeSelector.addEventListener('change', (e) => {
                    const selected = e.target.value;
                    
                    if (selected === '') {
                        // Clear manual choice, revert to age-based
                        localStorage.removeItem('kelly_archetype');
                        const autoArchetype = this.getArchetypeForAge(this.state.age);
                        const info = this.getArchetypeInfo(autoArchetype);
                        this.state.currentArchetype = autoArchetype;
                        labelDisplay.textContent = `${info.emoji} ${autoArchetype}`;
                        if (archetypePreview) archetypePreview.style.display = 'none';
                    } else {
                        // Store manual choice
                        localStorage.setItem('kelly_archetype', selected);
                        const info = this.getArchetypeInfo(selected);
                        this.state.currentArchetype = selected;
                        labelDisplay.textContent = `${info.emoji} ${selected}`;
                        if (archetypePreview) {
                            archetypePreview.innerHTML = `<div style="font-size: 0.9rem; color: #ccc;">${info.desc}</div>`;
                            archetypePreview.style.display = 'block';
                        }
                    }
                    
                    // Re-render lesson with new archetype
                    if (this.state.currentLesson) this.renderLessonState();
                    
                    console.log(`[Settings] Archetype changed to: ${this.state.currentArchetype}`);
                });
            }

            document.querySelectorAll('.seg-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    document.querySelectorAll('.seg-btn').forEach(b => {
                        b.classList.remove('active');
                        b.style.background = 'transparent';
                    });
                    e.target.classList.add('active');
                    e.target.style.background = 'rgba(255,255,255,0.1)';
                    
                    this.state.language = e.target.dataset.lang;
                    // Re-fetch or reload audio path
                    if (this.state.currentLesson) this.renderLessonState();
                });
            });

            document.getElementById('btn-reset-progress')?.addEventListener('click', () => {
                if(confirm('Are you sure? This will reset your streak and archetype choice.')) {
                    localStorage.removeItem('kelly_archetype');
                    alert('Progress reset.');
                    location.reload();
                }
            });
        }, 100);
    }

    // --- Data & Supabase ---

    async checkSession() {
        const { data: { session } } = await this.supabase.auth.getSession();
        if (session) {
            this.state.user = session.user;
            console.log('User logged in:', this.state.user.email);
            
            // Check if this is a new user (just signed up) with a referral
            this.linkReferralIfNeeded(session.user);
            
            // Load earnings data after session is confirmed
            this.loadEarningsData();
            
            // Hide attract mode, show dashboard
            document.getElementById('mode-attract')?.classList.remove('active');
            this.switchMode('dashboard');
        } else {
            // Show placeholder for non-logged in users
            this.loadEarningsData();
        }
        
        // Listen for auth state changes (e.g., after OAuth redirect)
        this.supabase.auth.onAuthStateChange((event, session) => {
            if (event === 'SIGNED_IN' && session) {
                this.state.user = session.user;
                console.log('Auth state change: SIGNED_IN', session.user.email);
                
                // Link referral on sign in (if they have a stored referrer)
                this.linkReferralIfNeeded(session.user);
                this.loadEarningsData();
                
                // Update UI
                document.getElementById('mode-attract')?.classList.remove('active');
                this.switchMode('dashboard');
            }
        });
    }

    /**
     * Login with Google OAuth
     */
    async loginWithGoogle() {
        try {
            const { error } = await this.supabase.auth.signInWithOAuth({
                provider: 'google',
                options: {
                    redirectTo: window.location.origin + window.location.pathname,
                    queryParams: {
                        access_type: 'offline',
                        prompt: 'consent'
                    }
                }
            });
            
            if (error) {
                console.error('Google login error:', error);
                alert('Login failed. Please try again.');
            }
        } catch (error) {
            console.error('Login error:', error);
            alert('Login failed. Please try again.');
        }
    }

    /**
     * Link referral code to new user if they signed up with a referral
     */
    async linkReferralIfNeeded(user) {
        // Check if user has a stored referral code
        const referrerCode = localStorage.getItem('kelly_referrer');
        if (!referrerCode) return;
        
        // Check if user already has a referrer (don't overwrite)
        const { data: userData, error } = await this.supabase
            .from('users')
            .select('referred_by_user_id')
            .eq('id', user.id)
            .single();
        
        if (error || userData?.referred_by_user_id) {
            // Either error or user already has referrer - skip
            return;
        }
        
        // Call the convert API to link the referral
        try {
            const response = await fetch(`${API_URL}/referral/convert`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    userId: user.id,
                    referralCode: referrerCode,
                    conversionType: 'signup'
                })
            });
            
            if (response.ok) {
                console.log('[Earn to Learn] Referral linked successfully!');
                // Keep the referrer code - it's lifetime attribution
            } else {
                console.warn('[Earn to Learn] Failed to link referral');
            }
        } catch (error) {
            console.warn('[Earn to Learn] Error linking referral:', error);
        }
    }

    /**
     * Fetch lesson from Supabase by day number
     * Uses ACTUAL schema: core_lessons + lesson_atoms (phase, archetype, content)
     */
    async fetchDailyLesson(dayNumber = 1) {
        try {
            console.log(`Fetching Day ${dayNumber} from Supabase...`);
            
            // 1. Get core lesson metadata
            const { data: lesson, error: lessonError } = await this.supabase
                .from('core_lessons')
                .select('*')
                .eq('day_number', dayNumber)
                .single();

            if (lessonError) throw lessonError;
            if (!lesson) throw new Error(`No lesson found for day ${dayNumber}`);

            console.log('Lesson loaded:', lesson.topic);

            // 2. Get atoms for this lesson (grouped by archetype and phase)
            const { data: atoms, error: atomsError } = await this.supabase
                .from('lesson_atoms')
                .select('id, phase, archetype, content')
                .eq('core_lesson_id', lesson.id);

            if (atomsError) throw atomsError;

            // 3. Organize atoms by archetype for easy lookup
            // Atom content structure: { script, options[], kellyPose, kellyEmotion, optionIntro, hintSystem }
            const atomsByArchetype = {};
            for (const atom of atoms || []) {
                const arch = atom.archetype;
                if (!atomsByArchetype[arch]) {
                    atomsByArchetype[arch] = {};
                }
                atomsByArchetype[arch][atom.phase] = atom.content;
            }

            // 4. Store structured lesson data
            this.state.currentLesson = {
                id: lesson.id,
                dayNumber: lesson.day_number,
                topic: lesson.topic,
                universalTruth: lesson.universal_truth,
                marketingHeadline: lesson.marketing_headline,
                atoms: atoms || [],
                atomsByArchetype
            };
            this.state.lessonId = lesson.id;
            this.state.currentArchetype = this.getArchetypeForAge(this.state.age); // Smart default (or user's saved choice)
            
            // 5. Update UI with lesson info
            this.populateDrawerData(lesson.topic, lesson.universal_truth);
            this.renderLessonState();

        } catch (e) {
            console.error("Failed to fetch lesson from Supabase:", e);
            // Fallback to show error state
            this.populateDrawerData("Loading...", "Unable to load lesson. Check connection.");
        }
    }

    /**
     * Map age to archetype (smart defaults)
     * Uses Primary Three for most ages, extended archetypes for specific profiles
     */
    getArchetypeForAge(age) {
        // Check if user has manually selected an archetype
        const manualChoice = localStorage.getItem('kelly_archetype');
        if (manualChoice && this.getAllArchetypes().includes(manualChoice)) {
            return manualChoice;
        }
        
        // Smart defaults based on age
        if (age <= 6) return 'The Storyteller';    // Little ones love stories
        if (age <= 12) return 'The Explorer';      // Kids love adventure
        if (age <= 18) return 'The Rebel';         // Teens respond to challenge
        if (age <= 25) return 'The Rebel';         // Young adults same
        if (age <= 40) return 'The Scientist';     // Adults want evidence
        if (age <= 55) return 'The Strategist';    // Mid-career wants edge
        if (age <= 70) return 'The Scientist';     // Experienced want depth
        return 'The Explorer';                      // Elders return to wonder
    }
    
    /**
     * Get all available archetypes
     */
    getAllArchetypes() {
        return [
            'The Explorer',    // Primary Three
            'The Scientist',
            'The Rebel',
            'The Architect',   // Extended Nine
            'The Diplomat',
            'The Empath',
            'The MacGyver',
            'The Mystic',
            'The Provider',
            'The Storyteller',
            'The Strategist',
            'The Survivor'
        ];
    }
    
    /**
     * Get archetype metadata for UI
     */
    getArchetypeInfo(archetype) {
        const info = {
            'The Explorer': { emoji: '🧭', desc: 'Wonder & Adventure' },
            'The Scientist': { emoji: '🔬', desc: 'Evidence & Proof' },
            'The Rebel': { emoji: '⚡', desc: 'Challenge & Edge' },
            'The Architect': { emoji: '🏛️', desc: 'Structure & Design' },
            'The Diplomat': { emoji: '🤝', desc: 'Connection & Harmony' },
            'The Empath': { emoji: '💗', desc: 'Feeling & Heart' },
            'The MacGyver': { emoji: '🔧', desc: 'Practical & Hands-On' },
            'The Mystic': { emoji: '✨', desc: 'Meaning & Wonder' },
            'The Provider': { emoji: '🛡️', desc: 'Care & Protection' },
            'The Storyteller': { emoji: '📖', desc: 'Narrative & Memory' },
            'The Strategist': { emoji: '🎯', desc: 'Planning & Winning' },
            'The Survivor': { emoji: '🏕️', desc: 'Resilience & Grit' }
        };
        return info[archetype] || { emoji: '🧭', desc: 'Learning Style' };
    }

    populateDrawerData(title, subtitle) {
        // Update the Drawer "Dashboard" Card
        if (this.dom.drawerTitle) this.dom.drawerTitle.textContent = title;
        
        // Update the Hidden Dashboard Card (if it ever shows)
        const elTitle = document.getElementById('dashboard-lesson-title');
        const elDesc = document.getElementById('dashboard-lesson-desc');
        if (elTitle) elTitle.textContent = title;
        if (elDesc) elDesc.textContent = subtitle;
        
        // Update Date
        const dateStr = new Date().toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
        if (this.dom.drawerDate) this.dom.drawerDate.textContent = dateStr;
    }

    // --- Lesson Logic (Using Supabase Atoms) ---

    renderLessonState() {
        if (!this.state.currentLesson) return;
        
        const lesson = this.state.currentLesson;
        const archetype = this.state.currentArchetype || 'The Explorer';
        
        const elTitle = document.getElementById('lesson-title');
        const elDesc = document.getElementById('topic-description');
        if (elTitle) elTitle.textContent = lesson.topic;
        if (elDesc) elDesc.textContent = lesson.universalTruth;
        
        this.renderPhase();
    }

    /**
     * Render current phase using atom content from Supabase
     * Phases: Hook -> Fact1 -> Fact2 -> Wisdom (or similar)
     */
    renderPhase() {
        const phase = this.state.lessonPhase;
        const archetype = this.state.currentArchetype || 'The Explorer';
        const lesson = this.state.currentLesson;
        
        if (!this.dom.choiceContainer || !lesson) return;
        
        this.dom.choiceContainer.innerHTML = '';
        
        // Map internal phases to database phases
        const phaseMap = {
            'welcome': null,  // No atom for welcome
            'Hook': 'Hook',
            'Fact1': 'Fact1', 
            'Fact2': 'Fact2',
            'Wisdom': 'Wisdom'
        };
        
        if (phase === 'welcome') {
            // Welcome phase - show intro
            if (this.dom.questionText) {
                this.dom.questionText.textContent = `Today: ${lesson.topic}`;
            }
            const btn = this.createButton("Let's Begin", () => {
                this.state.lessonPhase = 'Hook';
                this.renderPhase();
            });
            this.dom.choiceContainer.appendChild(btn);
            return;
        }
        
        // Get atom content for current phase/archetype
        const atomContent = lesson.atomsByArchetype?.[archetype]?.[phase];
        
        if (!atomContent) {
            console.warn(`No atom found for ${archetype}/${phase}, advancing...`);
            this.advancePhase();
            return;
        }
        
        // Render the atom content
        // Structure: { script, options[], kellyPose, kellyEmotion, optionIntro, hintSystem }
        if (this.dom.questionText) {
            this.dom.questionText.textContent = atomContent.script || 'Continue...';
        }
        
        // Render options as choice cards
        if (atomContent.options && atomContent.options.length > 0) {
            // Show option intro if available
            if (atomContent.optionIntro) {
                const intro = document.createElement('div');
                intro.className = 'option-intro';
                intro.textContent = atomContent.optionIntro;
                intro.style.cssText = 'color: #71717a; font-size: 0.9rem; margin-bottom: 12px;';
                this.dom.choiceContainer.appendChild(intro);
            }
            
            atomContent.options.forEach(opt => {
                const btn = this.createButton(opt.text, () => {
                    // Show response in question text
                    if (this.dom.questionText && opt.response) {
                        this.dom.questionText.textContent = opt.response;
                    }
                    // Advance after a delay
                    setTimeout(() => this.advancePhase(), 2500);
                });
                
                // Add hint styling if this is the "best" option
                if (opt.quality === 'best') {
                    btn.dataset.quality = 'best';
                }
                
                this.dom.choiceContainer.appendChild(btn);
            });
        } else {
            // No options - just a continue button
            const btn = this.createButton("Continue", () => this.advancePhase());
            this.dom.choiceContainer.appendChild(btn);
        }
        
        // Update Kelly pose if Unity is ready
        if (atomContent.kellyPose) {
            this.sendToUnity('kelly-pose', { pose: atomContent.kellyPose });
        }
    }
    
    /**
     * Advance to next phase in the lesson
     * Phases match database: Hook → Fact1 → Fact2 → Fact3 → Wisdom
     */
    advancePhase() {
        const phases = ['welcome', 'Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom', 'complete'];
        const currentIndex = phases.indexOf(this.state.lessonPhase);
        
        if (currentIndex < phases.length - 1) {
            this.state.lessonPhase = phases[currentIndex + 1];
            
            if (this.state.lessonPhase === 'complete') {
                // Lesson finished - show share prompt (Earn to Learn)
                this.showLessonCompleteWithShare();
            } else {
                this.renderPhase();
            }
        }
    }

    createButton(text, onClick) {
        const btn = document.createElement('div');
        btn.className = 'choice-card glass-panel-medium hover-lift';
        btn.textContent = text;
        btn.onclick = onClick;
        return btn;
    }

    /**
     * Handle age change - update archetype (if no manual override) and reload content
     */
    handleAgeChange(newAge) {
        this.state.age = newAge;
        
        // Only auto-change archetype if user hasn't manually selected one
        const manualChoice = localStorage.getItem('kelly_archetype');
        if (!manualChoice) {
            const newArchetype = this.getArchetypeForAge(newAge);
            
            if (this.state.currentArchetype !== newArchetype) {
                this.state.currentArchetype = newArchetype;
                console.log(`Age ${newAge} → Auto-Archetype: ${newArchetype}`);
                // Re-render with new archetype's content
                this.renderLessonState();
            }
        }
    }

    // --- Audio ---
    
    loadAudioForPhase(phase) {
        const filename = `${this.state.ageBucket}-${this.state.language}-${phase}.mp3`;
        // Corrected path relative to root (since app.js is module in player.html)
        const path = `lessons/audio/${this.state.lessonId}/${filename}`;
        this.dom.audio.src = path;
    }

    playAudio() {
        this.dom.audio.play()
            .then(() => {
                this.state.isPlaying = true;
                this.updatePlayButton();
                this.sendToUnity('kelly-play', {});
            })
            .catch(e => {
                // Suppress errors for missing audio files in production
                console.log("Audio not available for this phase yet.");
            });
    }

    togglePlay() {
        if (this.state.isPlaying) {
            this.dom.audio.pause();
            this.state.isPlaying = false;
            this.sendToUnity('kelly-idle', {});
        } else {
            this.playAudio();
        }
        this.updatePlayButton();
    }

    updatePlayButton() {
        const icon = this.dom.playButton.querySelector('.play-icon');
        icon.textContent = this.state.isPlaying ? '⏸' : '▶';
    }

    // --- Listeners ---

    setupEventListeners() {
        // 1. The Main Trigger (Hamburger)
        this.dom.osTrigger?.addEventListener('click', () => this.toggleDrawer());

        // 2. Drawer Actions
        document.querySelector('.btn-close-drawer')?.addEventListener('click', () => this.toggleDrawer(false));
        
        // "Start" button inside drawer
        this.dom.drawerStartBtn?.addEventListener('click', () => {
            this.switchMode('lesson');
            this.toggleDrawer(false);
        });

        // Drawer Navigation Items
        document.querySelectorAll('.drawer-item').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const target = e.currentTarget.dataset.target;
                const action = e.currentTarget.dataset.action;
                
                if (target === 'mode-lesson') {
                    this.switchMode('lesson');
                } else if (action === 'open-tuition') {
                    this.openModal('tuition');
                } else if (action === 'open-careers') {
                    this.openContentModal('Careers', 'Careers', 'We are hiring!');
                } else if (action === 'open-syllabus') {
                    this.openContentModal('Syllabus', '2025 Syllabus', 'Explore our curriculum.');
                } else if (action === 'open-newsroom') {
                    this.openContentModal('Newsroom', 'Newsroom', 'Latest updates.');
                } else if (action === 'open-settings') {
                    this.openSettingsModal();
                } else if (action === 'open-earnings') {
                    this.openEarningsDashboard();
                } else if (action === 'logout') {
                    this.supabase.auth.signOut().then(() => window.location.href = 'index.html');
                }
            });
        });

        // Modal Close
        document.querySelectorAll('.btn-close-modal').forEach(btn => {
            btn.addEventListener('click', () => this.closeAllModals());
        });
        
        // Auth: Google Login with Referral Attribution
        document.getElementById('btn-login-google')?.addEventListener('click', () => this.loginWithGoogle());
        
        // Auth: Guest mode
        document.getElementById('btn-enter-guest')?.addEventListener('click', () => {
            this.switchMode('lesson');
            // Hide attract mode panel
            document.getElementById('mode-attract')?.classList.remove('active');
        });
        
        // Start Lesson (Fallback from old dashboard if visible)
        document.getElementById('btn-start-lesson')?.addEventListener('click', () => this.switchMode('lesson'));

        // Audio Controls
        if (this.dom.playButton) this.dom.playButton.addEventListener('click', () => this.togglePlay());
        if (this.dom.audio) {
            this.dom.audio.addEventListener('ended', () => {
                this.state.isPlaying = false;
                this.updatePlayButton();
                this.sendToUnity('kelly-idle', {});
            });
        }
        
        // Age Slider - connected to archetype system
        if (this.dom.ageSlider) {
            this.dom.ageSlider.addEventListener('input', (e) => {
                const newAge = parseInt(e.target.value);
                if (this.dom.ageValue) this.dom.ageValue.textContent = newAge;
                this.handleAgeChange(newAge);
            });
        }
    }

    // --- Earn to Learn System ---

    /**
     * Capture referral code from URL and store for lifetime attribution
     * Called on page load - ?ref= parameter triggers tracking
     */
    captureReferral() {
        const urlParams = new URLSearchParams(window.location.search);
        const refCode = urlParams.get('ref');
        
        if (refCode) {
            // Store in localStorage (persists forever on this device)
            localStorage.setItem('kelly_referrer', refCode);
            localStorage.setItem('kelly_referrer_timestamp', Date.now().toString());
            
            // Also store in cookie for cross-subdomain access
            const maxAge = 31536000 * 10; // 10 years (effectively forever)
            document.cookie = `kelly_ref=${refCode}; max-age=${maxAge}; path=/; SameSite=Lax`;
            
            // Track the click server-side (lifetime attribution)
            this.trackReferralClick(refCode);
            
            console.log(`[Earn to Learn] Referral captured: ${refCode}`);
        }
    }

    /**
     * Track referral click with the API for lifetime attribution
     */
    async trackReferralClick(referralCode) {
        try {
            const response = await fetch(`${API_URL}/referral/track`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    referralCode: referralCode,
                    sourceUrl: document.referrer,
                    landingPage: window.location.pathname,
                    utmSource: new URLSearchParams(window.location.search).get('utm_source'),
                    utmMedium: new URLSearchParams(window.location.search).get('utm_medium'),
                    utmCampaign: new URLSearchParams(window.location.search).get('utm_campaign')
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('[Earn to Learn] Click tracked:', data.clickId);
            }
        } catch (error) {
            // Silent fail - don't break user experience for tracking
            console.warn('[Earn to Learn] Failed to track click:', error);
        }
    }

    /**
     * Load user's earnings data from Supabase
     */
    async loadEarningsData() {
        if (!this.state.user) {
            // Not logged in - show placeholder data
            this.updateEarningsUI({
                referral_code: 'sign-in-to-earn',
                commission_tier: 'new_learner',
                commission_rate: 0.10,
                pending_earnings: 0,
                available_earnings: 0,
                lifetime_earnings: 0,
                total_referrals: 0
            });
            return;
        }
        
        try {
            const { data: user, error } = await this.supabase
                .from('users')
                .select(`
                    referral_code,
                    commission_tier,
                    commission_rate,
                    pending_earnings,
                    available_earnings,
                    lifetime_earnings,
                    total_referrals,
                    total_lessons_completed
                `)
                .eq('id', this.state.user.id)
                .single();
            
            if (error) {
                console.error('[Earn to Learn] Failed to load earnings:', error);
                return;
            }
            
            if (user) {
                this.state.earnings = {
                    referralCode: user.referral_code,
                    commissionTier: user.commission_tier,
                    commissionRate: user.commission_rate,
                    pendingEarnings: parseFloat(user.pending_earnings) || 0,
                    availableEarnings: parseFloat(user.available_earnings) || 0,
                    lifetimeEarnings: parseFloat(user.lifetime_earnings) || 0,
                    totalReferrals: user.total_referrals || 0,
                    lessonsCompleted: user.total_lessons_completed || 0
                };
                this.updateEarningsUI(user);
            }
        } catch (error) {
            console.error('[Earn to Learn] Error loading earnings:', error);
        }
    }

    /**
     * Update the earnings UI in the drawer
     */
    updateEarningsUI(earnings) {
        // Update referral link
        const linkInput = document.getElementById('referral-link');
        if (linkInput && earnings.referral_code) {
            linkInput.value = `kelly.me/${earnings.referral_code}`;
        }
        
        // Update earnings display
        const pendingEl = document.getElementById('pending-earnings');
        const availableEl = document.getElementById('available-earnings');
        
        if (pendingEl) {
            pendingEl.textContent = `$${(parseFloat(earnings.pending_earnings) || 0).toFixed(2)}`;
        }
        if (availableEl) {
            availableEl.textContent = `$${(parseFloat(earnings.available_earnings) || 0).toFixed(2)}`;
        }
        
        // Update tier badge
        const tierNames = {
            'new_learner': 'New Learner',
            'active_learner': 'Active Learner',
            'committed_learner': 'Committed Learner',
            'dedicated_learner': 'Dedicated Learner',
            'complete_learner': 'Complete Learner',
            'legendary_learner': 'Legendary Learner'
        };
        
        const tierBadge = document.getElementById('commission-tier-badge');
        const rateDisplay = document.getElementById('commission-rate');
        
        if (tierBadge) {
            tierBadge.textContent = tierNames[earnings.commission_tier] || 'New Learner';
        }
        if (rateDisplay) {
            const rate = parseFloat(earnings.commission_rate) || 0.10;
            rateDisplay.textContent = `${(rate * 100).toFixed(0)}% commission`;
        }
    }

    /**
     * Setup share button listeners and copy functionality
     */
    setupShareListeners() {
        // Copy link button
        const copyBtn = document.getElementById('btn-copy-link');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                const linkInput = document.getElementById('referral-link');
                if (linkInput) {
                    // Full URL for sharing
                    const fullUrl = `https://curiouskelly.com/?ref=${this.state.earnings.referralCode || 'kelly'}`;
                    navigator.clipboard.writeText(fullUrl).then(() => {
                        copyBtn.classList.add('copied');
                        copyBtn.textContent = '✓';
                        setTimeout(() => {
                            copyBtn.classList.remove('copied');
                            copyBtn.textContent = '📋';
                        }, 2000);
                    });
                }
            });
        }
        
        // Social share buttons
        const setupShare = (id, urlBuilder) => {
            const btn = document.getElementById(id);
            if (btn) {
                btn.addEventListener('click', () => {
                    const refCode = this.state.earnings.referralCode || 'kelly';
                    const shareUrl = `https://curiouskelly.com/?ref=${refCode}`;
                    const shareText = "I'm learning something new every day with Curious Kelly! Join me:";
                    window.open(urlBuilder(shareText, shareUrl), '_blank', 'width=600,height=400');
                });
            }
        };
        
        setupShare('share-twitter', (text, url) => 
            `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`
        );
        
        setupShare('share-facebook', (text, url) => 
            `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`
        );
        
        setupShare('share-whatsapp', (text, url) => 
            `https://wa.me/?text=${encodeURIComponent(text + ' ' + url)}`
        );
        
        setupShare('share-email', (text, url) => 
            `mailto:?subject=${encodeURIComponent('Join me on Curious Kelly!')}&body=${encodeURIComponent(text + '\n\n' + url)}`
        );
        
        // View earnings dashboard
        const viewEarningsBtn = document.getElementById('btn-view-earnings');
        if (viewEarningsBtn) {
            viewEarningsBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.openEarningsDashboard();
            });
        }
    }

    /**
     * Show lesson complete screen with share prompt
     */
    showLessonCompleteWithShare() {
        const lesson = this.state.currentLesson;
        const refCode = this.state.earnings.referralCode || 'kelly';
        const shareUrl = `https://curiouskelly.com/?ref=${refCode}&day=${lesson?.dayNumber || 1}`;
        const commissionRate = (this.state.earnings.commissionRate * 100) || 10;
        
        if (this.dom.questionText) {
            this.dom.questionText.innerHTML = `
                <div class="lesson-complete-message">
                    <span class="complete-icon">✨</span>
                    <h3>Lesson Complete!</h3>
                    <p>"${lesson?.topic || 'Today\'s Lesson'}"</p>
                </div>
            `;
        }
        
        if (this.dom.choiceContainer) {
            this.dom.choiceContainer.innerHTML = `
                <div class="share-prompt glass-panel-medium">
                    <p class="share-prompt-text">Know someone who'd love this lesson?</p>
                    <div class="share-prompt-buttons">
                        <button class="share-btn-large" id="share-lesson-twitter">
                            Share on 𝕏
                        </button>
                        <button class="share-btn-large" id="share-lesson-whatsapp">
                            Share on WhatsApp
                        </button>
                    </div>
                    <p class="share-prompt-earnings">
                        You'll earn ${commissionRate}% if they subscribe!
                    </p>
                </div>
                <div class="choice-card glass-panel-medium hover-lift" id="btn-finish-lesson">
                    Continue to Dashboard →
                </div>
            `;
            
            // Add share listeners
            document.getElementById('share-lesson-twitter')?.addEventListener('click', () => {
                const text = `I just learned "${lesson?.topic || 'something amazing'}" with @CuriousKelly! 🎓 Join me:`;
                window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(shareUrl)}`, '_blank');
            });
            
            document.getElementById('share-lesson-whatsapp')?.addEventListener('click', () => {
                const text = `I just learned about "${lesson?.topic || 'something amazing'}" with Curious Kelly! You should try it:`;
                window.open(`https://wa.me/?text=${encodeURIComponent(text + ' ' + shareUrl)}`, '_blank');
            });
            
            document.getElementById('btn-finish-lesson')?.addEventListener('click', () => {
                this.switchMode('dashboard');
            });
        }
    }

    /**
     * Open full earnings dashboard modal
     */
    openEarningsDashboard() {
        const earnings = this.state.earnings;
        const tierNames = {
            'new_learner': 'New Learner',
            'active_learner': 'Active Learner',
            'committed_learner': 'Committed Learner',
            'dedicated_learner': 'Dedicated Learner',
            'complete_learner': 'Complete Learner',
            'legendary_learner': 'Legendary Learner'
        };
        
        // Calculate progress to next tier
        const tierThresholds = {
            'new_learner': { next: 'active_learner', lessons: 7 },
            'active_learner': { next: 'committed_learner', lessons: 30 },
            'committed_learner': { next: 'dedicated_learner', lessons: 100 },
            'dedicated_learner': { next: 'complete_learner', lessons: 365 },
            'complete_learner': { next: 'legendary_learner', lessons: 1000 },
            'legendary_learner': { next: null, lessons: null }
        };
        
        const currentTier = earnings.commissionTier || 'new_learner';
        const lessonsCompleted = earnings.lessonsCompleted || 0;
        const nextTierInfo = tierThresholds[currentTier];
        let progressPercent = 0;
        let progressText = 'You\'ve reached the highest tier! 🏆';
        
        if (nextTierInfo.next) {
            const currentMin = currentTier === 'new_learner' ? 0 : tierThresholds[Object.keys(tierThresholds).find(k => tierThresholds[k].next === currentTier)]?.lessons || 0;
            const range = nextTierInfo.lessons - currentMin;
            const progress = lessonsCompleted - currentMin;
            progressPercent = Math.min(100, (progress / range) * 100);
            const remaining = nextTierInfo.lessons - lessonsCompleted;
            progressText = `Complete ${remaining} more lesson${remaining !== 1 ? 's' : ''} to unlock ${tierNames[nextTierInfo.next]}!`;
        }
        
        const html = `
            <div class="earnings-dashboard">
                <!-- Summary Cards -->
                <div class="earnings-cards">
                    <div class="earnings-card">
                        <span class="card-label">Pending</span>
                        <span class="card-value">$${earnings.pendingEarnings.toFixed(2)}</span>
                        <span class="card-note">Clears in 7 days</span>
                    </div>
                    <div class="earnings-card">
                        <span class="card-label">Available</span>
                        <span class="card-value highlight">$${earnings.availableEarnings.toFixed(2)}</span>
                        <span class="card-note">Ready to withdraw</span>
                    </div>
                    <div class="earnings-card">
                        <span class="card-label">Lifetime</span>
                        <span class="card-value">$${earnings.lifetimeEarnings.toFixed(2)}</span>
                        <span class="card-note">Total earned</span>
                    </div>
                </div>
                
                <!-- Tier Progress -->
                <div class="tier-progress-section">
                    <h3>Your Commission Tier</h3>
                    <div class="tier-current">
                        <span class="tier-name">${tierNames[currentTier] || 'New Learner'}</span>
                        <span class="tier-rate">${((earnings.commissionRate || 0.10) * 100).toFixed(0)}%</span>
                    </div>
                    <div class="tier-progress-bar">
                        <div class="tier-progress-fill" style="width: ${progressPercent}%"></div>
                    </div>
                    <p class="tier-next">${progressText}</p>
                </div>
                
                <!-- Referral Stats -->
                <div class="referral-stats-section">
                    <h3>Your Network</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-number">${earnings.totalReferrals || 0}</span>
                            <span class="stat-desc">Total Referrals</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">${lessonsCompleted}</span>
                            <span class="stat-desc">Lessons Completed</span>
                        </div>
                    </div>
                </div>
                
                <!-- Your Link -->
                <div class="referral-stats-section">
                    <h3>Your Referral Link</h3>
                    <div class="referral-link-box" style="margin-top: 12px;">
                        <input type="text" readonly value="https://curiouskelly.com/?ref=${earnings.referralCode || 'kelly'}" class="referral-input" style="font-size: 0.8rem;">
                        <button class="btn-copy" id="modal-copy-link" title="Copy">📋</button>
                    </div>
                </div>
                
                <!-- Payout Button -->
                <div class="payout-section">
                    <button class="btn-primary-glass btn-payout" ${earnings.availableEarnings < 50 ? 'disabled' : ''}>
                        ${earnings.availableEarnings >= 50 ? 'Request Payout' : `Request Payout ($50 minimum)`}
                    </button>
                </div>
            </div>
        `;
        
        this.openModal('reader', { title: '💰 Your Earnings', body: html });
        
        // Setup copy button in modal
        setTimeout(() => {
            document.getElementById('modal-copy-link')?.addEventListener('click', function() {
                const input = this.previousElementSibling;
                navigator.clipboard.writeText(input.value).then(() => {
                    this.textContent = '✓';
                    this.classList.add('copied');
                    setTimeout(() => {
                        this.textContent = '📋';
                        this.classList.remove('copied');
                    }, 2000);
                });
            });
        }, 100);
    }

    // --- Unity ---

    setupUnity() {
        window.addEventListener('message', (event) => {
            if (event.data && event.data.source === 'kelly-webgl') {
                if (event.data.type === 'kelly-ready') {
                    this.state.unityReady = true;
                    this.dom.unityStatus.classList.add('hidden');
                    this.dom.unityIframe.classList.add('ready');
                }
            }
        });
        if (!this.dom.unityIframe.getAttribute('src')) {
             this.dom.unityIframe.src = "unity/kelly-v1/index.html";
        }
    }

    sendToUnity(type, payload) {
        if (this.state.unityReady && this.dom.unityIframe.contentWindow) {
            this.dom.unityIframe.contentWindow.postMessage({
                source: 'curiouskelly.com',
                destination: 'kelly-webgl',
                type,
                payload
            }, '*');
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.os = new KellyOS();
});