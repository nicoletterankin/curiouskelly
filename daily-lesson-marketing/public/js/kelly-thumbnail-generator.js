/**
 * KELLY THUMBNAIL GENERATOR
 * 
 * Generates dynamic, personalized lesson thumbnails that adapt to:
 * - User's age (visual style, Kelly's appearance)
 * - User's tone preference (colors, energy)
 * - User's language (text content)
 * - Lesson topic (visual theme, patterns)
 * 
 * Uses procedural CSS/SVG generation for instant, on-demand visuals.
 */

class KellyThumbnailGenerator {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' 
            ? document.querySelector(container) 
            : container;
        this.options = {
            width: options.width || 320,
            height: options.height || 200,
            animated: options.animated !== false,
            showKelly: options.showKelly !== false,
            showText: options.showText !== false,
            showDate: options.showDate !== false,
            clickable: options.clickable !== false,
            ...options
        };
        
        // Initialize visual DNA if available
        this.visualDNA = typeof LESSON_VISUAL_DNA !== 'undefined' ? LESSON_VISUAL_DNA : null;
    }

    /**
     * Generate a thumbnail for a specific lesson
     */
    generate(lessonData, settings = {}) {
        const { dayNumber, topic, date, hook, universalTruth } = lessonData;
        const { age = 25, tone = 'curious', language = 'en' } = settings;

        // Get visual data from DNA
        const visuals = this.visualDNA?.getLessonVisuals(dayNumber, settings) || this.getFallbackVisuals(dayNumber);

        // Create thumbnail container
        const thumb = document.createElement('div');
        thumb.className = 'kelly-thumbnail';
        thumb.setAttribute('data-day', dayNumber);
        thumb.setAttribute('data-age', age);
        thumb.setAttribute('data-tone', tone);

        // Apply styles
        this.applyStyles(thumb, visuals, settings);

        // Build layers
        thumb.innerHTML = `
            ${this.generateBackgroundLayer(visuals, settings)}
            ${this.generatePatternLayer(visuals, settings)}
            ${this.options.showKelly ? this.generateKellyLayer(visuals, settings) : ''}
            ${this.generateContentLayer(lessonData, visuals, settings)}
            ${this.options.animated ? this.generateAnimationLayer(visuals) : ''}
        `;

        // Add interactivity
        if (this.options.clickable) {
            thumb.style.cursor = 'pointer';
            thumb.addEventListener('click', () => {
                this.onThumbnailClick(dayNumber, settings);
            });
        }

        return thumb;
    }

    /**
     * Generate gradient background based on category
     */
    generateBackgroundLayer(visuals, settings) {
        const { colors, category, energy } = visuals;
        const { age } = settings;
        const ageGroup = this.visualDNA?.getAgeGroup(age) || 'youngAdults';
        
        // Different gradient styles per age group
        const gradientStyles = {
            young: `linear-gradient(135deg, ${colors.primary[0]} 0%, ${colors.accent[0]} 50%, ${colors.primary[2]} 100%)`,
            kids: `linear-gradient(150deg, ${colors.primary[0]} 0%, ${colors.primary[1]} 60%, ${colors.accent[1]} 100%)`,
            teens: `linear-gradient(180deg, ${colors.primary[0]} 0%, ${colors.primary[1]} 100%)`,
            youngAdults: `linear-gradient(160deg, ${colors.primary[0]} 0%, ${colors.primary[2]} 100%)`,
            adults: `linear-gradient(170deg, ${colors.primary[0]} 0%, ${colors.primary[1]} 80%, ${colors.primary[2]} 100%)`,
            seniors: `linear-gradient(180deg, ${colors.primary[1]} 0%, ${colors.primary[0]} 100%)`
        };

        return `
            <div class="thumb-bg" style="
                position: absolute;
                inset: 0;
                background: ${gradientStyles[ageGroup]};
                border-radius: inherit;
            "></div>
        `;
    }

    /**
     * Generate pattern overlay based on lesson category
     */
    generatePatternLayer(visuals, settings) {
        const { pattern, colors, patternVariant } = visuals;
        const { age } = settings;
        const opacity = age <= 12 ? 0.3 : age <= 35 ? 0.2 : 0.15;

        const patterns = {
            stardust: this.generateStardustPattern(colors, opacity),
            organic: this.generateOrganicPattern(colors, opacity),
            waves: this.generateWavesPattern(colors, opacity),
            neural: this.generateNeuralPattern(colors, opacity),
            heartbeat: this.generateHeartbeatPattern(colors, opacity),
            sunrise: this.generateSunrisePattern(colors, opacity),
            geometric: this.generateGeometricPattern(colors, opacity),
            cellular: this.generateCellularPattern(colors, opacity),
            connection: this.generateConnectionPattern(colors, opacity),
            terrain: this.generateTerrainPattern(colors, opacity)
        };

        return patterns[pattern] || patterns.geometric;
    }

    /**
     * Generate Kelly avatar layer
     */
    generateKellyLayer(visuals, settings) {
        const { kellyStyle, icon, kellySpeaks } = visuals;
        const { age, tone } = settings;
        
        // Different Kelly representations for different ages
        const kellySize = age <= 12 ? 80 : age <= 35 ? 60 : 50;
        const kellyPosition = age <= 12 ? 'bottom-right-large' : 'bottom-right';
        
        // Kelly avatar SVG (stylized based on age)
        const kellyAvatar = this.generateKellyAvatar(kellyStyle, kellySize);
        
        // Speech bubble with personalized message
        const speechBubble = kellySpeaks ? this.generateSpeechBubble(kellySpeaks, kellyStyle, age) : '';

        return `
            <div class="thumb-kelly ${kellyPosition}" style="
                position: absolute;
                bottom: 12px;
                right: 12px;
                display: flex;
                align-items: flex-end;
                gap: 8px;
                z-index: 10;
            ">
                ${speechBubble}
                ${kellyAvatar}
            </div>
        `;
    }

    /**
     * Generate Kelly's avatar SVG
     */
    generateKellyAvatar(kellyStyle, size) {
        const { avatarStyle, colors } = kellyStyle;
        
        // Different avatar styles
        const avatarStyles = {
            playful: { faceColor: '#FFE4C4', hairColor: '#8B4513', expression: '😊' },
            friendly: { faceColor: '#FFE4C4', hairColor: '#8B4513', expression: '🤗' },
            cool: { faceColor: '#FFE4C4', hairColor: '#654321', expression: '😎' },
            professional: { faceColor: '#FFE4C4', hairColor: '#5D4037', expression: '🧐' },
            refined: { faceColor: '#FFE4C4', hairColor: '#4A4A4A', expression: '😌' },
            warm: { faceColor: '#FFE4C4', hairColor: '#696969', expression: '🥰' }
        };

        const style = avatarStyles[avatarStyle] || avatarStyles.friendly;

        return `
            <div class="kelly-avatar" style="
                width: ${size}px;
                height: ${size}px;
                border-radius: 50%;
                background: linear-gradient(180deg, ${style.hairColor} 0%, ${style.hairColor} 35%, ${style.faceColor} 35%, ${style.faceColor} 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: ${size * 0.4}px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                border: 3px solid rgba(255,255,255,0.3);
                position: relative;
                overflow: hidden;
            ">
                <svg viewBox="0 0 100 100" width="${size}" height="${size}" style="position: absolute; top: 0; left: 0;">
                    <!-- Hair -->
                    <ellipse cx="50" cy="30" rx="40" ry="30" fill="${style.hairColor}"/>
                    <!-- Face -->
                    <ellipse cx="50" cy="55" rx="32" ry="35" fill="${style.faceColor}"/>
                    <!-- Eyes -->
                    <ellipse cx="38" cy="50" rx="5" ry="6" fill="#333"/>
                    <ellipse cx="62" cy="50" rx="5" ry="6" fill="#333"/>
                    <circle cx="40" cy="48" r="2" fill="#fff"/>
                    <circle cx="64" cy="48" r="2" fill="#fff"/>
                    <!-- Smile -->
                    <path d="M 38 65 Q 50 75 62 65" stroke="#333" stroke-width="2" fill="none" stroke-linecap="round"/>
                    <!-- Blush -->
                    <ellipse cx="30" cy="60" rx="6" ry="4" fill="#FFB6C1" opacity="0.5"/>
                    <ellipse cx="70" cy="60" rx="6" ry="4" fill="#FFB6C1" opacity="0.5"/>
                </svg>
                <!-- Sparkle indicator -->
                <span style="position: absolute; top: -5px; right: -5px; font-size: ${size * 0.3}px;">✨</span>
            </div>
        `;
    }

    /**
     * Generate speech bubble
     */
    generateSpeechBubble(message, kellyStyle, age) {
        const fontSize = age <= 12 ? '12px' : age <= 35 ? '11px' : '10px';
        const maxWidth = age <= 12 ? '140px' : '120px';
        const bubbleStyle = kellyStyle.speechBubble || 'rounded';
        
        const bubbleStyles = {
            cloud: 'border-radius: 20px; padding: 10px 14px;',
            rounded: 'border-radius: 12px; padding: 8px 12px;',
            modern: 'border-radius: 4px 12px 12px 12px; padding: 8px 12px;',
            minimal: 'border-radius: 8px; padding: 6px 10px; border: 1px solid rgba(255,255,255,0.3);',
            elegant: 'border-radius: 16px; padding: 10px 14px; font-style: italic;',
            classic: 'border-radius: 8px; padding: 10px 14px; border: 2px solid rgba(255,255,255,0.4);'
        };

        return `
            <div class="kelly-speech" style="
                background: rgba(255,255,255,0.95);
                color: #1a1a2e;
                font-size: ${fontSize};
                font-weight: 500;
                max-width: ${maxWidth};
                line-height: 1.3;
                ${bubbleStyles[bubbleStyle]}
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                position: relative;
            ">
                "${message}"
                <div style="
                    position: absolute;
                    right: -6px;
                    bottom: 8px;
                    width: 12px;
                    height: 12px;
                    background: rgba(255,255,255,0.95);
                    transform: rotate(45deg);
                "></div>
            </div>
        `;
    }

    /**
     * Generate content overlay (date, title, icon)
     */
    generateContentLayer(lessonData, visuals, settings) {
        const { dayNumber, topic, date, hook } = lessonData;
        const { icon, colors, kellyStyle } = visuals;
        const { age, language } = settings;
        
        const fontSize = kellyStyle?.fontSize || 'medium';
        const fontSizes = {
            'large': { date: '11px', title: '18px', hook: '12px' },
            'medium-large': { date: '10px', title: '16px', hook: '11px' },
            'medium': { date: '9px', title: '15px', hook: '10px' },
            'readable': { date: '9px', title: '14px', hook: '10px' },
            'comfortable': { date: '10px', title: '15px', hook: '11px' }
        };
        const sizes = fontSizes[fontSize] || fontSizes.medium;

        // Format date based on language
        const formattedDate = this.formatDate(dayNumber, language);

        return `
            <div class="thumb-content" style="
                position: absolute;
                inset: 0;
                padding: 16px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                z-index: 5;
            ">
                <!-- Top: Date & Icon -->
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    ${this.options.showDate ? `
                        <span style="
                            font-size: ${sizes.date};
                            text-transform: uppercase;
                            letter-spacing: 0.1em;
                            color: rgba(255,255,255,0.7);
                            font-weight: 600;
                        ">${formattedDate}</span>
                    ` : ''}
                    <span style="font-size: 24px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));">${icon}</span>
                </div>
                
                <!-- Bottom: Title -->
                <div style="max-width: ${this.options.showKelly ? '65%' : '90%'};">
                    <h3 style="
                        font-family: 'Fraunces', Georgia, serif;
                        font-size: ${sizes.title};
                        font-weight: 500;
                        color: #fff;
                        margin: 0;
                        line-height: 1.2;
                        text-shadow: 0 2px 8px rgba(0,0,0,0.4);
                    ">${topic}</h3>
                </div>
            </div>
        `;
    }

    /**
     * Generate animation layer (particles, glows)
     */
    generateAnimationLayer(visuals) {
        const { pattern, colors, energy } = visuals;
        const accent = colors.accent[0];

        return `
            <div class="thumb-animation" style="
                position: absolute;
                inset: 0;
                pointer-events: none;
                overflow: hidden;
                border-radius: inherit;
            ">
                <!-- Floating particles -->
                ${this.generateFloatingParticles(accent, energy)}
                <!-- Ambient glow -->
                <div style="
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    width: 150%;
                    height: 150%;
                    transform: translate(-50%, -50%);
                    background: radial-gradient(circle, ${accent}15 0%, transparent 70%);
                    animation: pulse 4s ease-in-out infinite;
                "></div>
            </div>
        `;
    }

    /**
     * Generate floating particles
     */
    generateFloatingParticles(color, energy) {
        const particleCount = energy === 'spark' ? 8 : energy === 'wonder' ? 6 : 4;
        let particles = '';
        
        for (let i = 0; i < particleCount; i++) {
            const size = 2 + Math.random() * 4;
            const x = 10 + Math.random() * 80;
            const y = 10 + Math.random() * 80;
            const delay = Math.random() * 3;
            const duration = 3 + Math.random() * 4;
            
            particles += `
                <div style="
                    position: absolute;
                    left: ${x}%;
                    top: ${y}%;
                    width: ${size}px;
                    height: ${size}px;
                    background: ${color};
                    border-radius: 50%;
                    opacity: 0.6;
                    animation: float ${duration}s ease-in-out ${delay}s infinite;
                    box-shadow: 0 0 ${size * 2}px ${color};
                "></div>
            `;
        }
        
        return particles;
    }

    // === PATTERN GENERATORS ===

    generateStardustPattern(colors, opacity) {
        const stars = [];
        for (let i = 0; i < 20; i++) {
            stars.push(`
                <circle 
                    cx="${Math.random() * 100}" 
                    cy="${Math.random() * 100}" 
                    r="${0.5 + Math.random() * 1.5}" 
                    fill="${colors.accent[Math.floor(Math.random() * colors.accent.length)]}"
                    opacity="${0.3 + Math.random() * 0.5}"
                />
            `);
        }
        return `
            <svg class="thumb-pattern" viewBox="0 0 100 100" preserveAspectRatio="none" style="
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                opacity: ${opacity};
            ">
                ${stars.join('')}
            </svg>
        `;
    }

    generateOrganicPattern(colors, opacity) {
        return `
            <svg class="thumb-pattern" viewBox="0 0 100 100" preserveAspectRatio="none" style="
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                opacity: ${opacity};
            ">
                <path d="M0,50 Q25,30 50,50 T100,50 L100,100 L0,100 Z" fill="${colors.accent[0]}" opacity="0.3"/>
                <path d="M0,60 Q30,40 60,60 T100,60 L100,100 L0,100 Z" fill="${colors.accent[1]}" opacity="0.2"/>
                <circle cx="70" cy="30" r="15" fill="${colors.accent[0]}" opacity="0.2"/>
                <circle cx="20" cy="20" r="10" fill="${colors.accent[1]}" opacity="0.15"/>
            </svg>
        `;
    }

    generateWavesPattern(colors, opacity) {
        return `
            <svg class="thumb-pattern" viewBox="0 0 100 100" preserveAspectRatio="none" style="
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                opacity: ${opacity};
            ">
                <path d="M-10,80 Q10,70 30,80 T70,80 T110,80 L110,110 L-10,110 Z" fill="${colors.accent[0]}" opacity="0.4">
                    <animate attributeName="d" 
                        values="M-10,80 Q10,70 30,80 T70,80 T110,80 L110,110 L-10,110 Z;
                                M-10,80 Q10,90 30,80 T70,80 T110,80 L110,110 L-10,110 Z;
                                M-10,80 Q10,70 30,80 T70,80 T110,80 L110,110 L-10,110 Z"
                        dur="4s" repeatCount="indefinite"/>
                </path>
                <path d="M-10,85 Q20,75 40,85 T80,85 T110,85 L110,110 L-10,110 Z" fill="${colors.accent[1]}" opacity="0.3">
                    <animate attributeName="d" 
                        values="M-10,85 Q20,75 40,85 T80,85 T110,85 L110,110 L-10,110 Z;
                                M-10,85 Q20,95 40,85 T80,85 T110,85 L110,110 L-10,110 Z;
                                M-10,85 Q20,75 40,85 T80,85 T110,85 L110,110 L-10,110 Z"
                        dur="5s" repeatCount="indefinite"/>
                </path>
            </svg>
        `;
    }

    generateNeuralPattern(colors, opacity) {
        const nodes = [];
        const connections = [];
        const nodePositions = [];
        
        for (let i = 0; i < 8; i++) {
            const x = 15 + Math.random() * 70;
            const y = 15 + Math.random() * 70;
            nodePositions.push({ x, y });
            nodes.push(`<circle cx="${x}" cy="${y}" r="3" fill="${colors.accent[0]}" opacity="0.6"/>`);
        }
        
        for (let i = 0; i < nodePositions.length; i++) {
            for (let j = i + 1; j < nodePositions.length; j++) {
                if (Math.random() > 0.6) {
                    connections.push(`
                        <line x1="${nodePositions[i].x}" y1="${nodePositions[i].y}" 
                              x2="${nodePositions[j].x}" y2="${nodePositions[j].y}" 
                              stroke="${colors.accent[1]}" stroke-width="0.5" opacity="0.3"/>
                    `);
                }
            }
        }
        
        return `
            <svg class="thumb-pattern" viewBox="0 0 100 100" preserveAspectRatio="none" style="
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                opacity: ${opacity};
            ">
                ${connections.join('')}
                ${nodes.join('')}
            </svg>
        `;
    }

    generateHeartbeatPattern(colors, opacity) {
        return `
            <svg class="thumb-pattern" viewBox="0 0 100 100" preserveAspectRatio="none" style="
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                opacity: ${opacity};
            ">
                <path d="M0,50 L20,50 L25,30 L35,70 L45,40 L50,50 L100,50" 
                      stroke="${colors.accent[0]}" fill="none" stroke-width="2" opacity="0.5">
                    <animate attributeName="stroke-dasharray" values="0,200;200,0" dur="2s" repeatCount="indefinite"/>
                </path>
            </svg>
        `;
    }

    generateSunrisePattern(colors, opacity) {
        return `
            <svg class="thumb-pattern" viewBox="0 0 100 100" preserveAspectRatio="none" style="
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                opacity: ${opacity};
            ">
                <defs>
                    <radialGradient id="sunGlow" cx="50%" cy="100%" r="60%">
                        <stop offset="0%" stop-color="${colors.accent[0]}" stop-opacity="0.5"/>
                        <stop offset="100%" stop-color="${colors.accent[0]}" stop-opacity="0"/>
                    </radialGradient>
                </defs>
                <circle cx="50" cy="100" r="40" fill="url(#sunGlow)"/>
                ${[...Array(8)].map((_, i) => `
                    <line x1="50" y1="100" x2="${50 + Math.cos((i * Math.PI) / 8) * 60}" 
                          y2="${100 - Math.sin((i * Math.PI) / 8) * 60}" 
                          stroke="${colors.accent[1]}" stroke-width="0.5" opacity="0.3"/>
                `).join('')}
            </svg>
        `;
    }

    generateGeometricPattern(colors, opacity) {
        return `
            <svg class="thumb-pattern" viewBox="0 0 100 100" preserveAspectRatio="none" style="
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                opacity: ${opacity};
            ">
                <polygon points="50,10 90,30 90,70 50,90 10,70 10,30" fill="none" 
                         stroke="${colors.accent[0]}" stroke-width="1" opacity="0.4"/>
                <polygon points="50,20 80,35 80,65 50,80 20,65 20,35" fill="none" 
                         stroke="${colors.accent[1]}" stroke-width="0.5" opacity="0.3"/>
                <circle cx="50" cy="50" r="15" fill="none" stroke="${colors.accent[2] || colors.accent[0]}" 
                        stroke-width="0.5" opacity="0.3"/>
            </svg>
        `;
    }

    generateCellularPattern(colors, opacity) {
        const cells = [];
        for (let i = 0; i < 6; i++) {
            const x = 20 + Math.random() * 60;
            const y = 20 + Math.random() * 60;
            const r = 5 + Math.random() * 10;
            cells.push(`
                <circle cx="${x}" cy="${y}" r="${r}" fill="none" 
                        stroke="${colors.accent[i % colors.accent.length]}" stroke-width="1" opacity="0.4"/>
                <circle cx="${x}" cy="${y}" r="${r * 0.3}" fill="${colors.accent[i % colors.accent.length]}" opacity="0.3"/>
            `);
        }
        return `
            <svg class="thumb-pattern" viewBox="0 0 100 100" preserveAspectRatio="none" style="
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                opacity: ${opacity};
            ">
                ${cells.join('')}
            </svg>
        `;
    }

    generateConnectionPattern(colors, opacity) {
        return `
            <svg class="thumb-pattern" viewBox="0 0 100 100" preserveAspectRatio="none" style="
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                opacity: ${opacity};
            ">
                <circle cx="20" cy="30" r="8" fill="${colors.accent[0]}" opacity="0.4"/>
                <circle cx="50" cy="20" r="6" fill="${colors.accent[1]}" opacity="0.3"/>
                <circle cx="80" cy="35" r="10" fill="${colors.accent[0]}" opacity="0.35"/>
                <circle cx="30" cy="70" r="7" fill="${colors.accent[1]}" opacity="0.3"/>
                <circle cx="70" cy="75" r="9" fill="${colors.accent[0]}" opacity="0.35"/>
                <line x1="20" y1="30" x2="50" y2="20" stroke="${colors.accent[1]}" stroke-width="1" opacity="0.3"/>
                <line x1="50" y1="20" x2="80" y2="35" stroke="${colors.accent[0]}" stroke-width="1" opacity="0.3"/>
                <line x1="20" y1="30" x2="30" y2="70" stroke="${colors.accent[1]}" stroke-width="1" opacity="0.3"/>
                <line x1="80" y1="35" x2="70" y2="75" stroke="${colors.accent[0]}" stroke-width="1" opacity="0.3"/>
                <line x1="30" y1="70" x2="70" y2="75" stroke="${colors.accent[1]}" stroke-width="1" opacity="0.3"/>
            </svg>
        `;
    }

    generateTerrainPattern(colors, opacity) {
        return `
            <svg class="thumb-pattern" viewBox="0 0 100 100" preserveAspectRatio="none" style="
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                opacity: ${opacity};
            ">
                <path d="M0,70 L20,60 L40,75 L60,55 L80,70 L100,60 L100,100 L0,100 Z" 
                      fill="${colors.accent[0]}" opacity="0.3"/>
                <path d="M0,80 L30,70 L50,85 L70,65 L100,80 L100,100 L0,100 Z" 
                      fill="${colors.accent[1]}" opacity="0.25"/>
            </svg>
        `;
    }

    // === UTILITY METHODS ===

    applyStyles(element, visuals, settings) {
        const { kellyStyle, colors } = visuals;
        const radius = kellyStyle?.cornerRadius || 16;

        element.style.cssText = `
            position: relative;
            width: ${this.options.width}px;
            height: ${this.options.height}px;
            border-radius: ${radius}px;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        `;

        // Add hover effect
        element.addEventListener('mouseenter', () => {
            element.style.transform = 'translateY(-4px) scale(1.02)';
            element.style.boxShadow = `0 12px 40px rgba(0,0,0,0.4), 0 0 30px ${colors.accent[0]}30`;
        });
        element.addEventListener('mouseleave', () => {
            element.style.transform = 'translateY(0) scale(1)';
            element.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
        });
    }

    formatDate(dayNumber, language) {
        const months = {
            en: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
            es: ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'],
            fr: ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
        };
        
        const daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let remaining = dayNumber;
        let monthIndex = 0;
        
        while (remaining > daysInMonth[monthIndex]) {
            remaining -= daysInMonth[monthIndex];
            monthIndex++;
        }
        
        const monthNames = months[language] || months.en;
        return `${monthNames[monthIndex]} ${remaining}, 2025`;
    }

    getFallbackVisuals(dayNumber) {
        const hue = (dayNumber * 37) % 360;
        return {
            dayNumber,
            category: 'growth',
            icon: '✨',
            kellySpeaks: 'Ready to learn something new?',
            patternVariant: 'A',
            colors: {
                primary: [`hsl(${hue}, 60%, 15%)`, `hsl(${hue}, 50%, 25%)`, `hsl(${hue}, 40%, 10%)`],
                accent: [`hsl(${(hue + 30) % 360}, 80%, 60%)`, `hsl(${(hue + 60) % 360}, 70%, 50%)`, `hsl(${(hue + 90) % 360}, 90%, 70%)`],
                hue,
                saturation: 70,
                brightness: 85
            },
            pattern: 'geometric',
            energy: 'grow',
            kellyStyle: {
                avatarStyle: 'friendly',
                bgEnergy: 'dynamic',
                lineWeight: 'medium',
                cornerRadius: 16,
                fontSize: 'medium',
                animation: 'float',
                kellyPose: 'pointing',
                speechBubble: 'rounded',
                colors: { saturation: 70, brightness: 85 }
            },
            tone: { accentShift: 0, energyBoost: 1.0, iconStyle: '🔍', animationSpeed: 1.0 },
            ageGroup: 'youngAdults'
        };
    }

    onThumbnailClick(dayNumber, settings) {
        const { language, age, tone } = settings;
        const url = `/learn.html?day=${dayNumber}&lang=${language || 'en'}&age=${age || 25}&tone=${tone || 'curious'}`;
        window.location.href = url;
    }

    /**
     * Generate CSS for animations (call once per page)
     */
    static injectStyles() {
        if (document.getElementById('kelly-thumbnail-styles')) return;
        
        const styles = document.createElement('style');
        styles.id = 'kelly-thumbnail-styles';
        styles.textContent = `
            @keyframes float {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 0.3; }
                50% { opacity: 0.6; }
            }
            
            @keyframes wiggle {
                0%, 100% { transform: rotate(-2deg); }
                50% { transform: rotate(2deg); }
            }
            
            @keyframes shimmer {
                0% { background-position: -200% 0; }
                100% { background-position: 200% 0; }
            }
            
            .kelly-thumbnail:focus {
                outline: 3px solid rgba(255,255,255,0.5);
                outline-offset: 2px;
            }
            
            .thumb-animation > div {
                will-change: transform, opacity;
            }
        `;
        document.head.appendChild(styles);
    }
}

// Auto-inject styles when script loads
if (typeof document !== 'undefined') {
    KellyThumbnailGenerator.injectStyles();
}

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = KellyThumbnailGenerator;
}



