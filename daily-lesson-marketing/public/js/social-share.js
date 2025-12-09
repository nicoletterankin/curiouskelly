/**
 * Curious Kelly Social Share System
 * Generates beautiful share cards for daily topics
 */

const SocialShare = {
    // Brand colors
    KELLY_BLUE: '#2563eb',
    DARK_BG: '#0f0f13',
    TEXT_WHITE: '#f4f4f5',
    
    /**
     * Generate share card as canvas
     */
    generateShareCard(topic, dayNumber, hook) {
        const canvas = document.createElement('canvas');
        canvas.width = 1200;
        canvas.height = 630;
        const ctx = canvas.getContext('2d');
        
        // Background gradient
        const gradient = ctx.createLinearGradient(0, 0, 1200, 630);
        gradient.addColorStop(0, '#0f0f13');
        gradient.addColorStop(1, '#1c1c24');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 1200, 630);
        
        // Add subtle pattern
        ctx.fillStyle = 'rgba(37, 99, 235, 0.05)';
        for (let i = 0; i < 20; i++) {
            const x = Math.random() * 1200;
            const y = Math.random() * 630;
            const size = Math.random() * 100 + 50;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fill();
        }
        
        // Kelly Blue accent bar at top
        ctx.fillStyle = this.KELLY_BLUE;
        ctx.fillRect(0, 0, 1200, 8);
        
        // Day badge
        ctx.fillStyle = this.KELLY_BLUE;
        const badgeText = `DAY ${dayNumber} OF 366`;
        ctx.font = 'bold 20px "Space Grotesk", sans-serif';
        const badgeWidth = ctx.measureText(badgeText).width + 32;
        this.roundRect(ctx, 60, 60, badgeWidth, 40, 20);
        ctx.fill();
        
        ctx.fillStyle = this.TEXT_WHITE;
        ctx.fillText(badgeText, 76, 87);
        
        // Topic title
        ctx.fillStyle = this.TEXT_WHITE;
        ctx.font = '600 64px "Fraunces", Georgia, serif';
        
        // Word wrap for long topics
        const words = topic.split(' ');
        let lines = [];
        let currentLine = '';
        const maxWidth = 1000;
        
        for (const word of words) {
            const testLine = currentLine + (currentLine ? ' ' : '') + word;
            if (ctx.measureText(testLine).width > maxWidth) {
                lines.push(currentLine);
                currentLine = word;
            } else {
                currentLine = testLine;
            }
        }
        lines.push(currentLine);
        
        // Draw topic lines
        let y = 200;
        for (const line of lines) {
            ctx.fillText(line, 60, y);
            y += 80;
        }
        
        // Hook text
        if (hook) {
            ctx.fillStyle = '#a1a1aa';
            ctx.font = '400 28px "Space Grotesk", sans-serif';
            ctx.fillText(`"${hook}"`, 60, y + 40);
        }
        
        // ✨ Curious Kelly branding
        ctx.fillStyle = this.KELLY_BLUE;
        ctx.font = '500 32px "Space Grotesk", sans-serif';
        ctx.fillText('✨ Curious Kelly', 60, 580);
        
        // Tagline
        ctx.fillStyle = '#71717a';
        ctx.font = '400 24px "Space Grotesk", sans-serif';
        ctx.fillText('Learn something wonderful. Every day.', 300, 580);
        
        return canvas;
    },
    
    /**
     * Helper: Draw rounded rectangle
     */
    roundRect(ctx, x, y, width, height, radius) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
    },
    
    /**
     * Get share card as data URL
     */
    getShareCardDataURL(topic, dayNumber, hook) {
        const canvas = this.generateShareCard(topic, dayNumber, hook);
        return canvas.toDataURL('image/png');
    },
    
    /**
     * Download share card
     */
    downloadShareCard(topic, dayNumber, hook) {
        const dataURL = this.getShareCardDataURL(topic, dayNumber, hook);
        const link = document.createElement('a');
        link.download = `curious-kelly-day-${dayNumber}.png`;
        link.href = dataURL;
        link.click();
    },
    
    /**
     * Generate share URLs for different platforms
     */
    getShareURLs(topic, dayNumber, hook) {
        const baseURL = 'https://curiouskelly.com';
        const lessonURL = `${baseURL}/learn.html?day=${dayNumber}`;
        const text = `Today I learned about ${topic} with Curious Kelly! 🌟\n\n"${hook}"\n\n`;
        const hashtags = 'CuriousKelly,DailyLesson,LearningTogether';
        
        return {
            twitter: `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(lessonURL)}&hashtags=${hashtags}`,
            facebook: `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(lessonURL)}&quote=${encodeURIComponent(text)}`,
            linkedin: `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(lessonURL)}`,
            whatsapp: `https://wa.me/?text=${encodeURIComponent(text + lessonURL)}`,
            telegram: `https://t.me/share/url?url=${encodeURIComponent(lessonURL)}&text=${encodeURIComponent(text)}`,
            email: `mailto:?subject=${encodeURIComponent(`Day ${dayNumber}: ${topic} - Curious Kelly`)}&body=${encodeURIComponent(text + '\n\n' + lessonURL)}`,
            copy: lessonURL
        };
    },
    
    /**
     * Open share dialog
     */
    share(platform, topic, dayNumber, hook) {
        const urls = this.getShareURLs(topic, dayNumber, hook);
        
        if (platform === 'copy') {
            navigator.clipboard.writeText(urls.copy)
                .then(() => this.showToast('Link copied!'))
                .catch(() => this.showToast('Failed to copy'));
            return;
        }
        
        if (platform === 'native' && navigator.share) {
            navigator.share({
                title: `Day ${dayNumber}: ${topic}`,
                text: `"${hook}" - Learn with Curious Kelly`,
                url: urls.copy
            }).catch(() => {});
            return;
        }
        
        const url = urls[platform];
        if (url) {
            window.open(url, '_blank', 'width=600,height=400');
        }
    },
    
    /**
     * Show toast notification
     */
    showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'share-toast';
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            background: #2563eb;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 500;
            z-index: 10000;
            animation: fadeInUp 0.3s ease-out;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    },
    
    /**
     * Create share button UI
     */
    createShareButton(topic, dayNumber, hook) {
        const container = document.createElement('div');
        container.className = 'share-buttons';
        container.innerHTML = `
            <button class="share-btn share-btn-primary" data-platform="native">
                <span class="share-icon">📤</span>
                Share
            </button>
            <div class="share-dropdown">
                <button class="share-btn" data-platform="twitter">
                    <span class="share-icon">𝕏</span> Twitter
                </button>
                <button class="share-btn" data-platform="facebook">
                    <span class="share-icon">📘</span> Facebook
                </button>
                <button class="share-btn" data-platform="linkedin">
                    <span class="share-icon">💼</span> LinkedIn
                </button>
                <button class="share-btn" data-platform="whatsapp">
                    <span class="share-icon">💬</span> WhatsApp
                </button>
                <button class="share-btn" data-platform="copy">
                    <span class="share-icon">🔗</span> Copy Link
                </button>
                <button class="share-btn share-btn-download" data-platform="download">
                    <span class="share-icon">📷</span> Download Card
                </button>
            </div>
        `;
        
        // Add styles
        if (!document.getElementById('share-button-styles')) {
            const styles = document.createElement('style');
            styles.id = 'share-button-styles';
            styles.textContent = `
                .share-buttons {
                    position: relative;
                    display: inline-block;
                }
                
                .share-btn {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 10px 16px;
                    background: #252530;
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    color: #f4f4f5;
                    font-size: 0.9rem;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                
                .share-btn:hover {
                    background: #2a2a36;
                    border-color: #2563eb;
                }
                
                .share-btn-primary {
                    background: #2563eb;
                    border-color: #2563eb;
                }
                
                .share-btn-primary:hover {
                    background: #3b82f6;
                }
                
                .share-dropdown {
                    position: absolute;
                    top: 100%;
                    right: 0;
                    margin-top: 8px;
                    background: #1c1c24;
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 12px;
                    padding: 8px;
                    display: none;
                    flex-direction: column;
                    gap: 4px;
                    min-width: 180px;
                    z-index: 100;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                }
                
                .share-buttons:hover .share-dropdown,
                .share-dropdown:hover {
                    display: flex;
                }
                
                .share-icon {
                    font-size: 1rem;
                }
                
                @keyframes fadeInUp {
                    from { opacity: 0; transform: translate(-50%, 10px); }
                    to { opacity: 1; transform: translate(-50%, 0); }
                }
            `;
            document.head.appendChild(styles);
        }
        
        // Add event listeners
        container.querySelectorAll('[data-platform]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const platform = e.currentTarget.dataset.platform;
                if (platform === 'download') {
                    this.downloadShareCard(topic, dayNumber, hook);
                } else {
                    this.share(platform, topic, dayNumber, hook);
                }
            });
        });
        
        return container;
    },
    
    /**
     * Update Open Graph meta tags dynamically
     */
    updateMetaTags(topic, dayNumber, hook) {
        // Title
        let title = document.querySelector('meta[property="og:title"]');
        if (!title) {
            title = document.createElement('meta');
            title.setAttribute('property', 'og:title');
            document.head.appendChild(title);
        }
        title.content = `Day ${dayNumber}: ${topic} - Curious Kelly`;
        
        // Description
        let desc = document.querySelector('meta[property="og:description"]');
        if (!desc) {
            desc = document.createElement('meta');
            desc.setAttribute('property', 'og:description');
            document.head.appendChild(desc);
        }
        desc.content = hook || `Learn about ${topic} in today's 5-minute class with Kelly.`;
        
        // Twitter card
        let twitterCard = document.querySelector('meta[name="twitter:card"]');
        if (!twitterCard) {
            twitterCard = document.createElement('meta');
            twitterCard.setAttribute('name', 'twitter:card');
            document.head.appendChild(twitterCard);
        }
        twitterCard.content = 'summary_large_image';
        
        // Image (would be server-generated in production)
        let image = document.querySelector('meta[property="og:image"]');
        if (!image) {
            image = document.createElement('meta');
            image.setAttribute('property', 'og:image');
            document.head.appendChild(image);
        }
        image.content = `https://curiouskelly.com/api/og?day=${dayNumber}`;
    }
};

// Export
window.SocialShare = SocialShare;








