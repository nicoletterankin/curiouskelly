/**
 * Age Gate Component for COPPA Compliance
 * 
 * Requires users to confirm they are 13 or older before signup.
 * Stores confirmation in localStorage to avoid repeat prompts.
 * 
 * LEGAL: This is a "neutral age screen" approach where we ask age
 * without influencing the answer. Under 13 users are directed to
 * parent signup information.
 */

(function() {
    'use strict';

    const AGE_VERIFIED_KEY = 'kelly_age_verified';
    const VERIFICATION_EXPIRY_DAYS = 30;

    /**
     * Check if user has already verified their age
     */
    function isAgeVerified() {
        try {
            const stored = localStorage.getItem(AGE_VERIFIED_KEY);
            if (!stored) return false;
            
            const data = JSON.parse(stored);
            const now = Date.now();
            
            // Check if verification has expired
            if (data.expires && now > data.expires) {
                localStorage.removeItem(AGE_VERIFIED_KEY);
                return false;
            }
            
            return data.verified === true && data.age >= 13;
        } catch (e) {
            return false;
        }
    }

    /**
     * Store age verification
     */
    function storeAgeVerification(age) {
        const data = {
            verified: true,
            age: age,
            timestamp: Date.now(),
            expires: Date.now() + (VERIFICATION_EXPIRY_DAYS * 24 * 60 * 60 * 1000)
        };
        localStorage.setItem(AGE_VERIFIED_KEY, JSON.stringify(data));
    }

    /**
     * Create and show the age gate modal
     * @param {Function} onVerified - Callback when user is verified 13+
     */
    function showAgeGate(onVerified) {
        // Don't show if already verified
        if (isAgeVerified()) {
            if (onVerified) onVerified();
            return;
        }

        // Create modal overlay
        const overlay = document.createElement('div');
        overlay.id = 'age-gate-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(8px);
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            animation: ageGateFadeIn 0.3s ease;
        `;

        // Create modal content
        const modal = document.createElement('div');
        modal.style.cssText = `
            background: #18181b;
            border: 1px solid #3f3f46;
            border-radius: 16px;
            padding: 40px;
            max-width: 420px;
            width: 100%;
            text-align: center;
            animation: ageGateSlideUp 0.3s ease;
        `;

        modal.innerHTML = `
            <div style="margin-bottom: 24px;">
                <img src="/images/brand/kelly-mark-circle-64.png" alt="Kelly" 
                     style="width: 64px; height: 64px; border-radius: 50%; margin-bottom: 16px;"
                     onerror="this.style.display='none'">
                <h2 style="font-family: 'Fraunces', serif; font-size: 1.5rem; color: #fafafa; margin: 0 0 8px 0;">
                    Before we begin...
                </h2>
                <p style="color: #a1a1aa; font-size: 0.95rem; margin: 0;">
                    Please confirm your age to continue.
                </p>
            </div>

            <div style="margin-bottom: 24px;">
                <label style="color: #a1a1aa; font-size: 0.85rem; display: block; margin-bottom: 8px; text-align: left;">
                    What is your age?
                </label>
                <select id="age-gate-select" style="
                    width: 100%;
                    padding: 14px 16px;
                    background: #27272a;
                    border: 1px solid #3f3f46;
                    border-radius: 10px;
                    color: #fafafa;
                    font-size: 1rem;
                    cursor: pointer;
                    appearance: none;
                    background-image: url('data:image/svg+xml;utf8,<svg fill=\"%23a1a1aa\" viewBox=\"0 0 24 24\" xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M7 10l5 5 5-5z\"/></svg>');
                    background-repeat: no-repeat;
                    background-position: right 12px center;
                    background-size: 20px;
                ">
                    <option value="" disabled selected>Select your age</option>
                    <option value="under13">Under 13</option>
                    <option value="13">13</option>
                    <option value="14">14</option>
                    <option value="15">15</option>
                    <option value="16">16</option>
                    <option value="17">17</option>
                    <option value="18-24">18-24</option>
                    <option value="25-34">25-34</option>
                    <option value="35-44">35-44</option>
                    <option value="45-54">45-54</option>
                    <option value="55-64">55-64</option>
                    <option value="65+">65+</option>
                </select>
            </div>

            <button id="age-gate-continue" disabled style="
                width: 100%;
                padding: 14px 24px;
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                opacity: 0.5;
            ">
                Continue
            </button>

            <p id="age-gate-message" style="
                margin-top: 16px;
                color: #a1a1aa;
                font-size: 0.8rem;
                display: none;
            "></p>

            <p style="margin-top: 24px; color: #52525b; font-size: 0.75rem;">
                By continuing, you agree to our 
                <a href="/terms.html" style="color: #3b82f6;">Terms</a> and 
                <a href="/privacy.html" style="color: #3b82f6;">Privacy Policy</a>.
            </p>
        `;

        overlay.appendChild(modal);

        // Add animation styles
        const style = document.createElement('style');
        style.textContent = `
            @keyframes ageGateFadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            @keyframes ageGateSlideUp {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            #age-gate-select:focus {
                outline: none;
                border-color: #3b82f6;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
            }
            #age-gate-continue:not(:disabled):hover {
                background: #1d4ed8;
            }
            #age-gate-continue:disabled {
                cursor: not-allowed;
            }
        `;
        document.head.appendChild(style);
        document.body.appendChild(overlay);

        // Get elements
        const select = document.getElementById('age-gate-select');
        const continueBtn = document.getElementById('age-gate-continue');
        const message = document.getElementById('age-gate-message');

        // Handle selection change
        select.addEventListener('change', function() {
            const value = this.value;
            
            if (value === 'under13') {
                // Under 13 - show parent message
                continueBtn.disabled = true;
                continueBtn.style.opacity = '0.5';
                message.style.display = 'block';
                message.style.color = '#f59e0b';
                message.innerHTML = `
                    <strong>Parent or Guardian Required</strong><br>
                    Curious Kelly requires users under 13 to have a parent or guardian create an account on their behalf. 
                    Please ask a parent to visit <a href="mailto:hello@curiouskelly.com" style="color: #3b82f6;">hello@curiouskelly.com</a> for family account options.
                `;
            } else if (value) {
                // 13+ - enable continue
                continueBtn.disabled = false;
                continueBtn.style.opacity = '1';
                message.style.display = 'none';
            }
        });

        // Handle continue click
        continueBtn.addEventListener('click', function() {
            const value = select.value;
            if (!value || value === 'under13') return;

            // Parse age for storage
            let age = 18; // default
            if (value === '13') age = 13;
            else if (value === '14') age = 14;
            else if (value === '15') age = 15;
            else if (value === '16') age = 16;
            else if (value === '17') age = 17;
            else if (value === '18-24') age = 21;
            else if (value === '25-34') age = 30;
            else if (value === '35-44') age = 40;
            else if (value === '45-54') age = 50;
            else if (value === '55-64') age = 60;
            else if (value === '65+') age = 70;

            // Store verification
            storeAgeVerification(age);

            // Remove modal
            overlay.remove();
            style.remove();

            // Callback
            if (onVerified) onVerified();
        });

        // Close on escape
        function handleEscape(e) {
            if (e.key === 'Escape') {
                overlay.remove();
                style.remove();
                document.removeEventListener('keydown', handleEscape);
            }
        }
        document.addEventListener('keydown', handleEscape);

        // Focus select
        setTimeout(() => select.focus(), 100);
    }

    /**
     * Require age verification before executing callback
     * @param {Function} callback - Function to execute if verified
     */
    function requireAgeVerification(callback) {
        if (isAgeVerified()) {
            callback();
        } else {
            showAgeGate(callback);
        }
    }

    /**
     * Clear age verification (for testing or logout)
     */
    function clearAgeVerification() {
        localStorage.removeItem(AGE_VERIFIED_KEY);
    }

    // Export to global scope
    window.AgeGate = {
        isVerified: isAgeVerified,
        show: showAgeGate,
        require: requireAgeVerification,
        clear: clearAgeVerification
    };

})();

