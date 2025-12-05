/**
 * Kelly Message System
 * Beautiful, Kelly-voiced feedback instead of ugly browser alerts
 * 
 * Kelly's Voice: Humble, Curious, Collaborative, Warm, Simple, Rich
 */

const KellyMessages = (function() {
  'use strict';

  // Inject styles on first use
  let stylesInjected = false;

  function injectStyles() {
    if (stylesInjected) return;
    stylesInjected = true;

    const style = document.createElement('style');
    style.id = 'kelly-messages-styles';
    style.textContent = `
      .kelly-message-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(10, 10, 11, 0.85);
        backdrop-filter: blur(4px);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 100000;
        animation: kellyFadeIn 0.2s ease-out;
        padding: 20px;
      }

      @keyframes kellyFadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }

      .kelly-message-card {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 16px;
        padding: 32px;
        max-width: 420px;
        width: 100%;
        text-align: center;
        animation: kellySlideUp 0.3s ease-out;
      }

      @keyframes kellySlideUp {
        from { 
          opacity: 0;
          transform: translateY(20px);
        }
        to { 
          opacity: 1;
          transform: translateY(0);
        }
      }

      .kelly-message-icon {
        font-size: 48px;
        margin-bottom: 16px;
      }

      .kelly-message-title {
        font-family: Georgia, serif;
        font-size: 22px;
        color: #f4f4f5;
        margin: 0 0 12px;
        font-weight: 500;
      }

      .kelly-message-text {
        font-family: Georgia, serif;
        font-size: 16px;
        color: #a1a1aa;
        line-height: 1.7;
        margin: 0 0 24px;
      }

      .kelly-message-text a {
        color: #3b82f6;
        text-decoration: underline;
      }

      .kelly-message-button {
        background: #27272a;
        border: 1px solid #3f3f46;
        color: #f4f4f5;
        padding: 12px 28px;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 15px;
        cursor: pointer;
        transition: all 0.15s ease;
      }

      .kelly-message-button:hover {
        background: #3f3f46;
        border-color: #52525b;
      }

      .kelly-message-button.primary {
        background: #3b82f6;
        border-color: #3b82f6;
      }

      .kelly-message-button.primary:hover {
        background: #2563eb;
        border-color: #2563eb;
      }

      .kelly-message-signature {
        color: #71717a;
        font-family: Georgia, serif;
        font-size: 14px;
        font-style: italic;
        margin-top: 20px;
      }

      /* Toast style for quick messages */
      .kelly-toast {
        position: fixed;
        bottom: 24px;
        left: 50%;
        transform: translateX(-50%);
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 16px 24px;
        color: #f4f4f5;
        font-family: Georgia, serif;
        font-size: 15px;
        z-index: 100001;
        animation: kellyToastIn 0.3s ease-out;
        max-width: 90%;
        text-align: center;
      }

      @keyframes kellyToastIn {
        from {
          opacity: 0;
          transform: translateX(-50%) translateY(20px);
        }
        to {
          opacity: 1;
          transform: translateX(-50%) translateY(0);
        }
      }

      .kelly-toast.leaving {
        animation: kellyToastOut 0.3s ease-out forwards;
      }

      @keyframes kellyToastOut {
        from {
          opacity: 1;
          transform: translateX(-50%) translateY(0);
        }
        to {
          opacity: 0;
          transform: translateX(-50%) translateY(20px);
        }
      }
    `;
    document.head.appendChild(style);
  }

  /**
   * Show a modal message
   */
  function show(options) {
    injectStyles();

    const {
      icon = '✨',
      title,
      message,
      buttonText = 'Okay',
      buttonPrimary = false,
      signature = true,
      onClose
    } = options;

    const overlay = document.createElement('div');
    overlay.className = 'kelly-message-overlay';
    overlay.onclick = (e) => {
      if (e.target === overlay) close();
    };

    overlay.innerHTML = `
      <div class="kelly-message-card">
        <div class="kelly-message-icon">${icon}</div>
        <h2 class="kelly-message-title">${title}</h2>
        <p class="kelly-message-text">${message}</p>
        <button class="kelly-message-button ${buttonPrimary ? 'primary' : ''}">${buttonText}</button>
        ${signature ? '<p class="kelly-message-signature">— Kelly</p>' : ''}
      </div>
    `;

    const button = overlay.querySelector('.kelly-message-button');
    button.onclick = close;

    document.body.appendChild(overlay);

    // Focus button for accessibility
    button.focus();

    function close() {
      overlay.remove();
      if (onClose) onClose();
    }

    return { close };
  }

  /**
   * Show a toast message (auto-dismissing)
   */
  function toast(message, duration = 3000) {
    injectStyles();

    const toast = document.createElement('div');
    toast.className = 'kelly-toast';
    toast.textContent = message;

    document.body.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('leaving');
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  // ============================================
  // PRE-BUILT MESSAGES (Kelly's Voice)
  // ============================================

  const messages = {
    // Email sent successfully
    emailSent: (email) => show({
      icon: '📬',
      title: 'Check your email',
      message: `I sent a link to <strong>${email}</strong>.<br><br>Click it and we'll start learning together.`,
      buttonText: 'Got it',
      buttonPrimary: true
    }),

    // Email error
    emailError: () => show({
      icon: '😔',
      title: 'That didn\'t work',
      message: `I couldn't send the email right now. Mind trying again in a moment?<br><br>Or try signing in with Google — it's faster.`,
      buttonText: 'I\'ll try again'
    }),

    // Invalid email
    invalidEmail: () => show({
      icon: '🤔',
      title: 'Hmm, that doesn\'t look right',
      message: `Could you double-check your email address? I want to make sure I can reach you.`,
      buttonText: 'Let me fix that'
    }),

    // Generic error
    somethingWrong: () => show({
      icon: '💭',
      title: 'Something went sideways',
      message: `Technology can be tricky sometimes. Want to try again?`,
      buttonText: 'Try again'
    }),

    // Age gate - too young
    tooYoung: () => show({
      icon: '🌱',
      title: 'Come back when you\'re a bit older',
      message: `Curious Kelly is designed for learners 13 and up. But don't worry — curiosity doesn't have an age limit. Keep asking questions.`,
      buttonText: 'Okay',
      signature: false
    }),

    // Welcome back
    welcomeBack: (name) => show({
      icon: '👋',
      title: `Welcome back${name ? ', ' + name : ''}`,
      message: `Ready to learn something wonderful?`,
      buttonText: 'Let\'s go',
      buttonPrimary: true
    }),

    // Account exists - sign in instead
    accountExists: (email) => show({
      icon: '👋',
      title: 'I remember you',
      message: `Looks like <strong>${email}</strong> already has an account. I sent you a sign-in link.`,
      buttonText: 'Got it'
    }),

    // Offline
    offline: () => show({
      icon: '📡',
      title: 'You seem to be offline',
      message: `I can't reach the internet right now. Check your connection and we'll try again.`,
      buttonText: 'Okay'
    }),

    // Success - generic
    success: (message) => show({
      icon: '✨',
      title: 'Done',
      message: message,
      buttonText: 'Wonderful',
      buttonPrimary: true
    }),

    // Loading state
    loading: (message = 'One moment...') => {
      injectStyles();
      const overlay = document.createElement('div');
      overlay.className = 'kelly-message-overlay';
      overlay.id = 'kelly-loading';
      overlay.innerHTML = `
        <div class="kelly-message-card">
          <div class="kelly-message-icon" style="animation: spin 1s linear infinite;">⏳</div>
          <p class="kelly-message-text">${message}</p>
        </div>
      `;
      document.body.appendChild(overlay);
      
      // Add spin animation
      const style = document.createElement('style');
      style.textContent = '@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }';
      overlay.appendChild(style);
      
      return {
        close: () => overlay.remove()
      };
    }
  };

  // Public API
  return {
    show,
    toast,
    ...messages
  };
})();

// Make available globally
window.KellyMessages = KellyMessages;

// Override native alert with Kelly's voice (optional)
// window.alert = (msg) => KellyMessages.somethingWrong();

