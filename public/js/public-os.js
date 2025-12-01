/**
 * Curious Kelly Public OS
 * Handles the window management for the marketing site.
 */

class PublicOS {
    constructor() {
        this.activeWindow = null;
        this.windows = {};
        this.init();
    }

    init() {
        // Cache Windows
        document.querySelectorAll('.os-window').forEach(win => {
            this.windows[win.id] = win;
        });

        // Dock Listeners
        document.querySelectorAll('[data-window]').forEach(trigger => {
            trigger.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = trigger.dataset.window;
                this.openWindow(targetId);
            });
        });

        // Close Listeners
        document.querySelectorAll('.btn-close').forEach(btn => {
            btn.addEventListener('click', () => {
                const win = btn.closest('.os-window');
                this.closeWindow(win.id);
            });
        });

        // Key Listener (ESC)
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.activeWindow) {
                this.closeWindow(this.activeWindow);
            }
        });

        console.log('✨ Curious Kelly OS (Public) Initialized');
    }

    openWindow(id) {
        const win = this.windows[id];
        if (!win) return;

        if (this.activeWindow && this.activeWindow !== id) {
            this.closeWindow(this.activeWindow);
        }

        win.classList.add('open');
        this.activeWindow = id;

        // Update Dock State
        document.querySelectorAll('.dock-icon').forEach(icon => {
            if (icon.dataset.window === id) icon.classList.add('active');
            else icon.classList.remove('active');
        });
    }

    closeWindow(id) {
        const win = this.windows[id];
        if (!win) return;

        win.classList.remove('open');
        this.activeWindow = null;

        // Clear Dock State
        document.querySelectorAll('.dock-icon').forEach(icon => {
            icon.classList.remove('active');
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.os = new PublicOS();
});












