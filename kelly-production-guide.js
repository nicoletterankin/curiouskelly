// Kelly Production Guide - Interactive Features

// Tab Switching
function switchTab(tabId) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => content.classList.remove('active'));
    
    // Deactivate all tab buttons
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(button => button.classList.remove('active'));
    
    // Show selected tab content
    const selectedTab = document.getElementById(tabId);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    
    // Activate corresponding button
    const selectedButton = document.querySelector(`[data-tab="${tabId}"]`);
    if (selectedButton) {
        selectedButton.classList.add('active');
    }
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
    
    // Save current tab to localStorage
    localStorage.setItem('kellyGuideCurrentTab', tabId);
}

// Initialize tabs
document.addEventListener('DOMContentLoaded', () => {
    // Set up tab button click handlers
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            switchTab(tabId);
        });
    });
    
    // Restore last viewed tab from localStorage
    const savedTab = localStorage.getItem('kellyGuideCurrentTab');
    if (savedTab && document.getElementById(savedTab)) {
        switchTab(savedTab);
    } else {
        switchTab('tab-assets'); // Default to first tab
    }
    
    // Initialize checkbox persistence
    initCheckboxPersistence();
});

// Checkbox Persistence
function initCheckboxPersistence() {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    
    checkboxes.forEach(checkbox => {
        // Generate unique ID if not present
        if (!checkbox.id) {
            const uniqueId = 'checkbox-' + Math.random().toString(36).substr(2, 9);
            checkbox.id = uniqueId;
        }
        
        // Load saved state
        const savedState = localStorage.getItem(checkbox.id);
        if (savedState === 'true') {
            checkbox.checked = true;
        }
        
        // Save state on change
        checkbox.addEventListener('change', () => {
            localStorage.setItem(checkbox.id, checkbox.checked);
            updateProgress();
        });
    });
}

// Progress Tracking
function updateProgress() {
    // Count total and checked checkboxes
    const allCheckboxes = document.querySelectorAll('input[type="checkbox"]');
    const checkedCheckboxes = document.querySelectorAll('input[type="checkbox"]:checked');
    
    const progress = {
        total: allCheckboxes.length,
        completed: checkedCheckboxes.length,
        percentage: Math.round((checkedCheckboxes.length / allCheckboxes.length) * 100)
    };
    
    // Save to localStorage
    localStorage.setItem('kellyGuideProgress', JSON.stringify(progress));
    
    // Dispatch custom event for other components
    const event = new CustomEvent('progressUpdated', { detail: progress });
    document.dispatchEvent(event);
}

// Export Progress Report
function exportProgress() {
    const progress = {
        timestamp: new Date().toISOString(),
        currentTab: localStorage.getItem('kellyGuideCurrentTab'),
        checkboxStates: {},
        overallProgress: JSON.parse(localStorage.getItem('kellyGuideProgress') || '{}')
    };
    
    // Collect all checkbox states
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        if (checkbox.id) {
            progress.checkboxStates[checkbox.id] = checkbox.checked;
        }
    });
    
    // Create download
    const dataStr = JSON.stringify(progress, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `kelly-production-progress-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// Reset Progress
function resetProgress() {
    if (confirm('Are you sure you want to reset all progress? This cannot be undone.')) {
        // Clear localStorage
        const keysToRemove = [];
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key.startsWith('checkbox-') || key === 'kellyGuideProgress' || key === 'kellyGuideCurrentTab') {
                keysToRemove.push(key);
            }
        }
        
        keysToRemove.forEach(key => localStorage.removeItem(key));
        
        // Uncheck all checkboxes
        const checkboxes = document.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => checkbox.checked = false);
        
        // Reset to first tab
        switchTab('tab-assets');
        
        alert('Progress has been reset!');
    }
}

// Print Current Tab
function printCurrentTab() {
    window.print();
}

// Keyboard Shortcuts
document.addEventListener('keydown', (e) => {
    // Alt + Arrow Left/Right to navigate tabs
    if (e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
        e.preventDefault();
        
        const tabs = ['tab-assets', 'tab-cc5', 'tab-hair', 'tab-iclone', 'tab-tts', 'tab-export'];
        const currentTab = document.querySelector('.tab-content.active').id;
        const currentIndex = tabs.indexOf(currentTab);
        
        if (e.key === 'ArrowLeft' && currentIndex > 0) {
            switchTab(tabs[currentIndex - 1]);
        } else if (e.key === 'ArrowRight' && currentIndex < tabs.length - 1) {
            switchTab(tabs[currentIndex + 1]);
        }
    }
    
    // Alt + P to print
    if (e.altKey && e.key === 'p') {
        e.preventDefault();
        printCurrentTab();
    }
    
    // Alt + E to export
    if (e.altKey && e.key === 'e') {
        e.preventDefault();
        exportProgress();
    }
});

// Auto-save Progress (every 30 seconds)
setInterval(() => {
    updateProgress();
}, 30000);

// Window unload - save progress one last time
window.addEventListener('beforeunload', () => {
    updateProgress();
});

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// Console helper for debugging
console.log('🎬 Kelly Production Guide loaded successfully!');
console.log('Keyboard shortcuts:');
console.log('  Alt + ← / → : Navigate tabs');
console.log('  Alt + P : Print current tab');
console.log('  Alt + E : Export progress');


