// Deployment Dashboard - Interactive Features & Progress Tracking

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
    initializeKnownAssets(); // Pre-check boxes for existing assets
    calculateLaunchCountdown();
    updateAllProgress();
    restoreState();
    startAutoSave();
    updateLastUpdated();
});

// Initialize Dashboard Components
function initializeDashboard() {
    // Set up phase toggle handlers
    const phaseHeaders = document.querySelectorAll('.phase-header');
    phaseHeaders.forEach(header => {
        // Phase toggle is already set via onclick in HTML
    });

    // Set up workflow toggle handlers
    const workflowHeaders = document.querySelectorAll('.workflow-header');
    workflowHeaders.forEach(header => {
        // Workflow toggle is already set via onclick in HTML
    });

    // Set up checkbox change handlers
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        // Assign unique ID if not present
        if (!checkbox.id) {
            const uniqueId = 'task-' + Math.random().toString(36).substr(2, 9);
            checkbox.id = uniqueId;
        }

        // Load saved state
        const savedState = localStorage.getItem(`dashboard-${checkbox.id}`);
        if (savedState === 'true') {
            checkbox.checked = true;
        }

        // Save state on change
        checkbox.addEventListener('change', () => {
            localStorage.setItem(`dashboard-${checkbox.id}`, checkbox.checked);
            saveState();
        });
    });
}

// Calculate Launch Countdown
function calculateLaunchCountdown() {
    const launchDate = new Date('2025-11-15T00:00:00');
    const now = new Date();
    const diff = launchDate - now;
    const daysRemaining = Math.ceil(diff / (1000 * 60 * 60 * 24));

    const daysElement = document.getElementById('days-to-launch');
    if (daysElement) {
        if (daysRemaining > 0) {
            daysElement.textContent = daysRemaining;
        } else if (daysRemaining === 0) {
            daysElement.textContent = 'TODAY!';
        } else {
            daysElement.textContent = 'Launched!';
        }
    }
}

// Toggle Phase Content
function togglePhase(phaseId) {
    const phaseContent = document.getElementById(`${phaseId}-content`);
    if (phaseContent) {
        phaseContent.classList.toggle('active');
        saveState();
    }
}

// Toggle All Phases
function toggleAllPhases() {
    const allPhases = document.querySelectorAll('.phase-content');
    const anyOpen = Array.from(allPhases).some(phase => phase.classList.contains('active'));

    allPhases.forEach(phase => {
        if (anyOpen) {
            phase.classList.remove('active');
        } else {
            phase.classList.add('active');
        }
    });
    saveState();
}

// Toggle Workflow Content
function toggleWorkflow(workflowId) {
    const workflowContent = document.getElementById(`${workflowId}-content`);
    if (workflowContent) {
        workflowContent.classList.toggle('active');
        saveState();
    }
}

// Update Phase Progress
function updatePhaseProgress(phaseId) {
    const phaseCheckboxes = document.querySelectorAll(`.phase-task[data-phase="${phaseId}"]`);
    const checkedBoxes = Array.from(phaseCheckboxes).filter(cb => cb.checked);
    const percentage = phaseCheckboxes.length > 0 
        ? Math.round((checkedBoxes.length / phaseCheckboxes.length) * 100)
        : 0;

    // Update progress bar
    const progressFill = document.getElementById(`${phaseId}-progress`);
    if (progressFill) {
        progressFill.style.width = `${percentage}%`;
    }

    // Update percentage text
    const progressText = document.getElementById(`${phaseId}-percent`);
    if (progressText) {
        progressText.textContent = `${percentage}%`;
    }

    // Update overall progress
    updateOverallProgress();
    saveState();
}

// Update Workflow Progress
function updateWorkflowProgress() {
    const workflows = ['workflow-assets', 'workflow-cc5', 'workflow-hair', 'workflow-iclone', 'workflow-tts', 'workflow-export'];

    workflows.forEach(workflowId => {
        const workflowCheckboxes = document.querySelectorAll(`.workflow-check[data-workflow="${workflowId}"]`);
        const checkedBoxes = Array.from(workflowCheckboxes).filter(cb => cb.checked);

        // Update workflow progress display
        const progressElement = document.getElementById(`${workflowId}-progress`);
        if (progressElement) {
            progressElement.textContent = `${checkedBoxes.length}/${workflowCheckboxes.length}`;
        }
    });

    // Update overall workflow progress text
    const allWorkflowChecks = document.querySelectorAll('.workflow-check');
    const allChecked = Array.from(allWorkflowChecks).filter(cb => cb.checked);
    const workflowProgressText = document.getElementById('workflow-progress-text');
    if (workflowProgressText) {
        workflowProgressText.textContent = `${allChecked.length} of ${allWorkflowChecks.length} steps completed`;
    }

    updateOverallProgress();
    saveState();
}

// Update Overall Progress
function updateOverallProgress() {
    const allCheckboxes = document.querySelectorAll('input[type="checkbox"]');
    const checkedCheckboxes = Array.from(allCheckboxes).filter(cb => cb.checked);
    const percentage = allCheckboxes.length > 0
        ? Math.round((checkedCheckboxes.length / allCheckboxes.length) * 100)
        : 0;

    // Update overall progress metric
    const overallProgressElement = document.getElementById('overall-progress');
    if (overallProgressElement) {
        overallProgressElement.textContent = `${percentage}%`;
    }

    // Update task counts
    const tasksCompletedElement = document.getElementById('tasks-completed');
    const tasksTotalElement = document.getElementById('tasks-total');
    if (tasksCompletedElement) {
        tasksCompletedElement.textContent = checkedCheckboxes.length;
    }
    if (tasksTotalElement) {
        tasksTotalElement.textContent = allCheckboxes.length;
    }
}

// Update All Progress (on load)
function updateAllProgress() {
    // Update phase progress
    ['phase1', 'phase2', 'phase3', 'phase4', 'phase5'].forEach(phaseId => {
        updatePhaseProgress(phaseId);
    });

    // Update workflow progress
    updateWorkflowProgress();

    // Update overall progress
    updateOverallProgress();

    // Update rendering progress (example - would be updated from actual data)
    updateRenderingProgress();
}

// Update Rendering Progress
function updateRenderingProgress() {
    // Check actual file existence (would connect to file system in production)
    // For now, use localStorage or default to 0
    let videosRendered = parseInt(localStorage.getItem('dashboard-videos-rendered')) || 0;
    
    // Check if test render exists (user can manually update this)
    const testRenderExists = localStorage.getItem('dashboard-test-render-complete') === 'true';
    if (testRenderExists && videosRendered === 0) {
        videosRendered = 1;
    }

    const videosRenderedElement = document.getElementById('videos-rendered');
    if (videosRenderedElement) {
        videosRenderedElement.textContent = `${videosRendered}/180`;
    }

    const renderProgress = document.getElementById('render-progress');
    const videosDone = document.getElementById('videos-done');
    if (renderProgress && videosDone) {
        const percentage = Math.round((videosRendered / 180) * 100);
        renderProgress.style.width = `${percentage}%`;
        videosDone.textContent = videosRendered;
    }

    // Save to localStorage for persistence
    localStorage.setItem('dashboard-videos-rendered', videosRendered);
}

// Initialize with known assets (2 audio files ready, 14 demo files ready)
function initializeKnownAssets() {
    // Mark audio prep as complete (files exist)
    const audioCheckboxes = document.querySelectorAll('.workflow-check[data-workflow="workflow-assets"]');
    if (audioCheckboxes.length >= 3) {
        // Check boxes for: ElevenLabs setup, test audio generation, verify quality
        audioCheckboxes[2].checked = true; // ElevenLabs access (user has it)
        audioCheckboxes[3].checked = true; // Test audio generated (kelly25_audio.wav exists)
        audioCheckboxes[4].checked = true; // Audio quality verified
    }
    
    // Mark hair physics as available
    const hairCheckboxes = document.querySelectorAll('.workflow-check[data-workflow="workflow-hair"]');
    if (hairCheckboxes.length >= 4) {
        hairCheckboxes[3].checked = true; // Import physics preset (file exists)
    }
    
    // Mark director's chair assets as ready
    const icloneCheckboxes = document.querySelectorAll('.workflow-check[data-workflow="workflow-iclone"]');
    if (icloneCheckboxes.length >= 3) {
        icloneCheckboxes[2].checked = true; // Chair backgrounds available
    }
}

// Save Dashboard State
function saveState() {
    const state = {
        timestamp: new Date().toISOString(),
        expandedPhases: [],
        expandedWorkflows: [],
        checkboxStates: {}
    };

    // Save expanded phases
    const phaseContents = document.querySelectorAll('.phase-content');
    phaseContents.forEach(content => {
        if (content.classList.contains('active')) {
            state.expandedPhases.push(content.id);
        }
    });

    // Save expanded workflows
    const workflowContents = document.querySelectorAll('.workflow-content');
    workflowContents.forEach(content => {
        if (content.classList.contains('active')) {
            state.expandedWorkflows.push(content.id);
        }
    });

    // Save checkbox states
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        if (checkbox.id) {
            state.checkboxStates[checkbox.id] = checkbox.checked;
        }
    });

    localStorage.setItem('dashboard-state', JSON.stringify(state));
    updateLastUpdated();
}

// Restore Dashboard State
function restoreState() {
    const savedState = localStorage.getItem('dashboard-state');
    if (!savedState) return;

    try {
        const state = JSON.parse(savedState);

        // Restore expanded phases
        state.expandedPhases?.forEach(phaseId => {
            const phase = document.getElementById(phaseId);
            if (phase) {
                phase.classList.add('active');
            }
        });

        // Restore expanded workflows
        state.expandedWorkflows?.forEach(workflowId => {
            const workflow = document.getElementById(workflowId);
            if (workflow) {
                workflow.classList.add('active');
            }
        });

        // Restore checkbox states
        Object.entries(state.checkboxStates || {}).forEach(([id, checked]) => {
            const checkbox = document.getElementById(id);
            if (checkbox) {
                checkbox.checked = checked;
            }
        });

        // Restore videos rendered count
        const videosRendered = localStorage.getItem('dashboard-videos-rendered');
        if (videosRendered) {
            const element = document.getElementById('videos-rendered');
            if (element) {
                element.textContent = `${videosRendered}/180`;
            }
        }
    } catch (error) {
        console.error('Error restoring dashboard state:', error);
    }
}

// Export Dashboard Report
function exportDashboard() {
    const report = {
        exportDate: new Date().toISOString(),
        launchCountdown: document.getElementById('days-to-launch')?.textContent || 'N/A',
        overallProgress: document.getElementById('overall-progress')?.textContent || '0%',
        videosRendered: document.getElementById('videos-rendered')?.textContent || '0/180',
        phases: {},
        workflows: {},
        checkboxStates: {}
    };

    // Export phase progress
    ['phase1', 'phase2', 'phase3', 'phase4', 'phase5'].forEach(phaseId => {
        const checkboxes = document.querySelectorAll(`.phase-task[data-phase="${phaseId}"]`);
        const checked = Array.from(checkboxes).filter(cb => cb.checked).length;
        report.phases[phaseId] = {
            total: checkboxes.length,
            completed: checked,
            percentage: checkboxes.length > 0 ? Math.round((checked / checkboxes.length) * 100) : 0
        };
    });

    // Export workflow progress
    ['workflow-assets', 'workflow-cc5', 'workflow-hair', 'workflow-iclone', 'workflow-tts', 'workflow-export'].forEach(workflowId => {
        const checkboxes = document.querySelectorAll(`.workflow-check[data-workflow="${workflowId}"]`);
        const checked = Array.from(checkboxes).filter(cb => cb.checked).length;
        report.workflows[workflowId] = {
            total: checkboxes.length,
            completed: checked
        };
    });

    // Export all checkbox states
    const allCheckboxes = document.querySelectorAll('input[type="checkbox"]');
    allCheckboxes.forEach(checkbox => {
        if (checkbox.id) {
            report.checkboxStates[checkbox.id] = checkbox.checked;
        }
    });

    // Create and download JSON file
    const dataStr = JSON.stringify(report, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `kelly-deployment-report-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    alert('✅ Dashboard report exported successfully!');
}

// Print Dashboard
function printDashboard() {
    // Expand all phases and workflows before printing
    const allPhases = document.querySelectorAll('.phase-content');
    const allWorkflows = document.querySelectorAll('.workflow-content');

    allPhases.forEach(phase => phase.classList.add('active'));
    allWorkflows.forEach(workflow => workflow.classList.add('active'));

    // Print
    window.print();

    // Note: Collapsed state is preserved by CSS, so manual restore not needed
}

// Reset Dashboard
function resetDashboard() {
    if (!confirm('⚠️ Are you sure you want to reset ALL progress? This cannot be undone!')) {
        return;
    }

    // Clear all checkboxes
    const allCheckboxes = document.querySelectorAll('input[type="checkbox"]');
    allCheckboxes.forEach(checkbox => {
        checkbox.checked = false;
    });

    // Clear localStorage
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key.startsWith('dashboard-')) {
            keysToRemove.push(key);
        }
    }
    keysToRemove.forEach(key => localStorage.removeItem(key));

    // Collapse all phases and workflows
    const allPhases = document.querySelectorAll('.phase-content');
    const allWorkflows = document.querySelectorAll('.workflow-content');
    allPhases.forEach(phase => phase.classList.remove('active'));
    allWorkflows.forEach(workflow => workflow.classList.remove('active'));

    // Update all progress displays
    updateAllProgress();

    alert('✅ Dashboard has been reset!');
}

// Auto-save every 30 seconds
function startAutoSave() {
    setInterval(() => {
        saveState();
    }, 30000); // 30 seconds
}

// Update Last Updated Timestamp
function updateLastUpdated() {
    const lastUpdatedElement = document.getElementById('last-updated');
    if (lastUpdatedElement) {
        const now = new Date();
        lastUpdatedElement.textContent = now.toLocaleString();
    }
}

// Window unload - save one last time
window.addEventListener('beforeunload', () => {
    saveState();
});

// Keyboard Shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + E to export
    if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault();
        exportDashboard();
    }

    // Ctrl/Cmd + P to print
    if ((e.ctrlKey || e.metaKey) && e.key === 'p') {
        e.preventDefault();
        printDashboard();
    }
});

// Update metrics periodically (for launch countdown)
setInterval(() => {
    calculateLaunchCountdown();
}, 60000); // Update every minute

// Initialize post-launch metrics (placeholder - would connect to real API)
function initializePostLaunchMetrics() {
    // These would be populated from a real backend API post-launch
    const metrics = {
        totalSignups: 0,
        payingSubscribers: 0,
        completionRate: 0,
        sessionDuration: 0,
        uptime: 100,
        pressMentions: 0
    };

    // Update DOM
    document.getElementById('total-signups').textContent = metrics.totalSignups || '--';
    document.getElementById('paying-subs').textContent = metrics.payingSubscribers || '--';
    document.getElementById('completion-rate').textContent = metrics.completionRate ? `${metrics.completionRate}%` : '--%';
    document.getElementById('session-duration').textContent = metrics.sessionDuration ? `${metrics.sessionDuration} min` : '-- min';
    document.getElementById('uptime').textContent = metrics.uptime ? `${metrics.uptime}%` : '--%';
    document.getElementById('press-mentions').textContent = metrics.pressMentions || '--';
}

// Update daily users metric
function updateDailyUsers() {
    const launchDate = new Date('2025-11-15T00:00:00');
    const now = new Date();
    
    const dailyUsersElement = document.getElementById('daily-users');
    if (dailyUsersElement) {
        if (now >= launchDate) {
            // Post-launch: would fetch from API
            dailyUsersElement.textContent = '--';
        } else {
            dailyUsersElement.textContent = 'Pre-Launch';
        }
    }
}

// Console helper
console.log('🚀 Kelly Deployment Dashboard loaded successfully!');
console.log('Keyboard shortcuts:');
console.log('  Ctrl/Cmd + E : Export dashboard report');
console.log('  Ctrl/Cmd + P : Print dashboard');
console.log('Auto-save: Every 30 seconds');

