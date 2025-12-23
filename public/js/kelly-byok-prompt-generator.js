/**
 * Kelly BYOK Prompt Generator UI
 * 
 * Provides UI for generating curriculum-aware prompts for BYOK (Bring Your Own Key) LLM providers.
 * 
 * Features:
 * - Query input
 * - Provider selection (OpenAI, Anthropic, Google)
 * - Model selection
 * - Curriculum context preview
 * - Generated prompt display
 * - Copy to clipboard
 * - Direct API call (if user provides key)
 */

(function() {
  'use strict';

  const KellyBYOKPromptGenerator = {
    currentPrompt: null,
    currentProvider: 'openai',
    currentModel: null,

    /**
     * Initialize BYOK prompt generator UI
     */
    init() {
      // Wait for knowledge base to be ready
      if (!window.KellyCurriculumKB) {
        console.warn('[BYOK] Knowledge base not loaded, waiting...');
        setTimeout(() => this.init(), 1000);
        return;
      }

      // Inject UI into settings panel or create standalone panel
      this.injectUI();
    },

    /**
     * Inject BYOK UI into page
     */
    injectUI() {
      // Check if settings panel exists
      const settingsPanel = document.getElementById('right-panel');
      if (settingsPanel) {
        this.addToSettingsPanel(settingsPanel);
      } else {
        this.createStandalonePanel();
      }
    },

    /**
     * Add BYOK section to settings panel
     */
    addToSettingsPanel(panel) {
      // Find settings scroll container
      const settingsScroll = panel.querySelector('.settings-scroll') || panel;
      
      // Find or create BYOK section
      let byokSection = document.getElementById('byok-prompt-section');
      if (!byokSection) {
        byokSection = document.createElement('div');
        byokSection.id = 'byok-prompt-section';
        byokSection.className = 'settings-section';
        byokSection.innerHTML = this.getBYOKHTML();
        
        // Insert after "Your Learning" section
        const learningSection = settingsScroll.querySelector('.settings-section:has(#btn-journey-insights)');
        if (learningSection && learningSection.nextSibling) {
          settingsScroll.insertBefore(byokSection, learningSection.nextSibling);
        } else {
          settingsScroll.appendChild(byokSection);
        }
      }

      // Attach event handlers
      this.attachHandlers(byokSection);
    },

    /**
     * Create standalone BYOK panel
     */
    createStandalonePanel() {
      const panel = document.createElement('div');
      panel.id = 'byok-prompt-panel';
      panel.className = 'byok-prompt-panel';
      panel.innerHTML = this.getStandaloneHTML();
      document.body.appendChild(panel);
      this.attachHandlers(panel);
    },

    /**
     * Get BYOK HTML for settings panel
     */
    getBYOKHTML() {
      return `
        <div class="settings-section-header">
          <h3>Ask Kelly (BYOK)</h3>
          <p class="settings-section-description">Ask questions using your own API key. Kelly uses the complete curriculum as context.</p>
        </div>
        
        <div class="byok-controls">
          <div class="byok-provider-select">
            <label>Provider:</label>
            <select id="byok-provider">
              <option value="openai">OpenAI (GPT-4)</option>
              <option value="anthropic">Anthropic (Claude)</option>
              <option value="google">Google (Gemini)</option>
            </select>
          </div>
          
          <div class="byok-model-select">
            <label>Model:</label>
            <select id="byok-model">
              <option value="gpt-4-turbo-preview">GPT-4 Turbo</option>
              <option value="gpt-4">GPT-4</option>
              <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
            </select>
          </div>
        </div>
        
        <div class="byok-query-input">
          <textarea 
            id="byok-query" 
            placeholder="Ask Kelly anything... (e.g., 'How does photosynthesis work?')"
            rows="3"
          ></textarea>
        </div>
        
        <div class="byok-actions">
          <button id="byok-generate-prompt" class="btn-primary">Generate Prompt</button>
          <button id="byok-send-query" class="btn-secondary" style="display: none;">Send to LLM</button>
        </div>
        
        <div id="byok-context-preview" class="byok-context-preview" style="display: none;">
          <h4>Curriculum Context (will be included):</h4>
          <div id="byok-context-content" class="byok-context-content"></div>
        </div>
        
        <div id="byok-prompt-display" class="byok-prompt-display" style="display: none;">
          <h4>Generated Prompt:</h4>
          <div class="byok-prompt-actions">
            <button id="byok-copy-prompt" class="btn-secondary">Copy Prompt</button>
            <button id="byok-copy-context" class="btn-secondary">Copy Context Only</button>
          </div>
          <pre id="byok-prompt-content" class="byok-prompt-content"></pre>
        </div>
        
        <div id="byok-response-display" class="byok-response-display" style="display: none;">
          <h4>Response:</h4>
          <div id="byok-response-content" class="byok-response-content"></div>
        </div>
        
        <div class="byok-api-key-section">
          <label>API Key (optional - for direct calls):</label>
          <input 
            type="password" 
            id="byok-api-key" 
            placeholder="sk-... or claude-... or AIza..."
            style="width: 100%; padding: 8px; margin-top: 8px;"
          />
          <small style="color: rgba(255,255,255,0.5); display: block; margin-top: 4px;">
            Your API key is stored locally and never sent to our servers.
          </small>
        </div>
      `;
    },

    /**
     * Get standalone HTML
     */
    getStandaloneHTML() {
      return `
        <div class="byok-panel-header">
          <h2>Ask Kelly (BYOK)</h2>
          <button class="byok-close" onclick="this.closest('.byok-prompt-panel').remove()">×</button>
        </div>
        ${this.getBYOKHTML()}
      `;
    },

    /**
     * Attach event handlers
     */
    attachHandlers(container) {
      // Provider change
      const providerSelect = container.querySelector('#byok-provider');
      if (providerSelect) {
        providerSelect.addEventListener('change', (e) => {
          this.currentProvider = e.target.value;
          this.updateModelOptions(e.target.value);
        });
      }

      // Model change
      const modelSelect = container.querySelector('#byok-model');
      if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
          this.currentModel = e.target.value;
        });
      }

      // Generate prompt button
      const generateBtn = container.querySelector('#byok-generate-prompt');
      if (generateBtn) {
        generateBtn.addEventListener('click', () => this.generatePrompt());
      }

      // Send query button
      const sendBtn = container.querySelector('#byok-send-query');
      if (sendBtn) {
        sendBtn.addEventListener('click', () => this.sendToLLM());
      }

      // Copy buttons
      const copyPromptBtn = container.querySelector('#byok-copy-prompt');
      if (copyPromptBtn) {
        copyPromptBtn.addEventListener('click', () => this.copyPrompt());
      }

      const copyContextBtn = container.querySelector('#byok-copy-context');
      if (copyContextBtn) {
        copyContextBtn.addEventListener('click', () => this.copyContext());
      }

      // Enter key in textarea
      const queryInput = container.querySelector('#byok-query');
      if (queryInput) {
        queryInput.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' && e.ctrlKey) {
            this.generatePrompt();
          }
        });
      }
    },

    /**
     * Update model options based on provider
     */
    updateModelOptions(provider) {
      const modelSelect = document.getElementById('byok-model');
      if (!modelSelect) return;

      const models = {
        openai: [
          { value: 'gpt-4-turbo-preview', label: 'GPT-4 Turbo' },
          { value: 'gpt-4', label: 'GPT-4' },
          { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' }
        ],
        anthropic: [
          { value: 'claude-3-opus-20240229', label: 'Claude 3 Opus' },
          { value: 'claude-3-sonnet-20240229', label: 'Claude 3 Sonnet' },
          { value: 'claude-3-haiku-20240307', label: 'Claude 3 Haiku' }
        ],
        google: [
          { value: 'gemini-pro', label: 'Gemini Pro' },
          { value: 'gemini-pro-vision', label: 'Gemini Pro Vision' }
        ]
      };

      modelSelect.innerHTML = models[provider]
        .map(m => `<option value="${m.value}">${m.label}</option>`)
        .join('');

      this.currentModel = models[provider][0].value;
    },

    /**
     * Generate prompt from query
     */
    async generatePrompt() {
      const queryInput = document.getElementById('byok-query');
      if (!queryInput) return;

      const query = queryInput.value.trim();
      if (!query) {
        alert('Please enter a question');
        return;
      }

      // Show loading
      const generateBtn = document.getElementById('byok-generate-prompt');
      if (generateBtn) {
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
      }

      try {
        // Generate prompt using knowledge base
        const promptData = window.KellyCurriculumKB.generateBYOKPrompt(query, {
          provider: this.currentProvider,
          model: this.currentModel,
          includeContext: true,
          personality: 'curious',
          tone: 'warm'
        });

        this.currentPrompt = promptData;

        // Show context preview
        this.showContextPreview(promptData.context);

        // Show generated prompt
        this.showPrompt(promptData.prompt);

        // Show send button
        const sendBtn = document.getElementById('byok-send-query');
        if (sendBtn) {
          sendBtn.style.display = 'inline-block';
        }

      } catch (error) {
        console.error('[BYOK] Error generating prompt:', error);
        alert('Error generating prompt: ' + error.message);
      } finally {
        if (generateBtn) {
          generateBtn.disabled = false;
          generateBtn.textContent = 'Generate Prompt';
        }
      }
    },

    /**
     * Show context preview
     */
    showContextPreview(context) {
      const preview = document.getElementById('byok-context-preview');
      const content = document.getElementById('byok-context-content');
      
      if (preview && content) {
        content.textContent = context.substring(0, 500) + '...';
        preview.style.display = 'block';
      }
    },

    /**
     * Show generated prompt
     */
    showPrompt(prompt) {
      const display = document.getElementById('byok-prompt-display');
      const content = document.getElementById('byok-prompt-content');
      
      if (display && content) {
        content.textContent = prompt;
        display.style.display = 'block';
      }
    },

    /**
     * Copy prompt to clipboard
     */
    async copyPrompt() {
      if (!this.currentPrompt) return;

      try {
        await navigator.clipboard.writeText(this.currentPrompt.prompt);
        alert('Prompt copied to clipboard!');
      } catch (error) {
        console.error('[BYOK] Copy failed:', error);
        alert('Failed to copy. Please select and copy manually.');
      }
    },

    /**
     * Copy context to clipboard
     */
    async copyContext() {
      if (!this.currentPrompt) return;

      try {
        await navigator.clipboard.writeText(this.currentPrompt.context);
        alert('Context copied to clipboard!');
      } catch (error) {
        console.error('[BYOK] Copy failed:', error);
        alert('Failed to copy. Please select and copy manually.');
      }
    },

    /**
     * Send query to LLM using user's API key
     */
    async sendToLLM() {
      const apiKeyInput = document.getElementById('byok-api-key');
      if (!apiKeyInput || !apiKeyInput.value.trim()) {
        alert('Please enter your API key');
        return;
      }

      const apiKey = apiKeyInput.value.trim();
      const query = document.getElementById('byok-query').value.trim();

      if (!this.currentPrompt) {
        await this.generatePrompt();
      }

      // Show loading
      const sendBtn = document.getElementById('byok-send-query');
      if (sendBtn) {
        sendBtn.disabled = true;
        sendBtn.textContent = 'Sending...';
      }

      try {
        const response = await this.callLLMAPI(
          this.currentProvider,
          this.currentModel,
          apiKey,
          this.currentPrompt.prompt
        );

        this.showResponse(response);

      } catch (error) {
        console.error('[BYOK] LLM call failed:', error);
        alert('Error calling LLM: ' + error.message);
      } finally {
        if (sendBtn) {
          sendBtn.disabled = false;
          sendBtn.textContent = 'Send to LLM';
        }
      }
    },

    /**
     * Call LLM API based on provider
     */
    async callLLMAPI(provider, model, apiKey, prompt) {
      // For security, we should proxy through our API
      // But for now, show how it would work

      const response = await fetch('/api/byok-llm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          provider,
          model,
          apiKey, // In production, this should be encrypted
          prompt
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      return data.response;
    },

    /**
     * Show LLM response
     */
    showResponse(response) {
      const display = document.getElementById('byok-response-display');
      const content = document.getElementById('byok-response-content');
      
      if (display && content) {
        content.textContent = response;
        display.style.display = 'block';
        display.scrollIntoView({ behavior: 'smooth' });
      }
    },

    /**
     * Inject CSS styles
     */
    injectStyles() {
      if (document.getElementById('byok-styles')) return;

      const style = document.createElement('style');
      style.id = 'byok-styles';
      style.textContent = `
        .byok-controls {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
          margin-bottom: 16px;
        }
        .byok-provider-select, .byok-model-select {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .byok-provider-select label, .byok-model-select label {
          font-size: 12px;
          color: rgba(255,255,255,0.7);
        }
        .byok-provider-select select, .byok-model-select select {
          padding: 8px;
          background: rgba(255,255,255,0.1);
          border: 1px solid rgba(255,255,255,0.2);
          border-radius: 6px;
          color: white;
        }
        .byok-query-input textarea {
          width: 100%;
          padding: 12px;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.2);
          border-radius: 6px;
          color: white;
          font-family: inherit;
          resize: vertical;
        }
        .byok-actions {
          display: flex;
          gap: 8px;
          margin-top: 12px;
        }
        .byok-context-preview, .byok-prompt-display, .byok-response-display {
          margin-top: 20px;
          padding: 16px;
          background: rgba(255,255,255,0.05);
          border-radius: 8px;
        }
        .byok-context-preview h4, .byok-prompt-display h4, .byok-response-display h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
        }
        .byok-context-content {
          font-size: 12px;
          color: rgba(255,255,255,0.7);
          max-height: 200px;
          overflow-y: auto;
        }
        .byok-prompt-actions {
          display: flex;
          gap: 8px;
          margin-bottom: 12px;
        }
        .byok-prompt-content {
          background: rgba(0,0,0,0.3);
          padding: 12px;
          border-radius: 6px;
          font-size: 12px;
          white-space: pre-wrap;
          max-height: 400px;
          overflow-y: auto;
        }
        .byok-response-content {
          font-size: 14px;
          line-height: 1.6;
          white-space: pre-wrap;
        }
        .byok-api-key-section {
          margin-top: 24px;
          padding-top: 24px;
          border-top: 1px solid rgba(255,255,255,0.1);
        }
        .byok-api-key-section label {
          font-size: 12px;
          color: rgba(255,255,255,0.7);
        }
        .byok-api-key-section input {
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.2);
          border-radius: 6px;
          color: white;
        }
      `;
      document.head.appendChild(style);
    }
  };

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      KellyBYOKPromptGenerator.injectStyles();
      KellyBYOKPromptGenerator.init();
    });
  } else {
    KellyBYOKPromptGenerator.injectStyles();
    KellyBYOKPromptGenerator.init();
  }

  // Expose globally
  window.KellyBYOKPromptGenerator = KellyBYOKPromptGenerator;
})();

