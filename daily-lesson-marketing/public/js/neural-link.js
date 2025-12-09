/**
 * Neural Link - Privacy-First Context System
 * Handles the "Power Up Kelly" connections (Meta, OpenAI, etc.)
 */

import { supabase } from './auth.js';

// Configuration
const PROVIDERS = {
    META: {
        id: 'meta',
        name: 'Meta (Facebook)',
        icon: 'assets/icons/meta.svg', // Placeholder
        description: 'Kelly learns from your public Likes to use metaphors you understand (e.g. "It\'s like gardening...").',
        scopes: ['public_profile', 'user_likes', 'user_hobbies']
    },
    OPENAI: {
        id: 'openai',
        name: 'OpenAI',
        icon: 'assets/icons/openai.svg', // Placeholder
        description: 'Verify your Plus subscription or bring your own API key for smarter, deeper lessons.',
        scopes: ['openid', 'profile']
    }
};

export class NeuralLink {
    constructor() {
        this.connectedProviders = {};
    }

    /**
     * Initialize the Neural Link system
     * Fetches current connection status from Supabase
     */
    async init() {
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) return;

        // In a real app, we'd fetch this from the 'users' table
        // For now, we'll check local storage or metadata
        // const { data } = await supabase.from('users').select('connectedProviders').single();
        // this.connectedProviders = data?.connectedProviders || {};
        
        console.log('Neural Link Initialized');
        this.renderUI();
    }

    /**
     * Render the "Power Up Kelly" section
     * @param {HTMLElement} container - The container to render into
     */
    renderUI(container) {
        if (!container) return;

        container.innerHTML = `
            <div class="neural-link-card">
                <div class="neural-header">
                    <h3>🧠 Power Up Kelly</h3>
                    <p>Grant Kelly specific knowledge to personalize your lessons. Data is used <strong>only</strong> for context.</p>
                </div>
                <div class="provider-list">
                    ${this._renderProvider(PROVIDERS.META)}
                    ${this._renderProvider(PROVIDERS.OPENAI)}
                </div>
            </div>
        `;

        // Add event listeners
        container.querySelectorAll('.btn-connect').forEach(btn => {
            btn.addEventListener('click', (e) => this.handleConnect(e.target.dataset.provider));
        });
    }

    _renderProvider(provider) {
        const isConnected = this.connectedProviders[provider.id];
        return `
            <div class="provider-item ${isConnected ? 'connected' : ''}">
                <div class="provider-info">
                    <h4>${provider.name}</h4>
                    <p>${provider.description}</p>
                </div>
                <button class="btn-connect ${isConnected ? 'btn-disconnect' : ''}" data-provider="${provider.id}">
                    ${isConnected ? 'Disconnect' : 'Connect'}
                </button>
            </div>
        `;
    }

    async handleConnect(providerId) {
        console.log(`Attempting to connect ${providerId}...`);
        
        if (providerId === 'meta') {
            // Meta / Facebook OAuth Flow for Data Access
            // In production, this would trigger FB.login() with specific scopes
            await this.mockConnect(providerId);
        } else if (providerId === 'openai') {
            // OpenAI Auth Flow
            await this.mockConnect(providerId);
        }
    }

    async mockConnect(providerId) {
        // Simulate API delay
        const btn = document.querySelector(`button[data-provider="${providerId}"]`);
        if(btn) {
            const originalText = btn.textContent;
            btn.textContent = 'Connecting...';
            btn.disabled = true;
        }

        await new Promise(r => setTimeout(r, 1500));

        // Toggle state
        this.connectedProviders[providerId] = !this.connectedProviders[providerId];
        
        // Re-render (in a real app, we'd update the specific DOM or use React state)
        const container = document.getElementById('neural-link-container');
        if (container) this.renderUI(container);

        // Persist to DB (Mock)
        console.log(`Updated ${providerId} status to: ${this.connectedProviders[providerId]}`);
        // await supabase.from('users').update({ connectedProviders: this.connectedProviders }).eq('id', user.id);
    }
}

export const neuralLink = new NeuralLink();
















