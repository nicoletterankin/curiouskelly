const DEFAULT_HEADERS = {
  'Content-Type': 'application/json',
};

export default class SessionClient {
  constructor(baseUrl = '/api/sessions', userId = 'guest-desktop') {
    this.baseUrl = baseUrl;
    this.userId = userId;
    this.storageKey = 'ck-active-session';
  }

  getStoredSession() {
    try {
      const raw = localStorage.getItem(this.storageKey);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  persistSession(data) {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(data));
    } catch {
      // ignore storage failures
    }
  }

  clearSession() {
    try {
      localStorage.removeItem(this.storageKey);
    } catch {
      // ignore
    }
  }

  async startSession(age, lessonId) {
    try {
      const response = await fetch(`${this.baseUrl}/start`, {
        method: 'POST',
        headers: DEFAULT_HEADERS,
        body: JSON.stringify({
          age,
          lessonId,
          userId: this.userId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Session start failed (${response.status})`);
      }

      const json = await response.json();
      if (json?.data?.sessionId) {
        this.persistSession({ sessionId: json.data.sessionId, lessonId });
      }
      return json.data;
    } catch (error) {
      console.warn('[SessionClient] startSession error', error.message);
      return null;
    }
  }

  async getSession(sessionId) {
    if (!sessionId) return null;
    try {
      const response = await fetch(`${this.baseUrl}/${sessionId}`);
      if (!response.ok) {
        if (response.status === 404) {
          this.clearSession();
        }
        throw new Error(`Session lookup failed (${response.status})`);
      }
      const json = await response.json();
      return json.data;
    } catch (error) {
      console.warn('[SessionClient] getSession error', error.message);
      return null;
    }
  }

  async updateProgress(sessionId, payload) {
    if (!sessionId) return null;
    try {
      const response = await fetch(`${this.baseUrl}/${sessionId}/progress`, {
        method: 'POST',
        headers: DEFAULT_HEADERS,
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(`Progress update failed (${response.status})`);
      }
      const json = await response.json();
      return json.data;
    } catch (error) {
      console.warn('[SessionClient] updateProgress error', error.message);
      return null;
    }
  }

  async completeSession(sessionId) {
    if (!sessionId) return null;
    try {
      const response = await fetch(`${this.baseUrl}/${sessionId}/complete`, {
        method: 'POST',
        headers: DEFAULT_HEADERS,
      });
      if (!response.ok) {
        throw new Error(`Complete failed (${response.status})`);
      }
      const json = await response.json();
      this.clearSession();
      return json.data;
    } catch (error) {
      console.warn('[SessionClient] completeSession error', error.message);
      return null;
    }
  }

  async fetchHistory(limit = 30) {
    try {
      const response = await fetch(`${this.baseUrl}/history/${this.userId}?limit=${limit}`);
      if (!response.ok) {
        throw new Error(`History fetch failed (${response.status})`);
      }
      const json = await response.json();
      return json.data?.history || [];
    } catch (error) {
      console.warn('[SessionClient] fetchHistory error', error.message);
      return [];
    }
  }
}




