export default class StateManager {
  constructor(initialState = {}) {
    this.state = { ...initialState };
    this.listeners = new Set();
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  getState() {
    return { ...this.state };
  }

  setState(partial) {
    const prevState = this.state;
    this.state = { ...prevState, ...partial };
    this.listeners.forEach((listener) => listener(this.getState(), prevState));
  }
}




