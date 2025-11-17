mergeInto(LibraryManager.library, {
  KellyPostMessage: function (messagePtr) {
    if (typeof window === 'undefined') {
      return;
    }
    const message = UTF8ToString(messagePtr);
    try {
      const payload = JSON.parse(message);
      const data = Object.assign({ source: 'kelly-webgl' }, payload);
      const targetOrigin = data.origin || '*';
      window.parent?.postMessage(data, targetOrigin);
    } catch (error) {
      console.error('[Kelly WebGL] Failed to post message to parent', error);
    }
  },

  KellySubscribeToMessages: function (objectPtr, methodPtr) {
    if (typeof window === 'undefined') {
      return;
    }

    if (window.__kellyWebGlListenerAttached) {
      return;
    }

    const objectName = UTF8ToString(objectPtr);
    const methodName = UTF8ToString(methodPtr);

    const handler = function (event) {
      const data = event?.data;
      if (!data || data.destination !== 'kelly-webgl') {
        return;
      }

      if (typeof SendMessage !== 'function') {
        console.warn('[Kelly WebGL] SendMessage not ready yet');
        return;
      }

      try {
        const payload = JSON.stringify(data);
        SendMessage(objectName, methodName, payload);
      } catch (error) {
        console.error('[Kelly WebGL] Failed to relay message to Unity', error);
      }
    };

    window.addEventListener('message', handler);
    window.__kellyWebGlListenerAttached = true;
  }
});

