(() => {
  const SOURCE = 'kelly-webgl';
  const DEFAULT_DELAY = 600;

  const post = (type, details = {}) => {
    const payload = {
      source: SOURCE,
      type,
      status: details.status ?? 'ok',
      lessonId: details.lessonId ?? null,
      message: details.message ?? null
    };
    window.parent?.postMessage(payload, '*');
  };

  const handleLoad = (event) => {
    if (!event?.data || event.data.destination !== SOURCE) {
      return;
    }

    const { type, payload } = event.data;
    if (type === 'kelly-load') {
      const lessonId = payload?.lessonId ?? 'demo';
      post('kelly-loading', { lessonId, status: 'pending' });
      setTimeout(() => post('kelly-playing', { lessonId }), DEFAULT_DELAY);
    } else if (type === 'kelly-stop') {
      post('kelly-stopped', { lessonId: payload?.lessonId ?? 'demo' });
    } else if (type === 'kelly-ping') {
      post('kelly-pong', {});
    }
  };

  window.addEventListener('message', handleLoad);
  window.addEventListener('load', () => {
    post('kelly-ready', {});
  });
})();




