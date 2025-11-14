const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const analyticsDir = path.join(__dirname, '../../analytics/Kelly');
const latencyFilePath = path.join(analyticsDir, 'voice-latency.csv');

let fileReadyPromise = null;

function hashSessionId(sessionId) {
  if (!sessionId) {
    return '';
  }

  try {
    return crypto
      .createHash('sha256')
      .update(String(sessionId))
      .digest('hex');
  } catch (error) {
    console.error('[voiceMetricsLogger] Failed to hash session id', error);
    return '';
  }
}

async function ensureFileReady() {
  if (!fileReadyPromise) {
    fileReadyPromise = fs.promises
      .mkdir(analyticsDir, { recursive: true })
      .then(() => fs.promises.access(latencyFilePath).catch(async () => {
        const header = [
          'timestamp_iso',
          'session_hash',
          'source',
          'topic',
          'learner_age',
          'age_bucket',
          'kelly_age',
          'kelly_persona',
          'request_started_at',
          'response_sent_at',
          'latency_ms',
          'status'
        ].join(',');
        await fs.promises.writeFile(`${latencyFilePath}`, `${header}\n`, 'utf8');
      }));
  }

  return fileReadyPromise;
}

function sanitizeCsvValue(value) {
  if (value === undefined || value === null) {
    return '';
  }

  const stringValue = String(value);
  if (stringValue.includes(',') || stringValue.includes('\n') || stringValue.includes('"')) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }

  return stringValue;
}

async function logVoiceLatency(metrics) {
  try {
    await ensureFileReady();

    const {
      sessionId,
      source = 'unknown',
      topic = '',
      learnerAge = '',
      ageBucket = '',
      kellyAge = '',
      kellyPersona = '',
      requestStartedAt,
      responseSentAt,
      latencyMs,
      status = 'ok'
    } = metrics;

    const timestampIso = new Date().toISOString();
    const line = [
      sanitizeCsvValue(timestampIso),
      sanitizeCsvValue(hashSessionId(sessionId)),
      sanitizeCsvValue(source),
      sanitizeCsvValue(topic),
      sanitizeCsvValue(learnerAge),
      sanitizeCsvValue(ageBucket),
      sanitizeCsvValue(kellyAge),
      sanitizeCsvValue(kellyPersona),
      sanitizeCsvValue(requestStartedAt),
      sanitizeCsvValue(responseSentAt),
      sanitizeCsvValue(latencyMs),
      sanitizeCsvValue(status)
    ].join(',');

    await fs.promises.appendFile(latencyFilePath, `${line}\n`, 'utf8');
  } catch (error) {
    console.error('[voiceMetricsLogger] Failed to log voice latency', error);
  }
}

module.exports = {
  logVoiceLatency,
  hashSessionId
};















