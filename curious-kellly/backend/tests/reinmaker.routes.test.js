const http = require('http');
const express = require('express');
const assert = require('assert');
const manifestRoute = require('../src/api/reinmaker/manifest.route');
const questsRoute = require('../src/api/reinmaker/quests.route');
const buildManifest = require('../src/services/reinmaker/buildManifest');

function httpRequest({ port, path, method = 'GET' }) {
  return new Promise((resolve, reject) => {
    const req = http.request(
      { port, path, method, headers: { 'Content-Type': 'application/json' } },
      (res) => {
        let data = '';
        res.on('data', (chunk) => {
          data += chunk;
        });
        res.on('end', () => {
          resolve({ statusCode: res.statusCode, body: data });
        });
      }
    );

    req.on('error', reject);
    req.end();
  });
}

(async () => {
  let server;
  try {
    process.env.FEATURES_REINMAKER = '1';

    const manifest = await buildManifest({ writeFile: true });
    assert(Array.isArray(manifest.quests) && manifest.quests.length > 0);

    const app = express();
    app.use('/api/reinmaker/manifest', manifestRoute);
    app.use('/api/reinmaker/quests', questsRoute);

    await new Promise((resolve) => {
      server = app.listen(0, resolve);
    });

    const { port } = server.address();

    const manifestResponse = await httpRequest({ port, path: '/api/reinmaker/manifest' });
    assert.strictEqual(manifestResponse.statusCode, 200, 'Manifest route did not return 200.');
    const manifestPayload = JSON.parse(manifestResponse.body);
    assert.strictEqual(manifestPayload.status, 'ok');
    assert(manifestPayload.manifest);
    const questId = manifestPayload.manifest.quests[0]?.id;
    assert(questId, 'Manifest response did not include any quest ids.');

    const questResponse = await httpRequest({ port, path: `/api/reinmaker/quests/${questId}` });
    assert.strictEqual(questResponse.statusCode, 200, 'Quest route did not return 200.');
    const questPayload = JSON.parse(questResponse.body);
    assert.strictEqual(questPayload.status, 'ok');
    assert.strictEqual(questPayload.quest.id, questId);

    console.log('✅ Reinmaker route tests passed.');
    server.close(() => process.exit(0));
  } catch (error) {
    console.error('❌ Reinmaker route tests failed:', error);
    if (server) {
      server.close(() => process.exit(1));
    } else {
      process.exit(1);
    }
  }
})();













