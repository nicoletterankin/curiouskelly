import http from 'node:http';
import { mkdir, appendFile } from 'node:fs/promises';
import path from 'node:path';

const port = Number.parseInt(process.env.CRM_MOCK_PORT ?? '8787', 10);
const dataDir = path.resolve(process.cwd(), '.data');
const logFile = path.join(dataDir, 'mock-crm.log');

async function ensureDataDir() {
  await mkdir(dataDir, { recursive: true });
}

async function logPayload(payload) {
  await ensureDataDir();
  const timestamp = new Date().toISOString();
  await appendFile(logFile, `${timestamp} ${JSON.stringify(payload)}\n`);
}

const server = http.createServer(async (req, res) => {
  if (req.method === 'POST' && req.url === '/api/mock-crm') {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
    });
    req.on('end', async () => {
      try {
        const payload = JSON.parse(body || '{}');
        await logPayload(payload);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'ok', received: true }));
      } catch (error) {
        console.error('[mock-crm] Error parsing payload', error);
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'error', message: error.message }));
      }
    });
    return;
  }

  if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'healthy' }));
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ status: 'not-found' }));
});

server.listen(port, () => {
  console.log(`[mock-crm] listening on http://localhost:${port}`);
});





