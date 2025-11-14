// Mock CRM server for local development
import http from 'http';

const PORT = 3001;

const server = http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/api/webhook') {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk.toString();
    });
    req.on('end', () => {
      const data = JSON.parse(body);
      console.log('📥 Mock CRM received lead:', JSON.stringify(data, null, 2));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: true, id: `mock-${Date.now()}` }));
    });
  } else {
    res.writeHead(404);
    res.end('Not found');
  }
});

server.listen(PORT, () => {
  console.log(`🚀 Mock CRM server running on http://localhost:${PORT}`);
  console.log(`   Webhook endpoint: http://localhost:${PORT}/api/webhook`);
});











