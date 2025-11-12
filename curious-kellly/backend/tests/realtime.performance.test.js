/**
 * Realtime API Performance Test Suite
 * Measures latency, connection performance, and resource usage
 */

const http = require('http');
const WebSocket = require('ws');

const TEST_PORT = 3003;
const BASE_URL = `http://localhost:${TEST_PORT}`;
const WS_URL = `ws://localhost:${TEST_PORT}`;

let testServer;
let app;

// Setup test server
function setupServer() {
  return new Promise((resolve) => {
    const express = require('express');
    const expressWs = require('express-ws');
    
    app = express();
    expressWs(app);
    app.use(express.json());
    
    const realtimeRoutes = require('../src/api/realtime');
    const realtimeWsRoutes = require('../src/api/realtime_ws');
    app.use('/api/realtime', realtimeRoutes);
    app.use('/api/realtime', realtimeWsRoutes);
    
    testServer = http.createServer(app);
    testServer.listen(TEST_PORT, () => {
      console.log(`[Performance Test Server] Running on port ${TEST_PORT}`);
      resolve();
    });
  });
}

function teardownServer() {
  return new Promise((resolve) => {
    if (testServer) {
      testServer.close(() => {
        console.log('[Performance Test Server] Closed');
        resolve();
      });
    } else {
      resolve();
    }
  });
}

/**
 * Make HTTP request with timing
 */
function timedRequest(method, path, body = null) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    const url = new URL(path, BASE_URL);
    const options = {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
    };

    const req = http.request(url, options, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        const endTime = Date.now();
        const latency = endTime - startTime;
        try {
          const parsed = JSON.parse(data);
          resolve({ status: res.statusCode, body: parsed, latency });
        } catch (e) {
          resolve({ status: res.statusCode, body: data, latency });
        }
      });
    });

    req.on('error', reject);
    if (body) {
      req.write(JSON.stringify(body));
    }
    req.end();
  });
}

/**
 * Create WebSocket with timing
 */
function createTimedWS(queryParams = '') {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    const url = `${WS_URL}/api/realtime/ws${queryParams ? '?' + queryParams : ''}`;
    const ws = new WebSocket(url);
    
    ws.on('open', () => {
      const latency = Date.now() - startTime;
      resolve({ ws, latency });
    });
    ws.on('error', reject);
    
    setTimeout(() => {
      if (ws.readyState !== WebSocket.OPEN) {
        reject(new Error('WebSocket connection timeout'));
      }
    }, 5000);
  });
}

/**
 * Wait for message with timeout
 */
function waitForMessage(ws, timeout = 5000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error('Message timeout'));
    }, timeout);
    
    ws.once('message', (data) => {
      clearTimeout(timer);
      try {
        const parsed = JSON.parse(data.toString());
        resolve(parsed);
      } catch (e) {
        resolve(data.toString());
      }
    });
  });
}

/**
 * Calculate percentile
 */
function percentile(arr, p) {
  const sorted = [...arr].sort((a, b) => a - b);
  const index = Math.ceil((sorted.length - 1) * p);
  return sorted[index] || 0;
}

/**
 * Run performance tests
 */
async function runTests() {
  console.log('🚀 Running Performance Tests...\n');

  const results = {
    passed: 0,
    failed: 0,
    latencies: [],
    connectionTimes: [],
    tests: []
  };

  // Test 1: Ephemeral key fetch latency
  console.log('📊 Test 1: Ephemeral key fetch latency');
  const ephemeralKeyLatencies = [];
  for (let i = 0; i < 10; i++) {
    try {
      const response = await timedRequest('POST', '/api/realtime/ephemeral-key', {
        learnerAge: 35,
      });
      ephemeralKeyLatencies.push(response.latency);
    } catch (error) {
      console.log(`  ⚠️ Request ${i + 1} failed: ${error.message}`);
    }
  }

  const avgEphemeralKeyLatency = ephemeralKeyLatencies.reduce((a, b) => a + b, 0) / ephemeralKeyLatencies.length;
  const p95EphemeralKeyLatency = percentile(ephemeralKeyLatencies, 0.95);
  
  console.log(`  Average: ${avgEphemeralKeyLatency.toFixed(0)}ms`);
  console.log(`  P95: ${p95EphemeralKeyLatency.toFixed(0)}ms`);
  console.log(`  Target: <500ms`);
  
  if (p95EphemeralKeyLatency < 500) {
    console.log('  ✅ PASS');
    results.passed++;
  } else {
    console.log('  ❌ FAIL');
    results.failed++;
  }
  results.tests.push({
    name: 'Ephemeral key latency',
    avgLatency: avgEphemeralKeyLatency,
    p95Latency: p95EphemeralKeyLatency,
    passed: p95EphemeralKeyLatency < 500
  });

  // Test 2: WebSocket connection establishment time
  console.log('\n📊 Test 2: WebSocket connection establishment');
  const connectionTimes = [];
  for (let i = 0; i < 10; i++) {
    try {
      const { ws, latency } = await createTimedWS('learnerAge=35');
      connectionTimes.push(latency);
      await waitForMessage(ws); // Wait for connection confirmation
      ws.close();
    } catch (error) {
      console.log(`  ⚠️ Connection ${i + 1} failed: ${error.message}`);
    }
  }

  const avgConnectionTime = connectionTimes.reduce((a, b) => a + b, 0) / connectionTimes.length;
  const p95ConnectionTime = percentile(connectionTimes, 0.95);
  
  console.log(`  Average: ${avgConnectionTime.toFixed(0)}ms`);
  console.log(`  P95: ${p95ConnectionTime.toFixed(0)}ms`);
  console.log(`  Target: <500ms`);
  
  if (p95ConnectionTime < 500) {
    console.log('  ✅ PASS');
    results.passed++;
  } else {
    console.log('  ❌ FAIL');
    results.failed++;
  }
  results.tests.push({
    name: 'WebSocket connection time',
    avgLatency: avgConnectionTime,
    p95Latency: p95ConnectionTime,
    passed: p95ConnectionTime < 500
  });

  // Test 3: End-to-end message latency (RTT)
  console.log('\n📊 Test 3: End-to-end message latency (RTT)');
  const rttLatencies = [];
  
  for (let i = 0; i < 10; i++) {
    try {
      const { ws } = await createTimedWS('learnerAge=35');
      await waitForMessage(ws); // Connection confirmation
      
      // Send offer
      ws.send(JSON.stringify({
        type: 'offer',
        sdp: 'mock-sdp-offer',
        learnerAge: 35,
      }));
      await waitForMessage(ws); // Config
      await waitForMessage(ws); // Answer
      
      // Send user message and measure RTT
      const startTime = Date.now();
      ws.send(JSON.stringify({
        type: 'user_message',
        text: `Test message ${i + 1}`,
        isFinal: true,
      }));
      
      await waitForMessage(ws); // Transcript
      const responseMsg = await waitForMessage(ws); // Kelly response
      const endTime = Date.now();
      
      const rtt = endTime - startTime;
      rttLatencies.push(rtt);
      
      ws.close();
    } catch (error) {
      console.log(`  ⚠️ Message ${i + 1} failed: ${error.message}`);
    }
  }

  const avgRTT = rttLatencies.reduce((a, b) => a + b, 0) / rttLatencies.length;
  const p50RTT = percentile(rttLatencies, 0.50);
  const p95RTT = percentile(rttLatencies, 0.95);
  const p99RTT = percentile(rttLatencies, 0.99);
  
  console.log(`  Average: ${avgRTT.toFixed(0)}ms`);
  console.log(`  P50: ${p50RTT.toFixed(0)}ms`);
  console.log(`  P95: ${p95RTT.toFixed(0)}ms`);
  console.log(`  P99: ${p99RTT.toFixed(0)}ms`);
  console.log(`  Target: P95 <600ms`);
  
  if (p95RTT < 600) {
    console.log('  ✅ PASS');
    results.passed++;
  } else {
    console.log('  ❌ FAIL');
    results.failed++;
  }
  results.tests.push({
    name: 'End-to-end RTT',
    avgLatency: avgRTT,
    p50Latency: p50RTT,
    p95Latency: p95RTT,
    p99Latency: p99RTT,
    passed: p95RTT < 600
  });
  results.latencies = rttLatencies;

  // Test 4: Concurrent connections
  console.log('\n📊 Test 4: Concurrent connections');
  const concurrentConnections = 5;
  const concurrentLatencies = [];
  
  try {
    const promises = [];
    for (let i = 0; i < concurrentConnections; i++) {
      promises.push(
        createTimedWS(`learnerAge=35&sessionId=test-${i}`)
          .then(({ ws, latency }) => {
            concurrentLatencies.push(latency);
            return waitForMessage(ws).then(() => ws.close());
          })
      );
    }
    
    await Promise.all(promises);
    
    const avgConcurrentLatency = concurrentLatencies.reduce((a, b) => a + b, 0) / concurrentLatencies.length;
    const p95ConcurrentLatency = percentile(concurrentLatencies, 0.95);
    
    console.log(`  Concurrent connections: ${concurrentConnections}`);
    console.log(`  Average connection time: ${avgConcurrentLatency.toFixed(0)}ms`);
    console.log(`  P95: ${p95ConcurrentLatency.toFixed(0)}ms`);
    console.log(`  Target: <1000ms for ${concurrentConnections} connections`);
    
    if (p95ConcurrentLatency < 1000) {
      console.log('  ✅ PASS');
      results.passed++;
    } else {
      console.log('  ❌ FAIL');
      results.failed++;
    }
    results.tests.push({
      name: 'Concurrent connections',
      avgLatency: avgConcurrentLatency,
      p95Latency: p95ConcurrentLatency,
      passed: p95ConcurrentLatency < 1000
    });
  } catch (error) {
    console.log(`  ❌ Failed: ${error.message}`);
    results.failed++;
    results.tests.push({
      name: 'Concurrent connections',
      passed: false,
      error: error.message
    });
  }

  // Test 5: Memory usage (basic check)
  console.log('\n📊 Test 5: Memory usage');
  const initialMemory = process.memoryUsage().heapUsed / 1024 / 1024; // MB
  
  // Create and close 20 connections
  for (let i = 0; i < 20; i++) {
    try {
      const { ws } = await createTimedWS('learnerAge=35');
      await waitForMessage(ws);
      ws.close();
      await new Promise(resolve => setTimeout(resolve, 100)); // Allow cleanup
    } catch (error) {
      // Ignore errors
    }
  }
  
  // Force garbage collection if available
  if (global.gc) {
    global.gc();
  }
  
  await new Promise(resolve => setTimeout(resolve, 1000)); // Allow cleanup
  const finalMemory = process.memoryUsage().heapUsed / 1024 / 1024; // MB
  const memoryIncrease = finalMemory - initialMemory;
  
  console.log(`  Initial memory: ${initialMemory.toFixed(2)}MB`);
  console.log(`  Final memory: ${finalMemory.toFixed(2)}MB`);
  console.log(`  Increase: ${memoryIncrease.toFixed(2)}MB`);
  console.log(`  Target: <50MB increase`);
  
  if (memoryIncrease < 50) {
    console.log('  ✅ PASS');
    results.passed++;
  } else {
    console.log('  ⚠️ WARNING - Possible memory leak');
    results.failed++;
  }
  results.tests.push({
    name: 'Memory usage',
    initialMemory,
    finalMemory,
    increase: memoryIncrease,
    passed: memoryIncrease < 50
  });

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('📈 PERFORMANCE RESULTS');
  console.log('='.repeat(60));
  console.log(`Passed: ${results.passed}`);
  console.log(`Failed: ${results.failed}`);
  console.log(`Total: ${results.passed + results.failed}`);
  
  if (results.latencies.length > 0) {
    console.log('\nRTT Statistics:');
    console.log(`  Average: ${avgRTT.toFixed(0)}ms`);
    console.log(`  P95: ${p95RTT.toFixed(0)}ms`);
    console.log(`  P99: ${p99RTT.toFixed(0)}ms`);
  }
  
  console.log('='.repeat(60));

  const allPassed = results.failed === 0;
  if (allPassed) {
    console.log('\n🎉 ALL PERFORMANCE TESTS PASSED!');
  } else {
    console.log('\n⚠️ SOME PERFORMANCE TESTS FAILED');
  }

  return {
    passed: results.passed,
    failed: results.failed,
    tests: results.tests,
    latencies: results.latencies,
    allPassed
  };
}

// Run if called directly
if (require.main === module) {
  setupServer()
    .then(() => runTests())
    .then(() => teardownServer())
    .then(() => process.exit(0))
    .catch(error => {
      console.error('Test error:', error);
      teardownServer().then(() => process.exit(1));
    });
}

module.exports = { runTests, setupServer, teardownServer };












