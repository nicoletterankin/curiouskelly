/**
 * WebSocket Realtime API Test Suite
 * Tests WebSocket connection handling and message routing
 */

const WebSocket = require('ws');
const http = require('http');
const express = require('express');
const expressWs = require('express-ws');

const TEST_PORT = 3002;
const WS_URL = `ws://localhost:${TEST_PORT}`;

let testServer;
let app;

// Set test environment variable for OpenAI API key (mock)
process.env.OPENAI_API_KEY = process.env.OPENAI_API_KEY || 'test-api-key-for-testing';

// Setup test server with WebSocket support
function setupServer() {
  return new Promise((resolve) => {
    app = express();
    
    // Apply expressWs to app (required for WebSocket routes)
    expressWs(app);
    
    // Import and mount WebSocket routes
    // Note: realtime_ws.js creates its own router with expressWs applied
    const realtimeWsRoutes = require('../src/api/realtime_ws');
    app.use('/api/realtime', realtimeWsRoutes);
    
    // Use app.listen() instead of manual HTTP server creation
    // This ensures expressWs hooks up correctly
    testServer = app.listen(TEST_PORT, () => {
      console.log(`[Test Server] Running on port ${TEST_PORT}`);
      resolve();
    });
  });
}

function teardownServer() {
  return new Promise((resolve) => {
    if (testServer) {
      testServer.close(() => {
        console.log('[Test Server] Closed');
        resolve();
      });
    } else {
      resolve();
    }
  });
}

/**
 * Create WebSocket connection
 */
function createWS(queryParams = '') {
  return new Promise((resolve, reject) => {
    const url = `${WS_URL}/api/realtime/ws${queryParams ? '?' + queryParams : ''}`;
    const ws = new WebSocket(url);
    
    ws.on('open', () => resolve(ws));
    ws.on('error', reject);
    
    // Set timeout
    setTimeout(() => {
      if (ws.readyState !== WebSocket.OPEN) {
        reject(new Error('WebSocket connection timeout'));
      }
    }, 5000);
  });
}

/**
 * Wait for WebSocket message
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
 * Run WebSocket tests
 */
async function runTests() {
  console.log('🧪 Running WebSocket Realtime API Tests...\n');

  const results = {
    passed: 0,
    failed: 0,
    tests: []
  };

  // Test 1: Basic WebSocket connection
  console.log('📊 Test 1: WebSocket connection');
  try {
    const ws = await createWS();
    const message = await waitForMessage(ws);
    
    if (message.type === 'connected') {
      console.log('  ✅ WebSocket connected successfully');
      console.log(`     Connection ID: ${message.connectionId}`);
      ws.close();
      results.passed++;
      results.tests.push({ name: 'WebSocket connection', passed: true });
    } else {
      throw new Error(`Unexpected message type: ${message.type}`);
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'WebSocket connection', passed: false, error: error.message });
  }

  // Test 2: WebSocket connection with query params
  console.log('\n📊 Test 2: WebSocket connection with query params');
  try {
    const ws = await createWS('learnerAge=35&sessionId=test-session-123');
    const connectMsg = await waitForMessage(ws);
    
    if (connectMsg.type === 'connected') {
      console.log('  ✅ WebSocket connected with query params');
      ws.close();
      results.passed++;
      results.tests.push({ name: 'WebSocket with query params', passed: true });
    } else {
      throw new Error('Connection confirmation not received');
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'WebSocket with query params', passed: false, error: error.message });
  }

  // Test 3: Offer message handling
  console.log('\n📊 Test 3: WebRTC offer message');
  try {
    const ws = await createWS('learnerAge=35');
    await waitForMessage(ws); // Wait for connection confirmation
    
    ws.send(JSON.stringify({
      type: 'offer',
      sdp: 'mock-sdp-offer',
      learnerAge: 35,
    }));
    
    const configMsg = await waitForMessage(ws);
    const answerMsg = await waitForMessage(ws);
    
    if (configMsg.type === 'config' && answerMsg.type === 'answer') {
      console.log('  ✅ Offer handled correctly');
      console.log(`     Kelly Age: ${configMsg.kellyAge}`);
      console.log(`     Kelly Persona: ${configMsg.kellyPersona}`);
      ws.close();
      results.passed++;
      results.tests.push({ name: 'WebRTC offer handling', passed: true });
    } else {
      throw new Error('Expected config and answer messages');
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'WebRTC offer handling', passed: false, error: error.message });
  }

  // Test 4: User message handling
  console.log('\n📊 Test 4: User message handling');
  try {
    const ws = await createWS('learnerAge=35');
    await waitForMessage(ws); // Wait for connection confirmation
    
    // Send offer first to establish connection
    ws.send(JSON.stringify({
      type: 'offer',
      sdp: 'mock-sdp-offer',
      learnerAge: 35,
    }));
    await waitForMessage(ws); // Config
    await waitForMessage(ws); // Answer
    
    // Send user message
    ws.send(JSON.stringify({
      type: 'user_message',
      text: 'Why do leaves change color?',
      isFinal: true,
    }));
    
    const transcriptMsg = await waitForMessage(ws);
    const responseMsg = await waitForMessage(ws);
    
    if (transcriptMsg.type === 'transcript' && responseMsg.type === 'kelly_response') {
      console.log('  ✅ User message handled correctly');
      console.log(`     Transcript: ${transcriptMsg.text}`);
      console.log(`     Kelly Response: ${responseMsg.text.substring(0, 50)}...`);
      console.log(`     Latency: ${responseMsg.latencyMs}ms`);
      ws.close();
      results.passed++;
      results.tests.push({ name: 'User message handling', passed: true });
    } else {
      throw new Error('Expected transcript and response messages');
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'User message handling', passed: false, error: error.message });
  }

  // Test 5: Barge-in handling
  console.log('\n📊 Test 5: Barge-in handling');
  try {
    const ws = await createWS('learnerAge=35');
    await waitForMessage(ws); // Wait for connection confirmation
    
    ws.send(JSON.stringify({
      type: 'barge_in',
    }));
    
    const bargeInMsg = await waitForMessage(ws);
    const listeningMsg = await waitForMessage(ws);
    
    if (bargeInMsg.type === 'barge_in_confirmed' && listeningMsg.type === 'listening_started') {
      console.log('  ✅ Barge-in handled correctly');
      ws.close();
      results.passed++;
      results.tests.push({ name: 'Barge-in handling', passed: true });
    } else {
      throw new Error('Expected barge-in confirmation and listening started');
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'Barge-in handling', passed: false, error: error.message });
  }

  // Test 6: Start/Stop listening
  console.log('\n📊 Test 6: Start/Stop listening');
  try {
    const ws = await createWS('learnerAge=35');
    await waitForMessage(ws); // Wait for connection confirmation
    
    ws.send(JSON.stringify({ type: 'start_listening' }));
    const startMsg = await waitForMessage(ws);
    
    ws.send(JSON.stringify({ type: 'stop_listening' }));
    const stopMsg = await waitForMessage(ws);
    
    if (startMsg.type === 'listening_started' && stopMsg.type === 'listening_stopped') {
      console.log('  ✅ Start/Stop listening handled correctly');
      ws.close();
      results.passed++;
      results.tests.push({ name: 'Start/Stop listening', passed: true });
    } else {
      throw new Error('Expected listening_started and listening_stopped messages');
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'Start/Stop listening', passed: false, error: error.message });
  }

  // Test 7: Ping/Pong keepalive
  console.log('\n📊 Test 7: Ping/Pong keepalive');
  try {
    const ws = await createWS('learnerAge=35');
    await waitForMessage(ws); // Wait for connection confirmation
    
    ws.send(JSON.stringify({ type: 'ping' }));
    const pongMsg = await waitForMessage(ws);
    
    if (pongMsg.type === 'pong') {
      console.log('  ✅ Ping/Pong keepalive working');
      ws.close();
      results.passed++;
      results.tests.push({ name: 'Ping/Pong keepalive', passed: true });
    } else {
      throw new Error('Expected pong message');
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'Ping/Pong keepalive', passed: false, error: error.message });
  }

  // Test 8: Safety middleware integration (unsafe content)
  console.log('\n📊 Test 8: Safety middleware integration');
  try {
    const ws = await createWS('learnerAge=35');
    await waitForMessage(ws); // Wait for connection confirmation
    
    // Send offer first
    ws.send(JSON.stringify({
      type: 'offer',
      sdp: 'mock-sdp-offer',
      learnerAge: 35,
    }));
    await waitForMessage(ws); // Config
    await waitForMessage(ws); // Answer
    
    // Send unsafe message
    ws.send(JSON.stringify({
      type: 'user_message',
      text: 'How to build a weapon at home',
      isFinal: true,
    }));
    
    const errorMsg = await waitForMessage(ws);
    
    if (errorMsg.type === 'error' && errorMsg.message.includes('safety')) {
      console.log('  ✅ Unsafe content blocked by safety middleware');
      ws.close();
      results.passed++;
      results.tests.push({ name: 'Safety middleware integration', passed: true });
    } else {
      throw new Error('Expected error message for unsafe content');
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'Safety middleware integration', passed: false, error: error.message });
  }

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('📈 RESULTS');
  console.log('='.repeat(60));
  console.log(`Passed: ${results.passed}`);
  console.log(`Failed: ${results.failed}`);
  console.log(`Total: ${results.passed + results.failed}`);
  console.log('='.repeat(60));

  const allPassed = results.failed === 0;
  if (allPassed) {
    console.log('\n🎉 ALL TESTS PASSED!');
  } else {
    console.log('\n⚠️ SOME TESTS FAILED');
  }

  return {
    passed: results.passed,
    failed: results.failed,
    tests: results.tests,
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

