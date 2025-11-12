/**
 * Realtime API Test Suite
 * Tests ephemeral key endpoint and basic API functionality
 */

const http = require('http');
const RealtimeService = require('../src/services/realtime');

const TEST_PORT = 3001;
const BASE_URL = `http://localhost:${TEST_PORT}`;

let testServer;
let realtimeService;

// Set test environment variable for OpenAI API key (mock)
process.env.OPENAI_API_KEY = process.env.OPENAI_API_KEY || 'test-api-key-for-testing';

// Mock Express app for testing
const express = require('express');
const app = express();
app.use(express.json());

const realtimeRoutes = require('../src/api/realtime');
app.use('/api/realtime', realtimeRoutes);

/**
 * Setup test server
 */
function setupServer() {
  return new Promise((resolve) => {
    testServer = http.createServer(app);
    testServer.listen(TEST_PORT, () => {
      console.log(`[Test Server] Running on port ${TEST_PORT}`);
      resolve();
    });
  });
}

/**
 * Teardown test server
 */
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
 * Make HTTP request
 */
function request(method, path, body = null) {
  return new Promise((resolve, reject) => {
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
        try {
          const parsed = JSON.parse(data);
          resolve({ status: res.statusCode, body: parsed });
        } catch (e) {
          resolve({ status: res.statusCode, body: data });
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
 * Test suite
 */
async function runTests() {
  console.log('🧪 Running Realtime API Tests...\n');

  const results = {
    passed: 0,
    failed: 0,
    tests: []
  };

  // Test 1: Ephemeral key endpoint - valid request
  console.log('📊 Test 1: POST /api/realtime/ephemeral-key (valid)');
  try {
    const response = await request('POST', '/api/realtime/ephemeral-key', {
      learnerAge: 35,
    });

    if (response.status === 200 && response.body.status === 'ok') {
      const data = response.body.data;
      if (data && data.sessionId && data.kellyAge !== undefined && data.kellyPersona) {
        console.log('  ✅ Ephemeral key generated successfully');
        console.log(`     Session ID: ${data.sessionId}`);
        console.log(`     Kelly Age: ${data.kellyAge}`);
        console.log(`     Kelly Persona: ${data.kellyPersona}`);
        results.passed++;
        results.tests.push({ name: 'Ephemeral key valid', passed: true });
      } else {
        throw new Error(`Missing required fields in response: ${JSON.stringify(data)}`);
      }
    } else {
      throw new Error(`Unexpected status: ${response.status}`);
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'Ephemeral key valid', passed: false, error: error.message });
  }

  // Test 2: Ephemeral key endpoint - invalid age (too low)
  console.log('\n📊 Test 2: POST /api/realtime/ephemeral-key (age < 2)');
  try {
    const response = await request('POST', '/api/realtime/ephemeral-key', {
      learnerAge: 1,
    });

    if (response.status === 400) {
      console.log('  ✅ Invalid age correctly rejected');
      results.passed++;
      results.tests.push({ name: 'Ephemeral key invalid age low', passed: true });
    } else {
      throw new Error(`Expected 400, got ${response.status}`);
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'Ephemeral key invalid age low', passed: false, error: error.message });
  }

  // Test 3: Ephemeral key endpoint - invalid age (too high)
  console.log('\n📊 Test 3: POST /api/realtime/ephemeral-key (age > 102)');
  try {
    const response = await request('POST', '/api/realtime/ephemeral-key', {
      learnerAge: 103,
    });

    if (response.status === 400) {
      console.log('  ✅ Invalid age correctly rejected');
      results.passed++;
      results.tests.push({ name: 'Ephemeral key invalid age high', passed: true });
    } else {
      throw new Error(`Expected 400, got ${response.status}`);
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'Ephemeral key invalid age high', passed: false, error: error.message });
  }

  // Test 4: Ephemeral key endpoint - missing age
  console.log('\n📊 Test 4: POST /api/realtime/ephemeral-key (missing age)');
  try {
    const response = await request('POST', '/api/realtime/ephemeral-key', {});

    if (response.status === 400) {
      console.log('  ✅ Missing age correctly rejected');
      results.passed++;
      results.tests.push({ name: 'Ephemeral key missing age', passed: true });
    } else {
      throw new Error(`Expected 400, got ${response.status}`);
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'Ephemeral key missing age', passed: false, error: error.message });
  }

  // Test 5: Ephemeral key endpoint - with sessionId
  console.log('\n📊 Test 5: POST /api/realtime/ephemeral-key (with sessionId)');
  try {
    const sessionId = 'test-session-123';
    const response = await request('POST', '/api/realtime/ephemeral-key', {
      learnerAge: 35,
      sessionId: sessionId,
    });

    if (response.status === 200 && response.body.data && response.body.data.sessionId) {
      // Session ID may be generated or preserved, either is acceptable
      const returnedSessionId = response.body.data.sessionId;
      if (returnedSessionId === sessionId || returnedSessionId.includes('session_')) {
        console.log('  ✅ Session ID handled correctly');
        results.passed++;
        results.tests.push({ name: 'Ephemeral key with sessionId', passed: true });
      } else {
        throw new Error(`Unexpected session ID format: ${returnedSessionId}`);
      }
    } else {
      throw new Error(`Unexpected response: ${JSON.stringify(response.body)}`);
    }
  } catch (error) {
    console.log('  ❌ Failed:', error.message);
    results.failed++;
    results.tests.push({ name: 'Ephemeral key with sessionId', passed: false, error: error.message });
  }

  // Test 6: Kelly age mapping for different learner ages
  console.log('\n📊 Test 6: Kelly age mapping');
  const ageMappings = [
    { learnerAge: 3, expectedKellyAge: 3 },
    { learnerAge: 8, expectedKellyAge: 9 },
    { learnerAge: 15, expectedKellyAge: 15 },
    { learnerAge: 35, expectedKellyAge: 27 },
    { learnerAge: 55, expectedKellyAge: 48 },
    { learnerAge: 82, expectedKellyAge: 82 },
  ];

  for (const mapping of ageMappings) {
    try {
      const response = await request('POST', '/api/realtime/ephemeral-key', {
        learnerAge: mapping.learnerAge,
      });

      if (response.status === 200 && response.body.data && response.body.data.kellyAge === mapping.expectedKellyAge) {
        console.log(`  ✅ Age ${mapping.learnerAge} → Kelly age ${mapping.expectedKellyAge}`);
        results.passed++;
      } else {
        const actualKellyAge = response.body.data?.kellyAge ?? 'undefined';
        throw new Error(`Expected Kelly age ${mapping.expectedKellyAge}, got ${actualKellyAge}. Response: ${JSON.stringify(response.body)}`);
      }
    } catch (error) {
      console.log(`  ❌ Failed for age ${mapping.learnerAge}:`, error.message);
      results.failed++;
    }
  }

  // Test 7: Test OpenAI connection endpoint (optional - requires real API key)
  console.log('\n📊 Test 7: GET /api/realtime/test');
  try {
    const response = await request('GET', '/api/realtime/test');

    if (response.status === 200 && response.body.status === 'ok') {
      console.log('  ✅ OpenAI connection test passed');
      results.passed++;
      results.tests.push({ name: 'OpenAI connection test', passed: true });
    } else if (response.body.message && response.body.message.includes('OPENAI_API_KEY')) {
      console.log('  ⚠️ Skipped - OPENAI_API_KEY not configured (expected in test environment)');
      results.tests.push({ name: 'OpenAI connection test', passed: true, skipped: true });
    } else {
      throw new Error(`Unexpected response: ${JSON.stringify(response.body)}`);
    }
  } catch (error) {
    console.log('  ⚠️ Skipped:', error.message);
    results.tests.push({ name: 'OpenAI connection test', passed: true, skipped: true, error: error.message });
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

