const express = require('express');
const expressWs = require('express-ws');
const OpenAI = require('openai');
const SafetyService = require('../services/safety');
const RealtimeService = require('../services/realtime');
const SessionService = require('../services/session');

const router = express.Router();

// Apply WebSocket support to this router
const wsInstance = expressWs(router);

// Store active connections
const connections = new Map();

// Session service instance
const sessionService = new SessionService();

/**
 * WebSocket endpoint for OpenAI Realtime API
 * ws://localhost:3000/api/realtime/ws
 * Query params: ?sessionId=xxx&learnerAge=35
 */
router.ws('/ws', async (ws, req) => {
  const connectionId = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const sessionId = req.query.sessionId || null;
  const learnerAge = req.query.learnerAge ? parseInt(req.query.learnerAge) : null;
  
  console.log(`[Realtime WS] New connection: ${connectionId} (session: ${sessionId}, age: ${learnerAge})`);
  
  const connectionData = {
    id: connectionId,
    ws,
    sessionId,
    learnerAge: learnerAge || null,
    kellyAge: null,
    openaiClient: new OpenAI({ apiKey: process.env.OPENAI_API_KEY }),
    realtimeService: new RealtimeService(),
    safetyService: new SafetyService(),
    isListening: false,
    isSpeaking: false,
    connectedAt: new Date(),
    lastActivity: new Date(),
    reconnectAttempts: 0,
    maxReconnectAttempts: 3,
  };
  
  // Initialize session if provided
  if (sessionId) {
    try {
      const session = await sessionService.getSession(sessionId);
      connectionData.learnerAge = connectionData.learnerAge || session.age;
      console.log(`[Realtime WS] Loaded session: ${sessionId} (age: ${connectionData.learnerAge})`);
    } catch (error) {
      console.warn(`[Realtime WS] Session not found: ${sessionId}, creating new connection`);
    }
  }
  
  connections.set(connectionId, connectionData);
  
  // Send connection confirmation
  ws.send(JSON.stringify({
    type: 'connected',
    connectionId,
    timestamp: new Date().toISOString(),
  }));
  
  // Handle incoming messages
  ws.on('message', async (message) => {
    try {
      connectionData.lastActivity = new Date();
      const data = JSON.parse(message);
      await handleMessage(connectionData, data);
    } catch (error) {
      console.error(`[Realtime WS] Error handling message:`, error);
      sendError(ws, error.message);
    }
  });
  
  // Handle connection close
  ws.on('close', async () => {
    console.log(`[Realtime WS] Connection closed: ${connectionId}`);
    
    // Update session activity (getSession automatically updates lastActivity)
    if (connectionData.sessionId) {
      try {
        await sessionService.getSession(connectionData.sessionId);
      } catch (error) {
        console.warn(`[Realtime WS] Failed to update session activity: ${error.message}`);
      }
    }
    
    connections.delete(connectionId);
  });
  
  // Handle errors
  ws.on('error', (error) => {
    console.error(`[Realtime WS] Connection error:`, error);
    connections.delete(connectionId);
  });
  
  // Handle ping/pong for connection keepalive
  const pingInterval = setInterval(() => {
    if (ws.readyState === 1) { // WebSocket.OPEN
      ws.ping();
    } else {
      clearInterval(pingInterval);
    }
  }, 30000); // Ping every 30 seconds
  
  ws.on('close', () => {
    clearInterval(pingInterval);
  });
});

/**
 * Handle WebSocket messages
 */
async function handleMessage(conn, data) {
  const { type } = data;
  console.log(`[Realtime WS] Message type: ${type}`);
  
  switch (type) {
    case 'offer':
      await handleOffer(conn, data);
      break;
    case 'ice_candidate':
      await handleIceCandidate(conn, data);
      break;
    case 'start_listening':
      await handleStartListening(conn);
      break;
    case 'stop_listening':
      await handleStopListening(conn);
      break;
    case 'user_message':
      await handleUserMessage(conn, data);
      break;
    case 'barge_in':
      await handleBargeIn(conn);
      break;
    case 'reconnect':
      await handleReconnect(conn, data);
      break;
    case 'ping':
      // Respond to ping for keepalive
      conn.ws.send(JSON.stringify({
        type: 'pong',
        timestamp: new Date().toISOString(),
      }));
      break;
    default:
      console.warn(`[Realtime WS] Unknown message type: ${type}`);
  }
}

/**
 * Handle WebRTC offer
 */
async function handleOffer(conn, data) {
  const { sdp, learnerAge, sessionId } = data;
  
  if (!learnerAge || learnerAge < 2 || learnerAge > 102) {
    sendError(conn.ws, 'Invalid learnerAge (must be 2-102)');
    return;
  }
  
  conn.learnerAge = learnerAge;
  conn.kellyAge = conn.realtimeService.getKellyAge(learnerAge);
  
  // Update or create session
  if (sessionId) {
    conn.sessionId = sessionId;
    try {
      await sessionService.getSession(sessionId);
    } catch (error) {
      // Create new session if it doesn't exist
      try {
        await sessionService.createSession(learnerAge, 'general', null);
        conn.sessionId = sessionId;
      } catch (createError) {
        console.warn(`[Realtime WS] Failed to create session: ${createError.message}`);
      }
    }
  }
  
  console.log(`[Realtime WS] Offer received for age ${learnerAge} (Kelly age: ${conn.kellyAge}, session: ${conn.sessionId})`);
  
  // Create Realtime API configuration
  const config = conn.realtimeService.createRealtimeConfig(learnerAge, conn.sessionId || conn.id);
  
  // Send configuration to client
  conn.ws.send(JSON.stringify({
    type: 'config',
    config,
    kellyAge: conn.kellyAge,
    kellyPersona: conn.realtimeService.getKellyPersona(learnerAge),
    timestamp: new Date().toISOString(),
  }));
  
  // For WebRTC, we would establish connection to OpenAI Realtime API
  // For now, we'll use WebSocket-based signaling
  const answer = {
    type: 'answer',
    sdp: sdp, // Echo back for now - in production, forward to OpenAI Realtime API
    sessionId: conn.sessionId || conn.id,
  };
  
  conn.ws.send(JSON.stringify(answer));
  
  console.log(`[Realtime WS] Answer sent`);
}

/**
 * Handle ICE candidate
 */
async function handleIceCandidate(conn, data) {
  const { candidate } = data;
  console.log(`[Realtime WS] ICE candidate received`);
  
  // Forward to OpenAI (in production implementation)
  // For now, just log it
}

/**
 * Handle start listening
 */
async function handleStartListening(conn) {
  conn.isListening = true;
  console.log(`[Realtime WS] Started listening`);
  
  conn.ws.send(JSON.stringify({
    type: 'listening_started',
    timestamp: new Date().toISOString(),
  }));
}

/**
 * Handle stop listening
 */
async function handleStopListening(conn) {
  conn.isListening = false;
  console.log(`[Realtime WS] Stopped listening`);
  
  conn.ws.send(JSON.stringify({
    type: 'listening_stopped',
    timestamp: new Date().toISOString(),
  }));
}

/**
 * Handle user message (text or transcribed speech)
 */
async function handleUserMessage(conn, data) {
  const { text, isFinal = true } = data;
  
  if (!text || text.trim().length === 0) {
    return;
  }
  
  console.log(`[Realtime WS] User message: "${text}" (final: ${isFinal})`);
  
  // Update session activity
  if (conn.sessionId) {
    try {
      await sessionService.updateSessionActivity(conn.sessionId);
    } catch (error) {
      console.warn(`[Realtime WS] Failed to update session: ${error.message}`);
    }
  }
  
  // Safety check (only for final transcripts)
  if (isFinal) {
    const moderationResult = await conn.safetyService.moderateContent(text);
    if (!moderationResult.safe) {
      sendError(conn.ws, 'Message blocked by safety filter', {
        categories: moderationResult.categories,
      });
      return;
    }
    
    // Age-appropriateness check
    if (conn.learnerAge) {
      const ageCheck = conn.safetyService.isAgeAppropriate(text, conn.learnerAge);
      if (!ageCheck.appropriate) {
        sendError(conn.ws, 'Message not age-appropriate', {
          reason: ageCheck.reason,
          recommendedMinAge: ageCheck.recommendedMinAge,
        });
        return;
      }
    }
  }
  
  // Send transcript confirmation
  conn.ws.send(JSON.stringify({
    type: 'transcript',
    text: text,
    isFinal: isFinal,
    timestamp: new Date().toISOString(),
  }));
  
  // Only process final transcripts
  if (!isFinal) {
    return;
  }
  
  // Get Kelly's response
  const requestStartTime = Date.now();
  const result = await conn.realtimeService.getKellyResponse(
    conn.learnerAge || 35,
    'general', // Topic - should come from lesson context
    text
  );
  
  const latencyMs = Date.now() - requestStartTime;
  
  if (result.success) {
    // Safety check Kelly's response
    const outputModeration = await conn.safetyService.moderateContent(result.response);
    let finalResponse = result.response;
    let safetyRewritten = false;
    
    if (!outputModeration.safe) {
      // Rewrite to safe version
      const safeVersion = await conn.safetyService.safeCompletion(result.response);
      finalResponse = safeVersion.safeVersion;
      safetyRewritten = true;
    }
    
    // Send Kelly's response
    conn.ws.send(JSON.stringify({
      type: 'kelly_response',
      text: finalResponse,
      kellyAge: result.kellyAge,
      kellyPersona: result.kellyPersona,
      latencyMs,
      safetyRewritten,
      // audio: base64AudioData, // Would include audio in production
      timestamp: new Date().toISOString(),
    }));
    
    console.log(`[Realtime WS] Kelly response sent (${latencyMs}ms): "${finalResponse.substring(0, 50)}..."`);
    
    // Update session progress
    if (conn.sessionId) {
      try {
        await sessionService.updateProgress(conn.sessionId, {
          currentPhase: 'teaching',
          interactionCompleted: text,
        });
      } catch (error) {
        console.warn(`[Realtime WS] Failed to update progress: ${error.message}`);
      }
    }
  } else {
    sendError(conn.ws, 'Failed to get Kelly response', { error: result.error });
  }
}

/**
 * Handle barge-in (interrupt Kelly)
 */
async function handleBargeIn(conn) {
  console.log(`[Realtime WS] Barge-in triggered`);
  
  conn.isSpeaking = false;
  conn.isListening = true;
  
  // Stop any ongoing audio playback (client-side handles this)
  conn.ws.send(JSON.stringify({
    type: 'barge_in_confirmed',
    timestamp: new Date().toISOString(),
  }));
  
  // Start listening immediately
  await handleStartListening(conn);
  
  // Update session activity (getSession automatically updates lastActivity)
  if (conn.sessionId) {
    try {
      await sessionService.getSession(conn.sessionId);
    } catch (error) {
      console.warn(`[Realtime WS] Failed to update session: ${error.message}`);
    }
  }
}

/**
 * Handle reconnection attempt
 */
async function handleReconnect(conn, data) {
  const { sessionId, learnerAge } = data;
  
  console.log(`[Realtime WS] Reconnection attempt: ${conn.id}`);
  
  if (conn.reconnectAttempts >= conn.maxReconnectAttempts) {
    sendError(conn.ws, 'Maximum reconnection attempts reached');
    return;
  }
  
  conn.reconnectAttempts++;
  
  // Restore session if provided
  if (sessionId) {
    try {
      const session = await sessionService.getSession(sessionId);
      conn.sessionId = sessionId;
      conn.learnerAge = learnerAge || session.age;
      conn.kellyAge = conn.realtimeService.getKellyAge(conn.learnerAge);
      
      conn.ws.send(JSON.stringify({
        type: 'reconnected',
        sessionId,
        learnerAge: conn.learnerAge,
        kellyAge: conn.kellyAge,
        timestamp: new Date().toISOString(),
      }));
      
      console.log(`[Realtime WS] Reconnected successfully: ${sessionId}`);
    } catch (error) {
      console.warn(`[Realtime WS] Failed to restore session: ${error.message}`);
      sendError(conn.ws, 'Failed to restore session');
    }
  }
}

/**
 * Send error message
 */
function sendError(ws, message, data = {}) {
  ws.send(JSON.stringify({
    type: 'error',
    message: message,
    ...data,
    timestamp: new Date().toISOString(),
  }));
}

module.exports = router;



