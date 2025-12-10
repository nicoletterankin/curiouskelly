/**
 * Replicate API Client with retry logic and polling
 */

const https = require('https');
const config = require('./config');

class ReplicateClient {
  constructor(apiToken) {
    this.apiToken = apiToken;
    this.baseUrl = 'api.replicate.com';
  }
  
  async request(method, path, data = null) {
    return new Promise((resolve, reject) => {
      const options = {
        hostname: this.baseUrl,
        path: `/v1${path}`,
        method,
        headers: {
          'Authorization': `Bearer ${this.apiToken}`,
          'Content-Type': 'application/json',
        },
      };
      
      const req = https.request(options, (res) => {
        let body = [];
        res.on('data', chunk => body.push(chunk));
        res.on('end', () => {
          const buffer = Buffer.concat(body);
          try {
            const json = JSON.parse(buffer.toString());
            if (res.statusCode >= 400) {
              reject(new Error(`API Error ${res.statusCode}: ${JSON.stringify(json)}`));
            } else {
              resolve(json);
            }
          } catch (e) {
            resolve({ status: res.statusCode, data: buffer });
          }
        });
      });
      
      req.on('error', reject);
      if (data) req.write(JSON.stringify(data));
      req.end();
    });
  }
  
  async getModelVersion(modelId) {
    const response = await this.request('GET', `/models/${modelId}`);
    return response.latest_version.id;
  }
  
  async createPrediction(version, input) {
    return this.request('POST', '/predictions', { version, input });
  }
  
  async getPrediction(id) {
    return this.request('GET', `/predictions/${id}`);
  }
  
  async runWithPolling(version, input, onProgress = null) {
    const prediction = await this.createPrediction(version, input);
    const predictionId = prediction.id;
    
    let attempts = 0;
    const maxAttempts = config.polling.maxAttempts;
    const intervalMs = config.polling.intervalMs;
    
    while (attempts < maxAttempts) {
      await this.sleep(intervalMs);
      
      const status = await this.getPrediction(predictionId);
      
      if (onProgress) {
        onProgress({
          status: status.status,
          elapsed: attempts * intervalMs / 1000,
          predictionId,
        });
      }
      
      if (status.status === 'succeeded') {
        return status.output;
      } else if (status.status === 'failed') {
        throw new Error(`Prediction failed: ${status.error}`);
      } else if (status.status === 'canceled') {
        throw new Error('Prediction was canceled');
      }
      
      attempts++;
    }
    
    throw new Error(`Prediction timed out after ${maxAttempts * intervalMs / 1000}s`);
  }
  
  async runWithRetry(version, input, onProgress = null) {
    const maxAttempts = config.retry.maxAttempts;
    let lastError;
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await this.runWithPolling(version, input, onProgress);
      } catch (error) {
        lastError = error;
        
        if (attempt < maxAttempts) {
          const delay = config.retry.delayMs * Math.pow(config.retry.backoffMultiplier, attempt - 1);
          console.log(`   ⚠️ Attempt ${attempt} failed, retrying in ${delay/1000}s...`);
          await this.sleep(delay);
        }
      }
    }
    
    throw lastError;
  }
  
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = ReplicateClient;



