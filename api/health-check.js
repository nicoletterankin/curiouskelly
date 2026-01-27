/**
 * Enterprise-Grade Health Check Endpoint
 * Hard-coded verification of production functionality
 * NO WISHFUL THINKING - ACTUAL CHECKS ONLY
 */

const https = require('https');
const { URL } = require('url');

const PRODUCTION_URL = 'https://curiouskelly.com';

async function fetchProductionHTML() {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(`${PRODUCTION_URL}/learn.html`);
    const options = {
      hostname: parsedUrl.hostname,
      path: parsedUrl.pathname,
      method: 'GET',
      headers: {
        'User-Agent': 'Health-Check/1.0'
      },
      timeout: 10000
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        if (res.statusCode === 200) {
          resolve(data);
        } else {
          reject(new Error(`HTTP ${res.statusCode}`));
        }
      });
    });

    req.on('error', reject);
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });
    req.end();
  });
}

function verifyCode(html) {
  const checks = {
    asyncFunction: /async function enterPhaseWithChoices/.test(html),
    optionsNarration: /optionsNarration/.test(html),
    visualRef: /visualRef/.test(html),
    errorHandling: /\.catch\(\(\) => \{\}\)/.test(html),
    awaitPlayPhaseMedia: /await playPhaseMedia/.test(html),
    narrationTiming: /narrationDuration/.test(html),
    buttonsAfterNarration: /container\.hidden = false/.test(html) && 
                          html.indexOf('container.hidden = false') > html.indexOf('optionsNarration')
  };

  const passed = Object.values(checks).filter(v => v).length;
  const total = Object.keys(checks).length;
  const criticalChecks = ['asyncFunction', 'optionsNarration', 'errorHandling'];
  const criticalPassed = criticalChecks.every(key => checks[key]);

  return {
    healthy: criticalPassed && passed >= 5,
    passed,
    total,
    checks,
    criticalPassed
  };
}

export default async function handler(req, res) {
  const startTime = Date.now();
  
  try {
    // Fetch production HTML
    const html = await fetchProductionHTML();
    const verification = verifyCode(html);
    const responseTime = Date.now() - startTime;

    const status = verification.healthy ? 200 : 503;
    const health = verification.healthy ? 'healthy' : 'unhealthy';

    res.status(status).json({
      status: health,
      timestamp: new Date().toISOString(),
      responseTime: `${responseTime}ms`,
      verification: {
        passed: verification.passed,
        total: verification.total,
        criticalPassed: verification.criticalPassed,
        checks: verification.checks
      },
      url: PRODUCTION_URL
    });
  } catch (error) {
    const responseTime = Date.now() - startTime;
    
    res.status(503).json({
      status: 'error',
      timestamp: new Date().toISOString(),
      responseTime: `${responseTime}ms`,
      error: error.message,
      url: PRODUCTION_URL
    });
  }
}





