#!/usr/bin/env npx tsx
/**
 * Configure Vercel Edge Config and Blob Storage
 * 
 * This script uses the Vercel API to:
 * 1. Create Edge Config
 * 2. Create Blob Storage buckets
 * 3. Set environment variables
 * 
 * Usage:
 *   npx tsx scripts/configure-vercel-edge.ts
 */

import 'dotenv/config';

const VERCEL_API_BASE = 'https://api.vercel.com';
const TEAM_ID = 'lotd';
const PROJECT_NAME = 'curiouskelly';

async function getVercelToken(): Promise<string> {
  const token = process.env.VERCEL_TOKEN;
  if (!token) {
    throw new Error('VERCEL_TOKEN environment variable not set. Get it from: https://vercel.com/account/tokens');
  }
  return token;
}

async function apiRequest(
  method: string,
  endpoint: string,
  token: string,
  body?: any
): Promise<any> {
  const url = `${VERCEL_API_BASE}${endpoint}`;
  const headers: Record<string, string> = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };

  const options: RequestInit = {
    method,
    headers,
  };

  if (body) {
    options.body = JSON.stringify(body);
  }

  const response = await fetch(url, options);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${JSON.stringify(data)}`);
  }

  return data;
}

async function getProjectId(token: string): Promise<string> {
  const projects = await apiRequest('GET', `/v9/projects?teamId=${TEAM_ID}`, token);
  const project = projects.projects.find((p: any) => p.name === PROJECT_NAME);
  
  if (!project) {
    throw new Error(`Project "${PROJECT_NAME}" not found. Available projects: ${projects.projects.map((p: any) => p.name).join(', ')}`);
  }
  
  return project.id;
}

async function createEdgeConfig(token: string): Promise<string> {
  console.log('📦 Creating Edge Config...');
  
  try {
    const config = await apiRequest('POST', `/v1/edge-config`, token, {
      name: 'curious-kelly-lessons',
      slug: 'curious-kelly-lessons',
      items: [],
    });
    
    console.log(`✅ Edge Config created: ${config.id}`);
    return config.id;
  } catch (error: any) {
    if (error.message.includes('already exists') || error.message.includes('duplicate')) {
      console.log('⚠️  Edge Config already exists, fetching existing...');
      const configs = await apiRequest('GET', `/v1/edge-config?teamId=${TEAM_ID}`, token);
      const existing = configs.configs?.find((c: any) => c.name === 'curious-kelly-lessons' || c.slug === 'curious-kelly-lessons');
      if (existing) {
        console.log(`✅ Using existing Edge Config: ${existing.id}`);
        return existing.id;
      }
    }
    throw error;
  }
}

async function createBlobBucket(token: string, bucketName: string): Promise<void> {
  console.log(`📦 Creating Blob bucket: ${bucketName}...`);
  
  try {
    // Try using Vercel CLI blob command first
    const { execSync } = require('child_process');
    try {
      execSync(`vercel blob create ${bucketName} --public`, { stdio: 'inherit' });
      console.log(`✅ Blob bucket created via CLI: ${bucketName}`);
      return;
    } catch (cliError) {
      // Fallback to API
      console.log(`⚠️  CLI failed, trying API...`);
    }
    
    // API fallback
    await apiRequest('POST', `/v2/storage/buckets?teamId=${TEAM_ID}`, token, {
      name: bucketName,
      public: true,
    });
    console.log(`✅ Blob bucket created via API: ${bucketName}`);
  } catch (error: any) {
    if (error.message.includes('already exists') || error.message.includes('duplicate')) {
      console.log(`⚠️  Blob bucket already exists: ${bucketName}`);
    } else {
      throw error;
    }
  }
}

async function setEnvironmentVariable(
  token: string,
  projectId: string,
  key: string,
  value: string,
  environments: string[] = ['production', 'preview', 'development']
): Promise<void> {
  console.log(`🔐 Setting environment variable: ${key}...`);
  
  for (const env of environments) {
    try {
      await apiRequest('POST', `/v10/projects/${projectId}/env`, token, {
        key,
        value,
        type: 'encrypted',
        target: [env],
      });
      console.log(`✅ Set ${key} for ${env}`);
    } catch (error: any) {
      if (error.message.includes('already exists')) {
        console.log(`⚠️  ${key} already exists for ${env}, updating...`);
        // Update existing env var
        const envs = await apiRequest('GET', `/v10/projects/${projectId}/env`, token);
        const existing = envs.envs.find((e: any) => e.key === key && e.target.includes(env));
        if (existing) {
          await apiRequest('PATCH', `/v10/projects/${projectId}/env/${existing.id}`, token, {
            value,
            target: [env],
          });
          console.log(`✅ Updated ${key} for ${env}`);
        }
      } else {
        throw error;
      }
    }
  }
}

async function main() {
  console.log('🚀 Configuring Vercel Edge Optimization...\n');
  
  try {
    const token = await getVercelToken();
    console.log('✅ Vercel token found\n');
    
    // Get project ID
    const projectId = await getProjectId(token);
    console.log(`✅ Project ID: ${projectId}\n`);
    
    // Create Edge Config
    const edgeConfigId = await createEdgeConfig(token);
    console.log('');
    
    // Get Edge Config connection string
    const edgeConfigs = await apiRequest('GET', `/v1/edge-config`, token);
    const config = edgeConfigs.configs.find((c: any) => c.id === edgeConfigId);
    const connectionString = config?.connectionString;
    
    if (!connectionString) {
      throw new Error('Could not retrieve Edge Config connection string');
    }
    
    console.log(`📋 Edge Config Connection String: ${connectionString}\n`);
    
    // Create Blob buckets
    await createBlobBucket(token, 'curious-kelly-videos');
    await createBlobBucket(token, 'curious-kelly-audio');
    await createBlobBucket(token, 'curious-kelly-visuals');
    console.log('');
    
    // Generate sync secret
    const syncSecret = Array.from(crypto.getRandomValues(new Uint8Array(32)))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
    
    // Set environment variables
    await setEnvironmentVariable(token, projectId, 'EDGE_CONFIG', connectionString);
    await setEnvironmentVariable(token, projectId, 'EDGE_CONFIG_SYNC_SECRET', syncSecret);
    
    console.log('\n✅ Configuration complete!');
    console.log('\n📋 Summary:');
    console.log(`   - Edge Config ID: ${edgeConfigId}`);
    console.log(`   - Edge Config Connection: ${connectionString}`);
    console.log(`   - Sync Secret: ${syncSecret}`);
    console.log(`   - Blob Buckets: curious-kelly-videos, curious-kelly-audio, curious-kelly-visuals`);
    console.log('\n🎯 Next Steps:');
    console.log('   1. Run initial sync: npm run sync-edge-config');
    console.log('   2. Migrate assets: npx tsx scripts/migrate-to-blob.ts --dry-run');
    
  } catch (error: any) {
    console.error('\n❌ Error:', error.message);
    console.error('\n💡 Troubleshooting:');
    console.error('   1. Make sure VERCEL_TOKEN is set in environment');
    console.error('   2. Get token from: https://vercel.com/account/tokens');
    console.error('   3. Make sure you have access to the "lotd" team');
    process.exit(1);
  }
}

main();

