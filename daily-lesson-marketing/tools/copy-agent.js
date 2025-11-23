#!/usr/bin/env node

/**
 * Marketing Copy Agent CLI
 * AI-powered copy generation and validation for The Daily Lesson by Curious Kelly
 * 
 * Usage:
 *   node tools/copy-agent.js validate    # Validate existing copy
 *   node tools/copy-agent.js generate    # Generate new copy section
 *   node tools/copy-agent.js translate   # Translate EN → ES/PT
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load Marketing Copy Agent guidelines
const AGENT_GUIDELINES = fs.readFileSync(
  path.join(__dirname, '../MARKETING_COPY_AGENT.md'),
  'utf-8'
);

// Forbidden words list
const FORBIDDEN_WORDS = [
  'concierge',
  'cohort',
  'onboarding',
  'founding members',
  'pilot partners',
  'district-level',
  'enrollment'
];

// Required elements
const REQUIRED_ELEMENTS = {
  pricing: ['$4.99', '$49.99'],
  trial: ['7 days free', '7-day free trial'],
  languages: ['English', 'Spanish', 'Portuguese'],
  ageRange: ['ages 2', '102', 'age-adaptive']
};

/**
 * Validate copy against brand guidelines
 */
function validateCopy(filePath) {
  console.log(`\n🔍 Validating: ${filePath}\n`);
  
  const content = fs.readFileSync(filePath, 'utf-8');
  const errors = [];
  const warnings = [];

  // Check for forbidden words
  FORBIDDEN_WORDS.forEach(word => {
    const regex = new RegExp(`\\b${word}\\b`, 'gi');
    const matches = content.match(regex);
    if (matches) {
      errors.push(`❌ Found forbidden word "${word}" (${matches.length} occurrences)`);
    }
  });

  // Check for pricing mentions
  const hasPricing = REQUIRED_ELEMENTS.pricing.some(price => 
    content.includes(price)
  );
  if (!hasPricing) {
    warnings.push(`⚠️  No pricing found ($4.99/month or $49.99/year)`);
  }

  // Check for trial mention
  const hasTrial = REQUIRED_ELEMENTS.trial.some(trial => 
    content.toLowerCase().includes(trial.toLowerCase())
  );
  if (!hasTrial) {
    warnings.push(`⚠️  No free trial mention found`);
  }

  // Check language mentions
  const languageMentions = REQUIRED_ELEMENTS.languages.filter(lang =>
    content.includes(lang)
  );
  if (languageMentions.length < 3) {
    warnings.push(`⚠️  Not all three languages mentioned (found: ${languageMentions.join(', ')})`);
  }

  // Report results
  if (errors.length === 0 && warnings.length === 0) {
    console.log('✅ All checks passed!\n');
    return true;
  }

  if (errors.length > 0) {
    console.log('ERRORS:\n');
    errors.forEach(e => console.log(e));
    console.log('');
  }

  if (warnings.length > 0) {
    console.log('WARNINGS:\n');
    warnings.forEach(w => console.log(w));
    console.log('');
  }

  return errors.length === 0;
}

/**
 * Generate copy using AI (Claude API integration)
 */
async function generateCopy(section, audience) {
  console.log(`\n🤖 Generating ${section} copy for ${audience}...\n`);
  
  // Check for API key
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error('❌ Missing ANTHROPIC_API_KEY environment variable');
    console.log('\nTo use copy generation:');
    console.log('1. Get your API key from https://console.anthropic.com/');
    console.log('2. Set environment variable: export ANTHROPIC_API_KEY=your-key-here');
    console.log('3. Run this command again\n');
    return;
  }

  const prompt = buildPrompt(section, audience);
  
  console.log('📝 Prompt:');
  console.log('---');
  console.log(prompt);
  console.log('---\n');
  
  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 2000,
        messages: [{
          role: 'user',
          content: prompt
        }]
      })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    const generatedCopy = data.content[0].text;
    
    console.log('✨ Generated copy:\n');
    console.log(generatedCopy);
    console.log('\n');
    
    // Save to file
    const outputPath = path.join(__dirname, `../generated-copy-${Date.now()}.md`);
    fs.writeFileSync(outputPath, `# Generated Copy\n\n${generatedCopy}\n\n---\n\nPrompt used:\n${prompt}`);
    console.log(`💾 Saved to: ${outputPath}\n`);
    
  } catch (error) {
    console.error('❌ Error generating copy:', error.message);
  }
}

/**
 * Build prompt for AI copy generation
 */
function buildPrompt(section, audience) {
  return `You are the Marketing Copy Agent for The Daily Lesson by Curious Kelly.

CONTEXT:
${AGENT_GUIDELINES}

TASK:
Generate ${section} copy for the ${audience} audience.

REQUIREMENTS:
- Follow the brand voice and messaging framework above
- Include pricing: $4.99/month or $49.99/year
- Mention 7-day free trial (no credit card)
- Highlight: 8-minute lessons, age-adaptive, 3 languages
- Avoid all forbidden words (concierge, cohort, onboarding, etc.)
- Keep tone warm, accessible, and action-oriented
- Include a clear call-to-action

OUTPUT FORMAT:
Provide the copy in clean markdown format, ready to paste into our i18n dictionary.`;
}

/**
 * Translate copy to Spanish and Portuguese
 */
async function translateCopy(text, targetLang) {
  console.log(`\n🌍 Translating to ${targetLang}...\n`);
  
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error('❌ Missing ANTHROPIC_API_KEY environment variable\n');
    return;
  }

  const langName = targetLang === 'es' ? 'Spanish (Spain)' : 'Brazilian Portuguese';
  
  const prompt = `You are translating marketing copy for The Daily Lesson by Curious Kelly.

GUIDELINES:
- Translate to ${langName}
- Keep "Kelly" and "The Daily Lesson" in English
- Maintain warm, accessible tone
- Preserve all pricing and numbers
- Keep the same emotional impact

ORIGINAL TEXT:
${text}

Provide the translation only, no explanations.`;

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 2000,
        messages: [{
          role: 'user',
          content: prompt
        }]
      })
    });

    const data = await response.json();
    const translation = data.content[0].text;
    
    console.log(`✨ ${langName} translation:\n`);
    console.log(translation);
    console.log('\n');
    
    return translation;
    
  } catch (error) {
    console.error('❌ Error translating:', error.message);
  }
}

/**
 * CLI command router
 */
async function main() {
  const command = process.argv[2];
  const arg1 = process.argv[3];
  const arg2 = process.argv[4];

  console.log('\n🎨 Marketing Copy Agent for The Daily Lesson by Curious Kelly\n');

  switch (command) {
    case 'validate':
      const fileToValidate = arg1 || 'src/lib/i18n/en-us.ts';
      const fullPath = path.isAbsolute(fileToValidate) 
        ? fileToValidate 
        : path.join(process.cwd(), fileToValidate);
      if (!fs.existsSync(fullPath)) {
        console.error(`❌ File not found: ${fullPath}\n`);
        process.exit(1);
      }
      const isValid = validateCopy(fullPath);
      process.exit(isValid ? 0 : 1);
      break;

    case 'generate':
      const section = arg1 || 'hero';
      const audience = arg2 || 'adults';
      await generateCopy(section, audience);
      break;

    case 'translate':
      const textFile = arg1;
      if (!textFile) {
        console.error('❌ Usage: node copy-agent.js translate <file-with-text>\n');
        process.exit(1);
      }
      const textPath = path.join(__dirname, textFile);
      if (!fs.existsSync(textPath)) {
        console.error(`❌ File not found: ${textPath}\n`);
        process.exit(1);
      }
      const textContent = fs.readFileSync(textPath, 'utf-8');
      await translateCopy(textContent, 'es');
      await translateCopy(textContent, 'pt');
      break;

    default:
      console.log('Usage:');
      console.log('  node tools/copy-agent.js validate [file]');
      console.log('  node tools/copy-agent.js generate [section] [audience]');
      console.log('  node tools/copy-agent.js translate <file>');
      console.log('\nExamples:');
      console.log('  node tools/copy-agent.js validate');
      console.log('  node tools/copy-agent.js generate hero adults');
      console.log('  node tools/copy-agent.js translate copy.txt\n');
  }
}

main().catch(console.error);

