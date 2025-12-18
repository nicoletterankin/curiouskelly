/**
 * Kelly's Response Generator
 * 
 * Uses AI to craft personalized, on-brand responses
 * based on email classification and templates.
 */

import { EmailClassification } from './classifier';
import { 
  EmailTemplate, 
  ALL_TEMPLATES, 
  fillTemplate, 
  getTemplatesByCategory,
  KELLY_VOICE 
} from './kelly-templates';

export interface GeneratedResponse {
  subject: string;
  bodyText: string;
  bodyHtml: string;
  templateUsed: string | null;
  confidence: number;
  needsReview: boolean;
  reviewReason?: string;
}

export interface ResponseContext {
  originalEmail: {
    from: string;
    fromName?: string;
    subject: string;
    bodyText: string;
  };
  classification: EmailClassification;
  additionalContext?: Record<string, string>; // Links, user data, etc.
}

const RESPONSE_GENERATION_PROMPT = `You are Kelly, the AI teacher for Curious Kelly. You're drafting a response to an email.

YOUR PERSONALITY:
${KELLY_VOICE.principles.map(p => `- ${p}`).join('\n')}

WORDS TO USE: ${KELLY_VOICE.wordsToUse.join(', ')}
WORDS TO AVOID: ${KELLY_VOICE.wordsToAvoid.join(', ')}

ORIGINAL EMAIL:
From: {{from}}
Subject: {{subject}}
Body:
{{body}}

CLASSIFICATION:
- Category: {{category}}
- Intent: {{intent}}
- Sentiment: {{sentiment}}
- Summary: {{summary}}

TEMPLATE TO USE AS A BASE:
{{template}}

ADDITIONAL CONTEXT:
{{context}}

YOUR TASK:
Write a personalized response that:
1. Acknowledges their specific situation
2. Provides genuinely helpful information
3. Uses the template as a starting point but personalizes it
4. Includes specific details from their email
5. Sounds warm and human, not robotic
6. Stays concise (under 200 words ideal)

Respond with JSON:
{
  "subject": "The email subject line",
  "body": "The full email body text",
  "needsReview": false,
  "reviewReason": null
}

If you're not confident about the response (unclear request, sensitive topic, etc.), set needsReview to true and explain why.`;

export async function generateResponse(context: ResponseContext): Promise<GeneratedResponse> {
  const { originalEmail, classification, additionalContext } = context;
  
  // Find the best template
  const template = findBestTemplate(classification);
  
  // For categories that always need review, skip AI and use template
  const alwaysReviewCategories = ['enterprise', 'press', 'billing', 'partner'];
  if (alwaysReviewCategories.includes(classification.category)) {
    return generateDraftFromTemplate(template, originalEmail, additionalContext || {});
  }

  // Try AI-powered personalization for support/feedback
  try {
    const aiResponse = await generateWithAI(context, template);
    return aiResponse;
  } catch (error) {
    console.error('AI response generation failed, falling back to template:', error);
    return generateDraftFromTemplate(template, originalEmail, additionalContext || {});
  }
}

async function generateWithAI(
  context: ResponseContext,
  template: EmailTemplate | null
): Promise<GeneratedResponse> {
  const apiKey = process.env.ANT_API_KEY || process.env.ANTHROPIC_API_KEY;
  
  if (!apiKey) {
    throw new Error('Missing AI API key');
  }

  const prompt = RESPONSE_GENERATION_PROMPT
    .replace('{{from}}', `${context.originalEmail.from}${context.originalEmail.fromName ? ` (${context.originalEmail.fromName})` : ''}`)
    .replace('{{subject}}', context.originalEmail.subject)
    .replace('{{body}}', context.originalEmail.bodyText.slice(0, 2000))
    .replace('{{category}}', context.classification.category)
    .replace('{{intent}}', context.classification.intent || 'general')
    .replace('{{sentiment}}', context.classification.sentiment)
    .replace('{{summary}}', context.classification.summary)
    .replace('{{template}}', template ? template.body : 'No specific template - use your judgment')
    .replace('{{context}}', context.additionalContext 
      ? Object.entries(context.additionalContext).map(([k, v]) => `${k}: ${v}`).join('\n')
      : 'None');

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-3-haiku-20240307',
      max_tokens: 1000,
      messages: [{ role: 'user', content: prompt }],
    }),
  });

  if (!response.ok) {
    throw new Error(`AI API error: ${response.status}`);
  }

  const data = await response.json();
  const content = data.content?.[0]?.text;

  if (!content) {
    throw new Error('Empty AI response');
  }

  // Parse JSON
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    throw new Error('Invalid AI response format');
  }

  const parsed = JSON.parse(jsonMatch[0]);
  
  return {
    subject: parsed.subject || `Re: ${context.originalEmail.subject}`,
    bodyText: parsed.body,
    bodyHtml: textToHtml(parsed.body),
    templateUsed: template?.id || null,
    confidence: parsed.needsReview ? 0.5 : 0.9,
    needsReview: parsed.needsReview || false,
    reviewReason: parsed.reviewReason || undefined,
  };
}

function generateDraftFromTemplate(
  template: EmailTemplate | null,
  originalEmail: { from: string; fromName?: string; subject: string; bodyText: string },
  additionalContext: Record<string, string>
): GeneratedResponse {
  if (!template) {
    // Fallback generic response
    return {
      subject: `Re: ${originalEmail.subject}`,
      bodyText: `Hi${originalEmail.fromName ? ` ${originalEmail.fromName.split(' ')[0]}` : ''}!

Thank you for reaching out. I've received your message and will get back to you soon.

Stay curious,
Kelly ✨`,
      bodyHtml: '',
      templateUsed: null,
      confidence: 0.5,
      needsReview: true,
      reviewReason: 'No suitable template found - needs personalization',
    };
  }

  // Extract first name
  const firstName = originalEmail.fromName?.split(' ')[0] || 'there';
  
  const variables: Record<string, string> = {
    name: firstName,
    ...additionalContext,
    response_content: '[PLEASE ADD SPECIFIC RESPONSE HERE]', // Placeholder for human
  };

  const { subject, body } = fillTemplate(template, variables);

  return {
    subject: subject.startsWith('Re:') ? subject : `Re: ${originalEmail.subject}`,
    bodyText: body,
    bodyHtml: textToHtml(body),
    templateUsed: template.id,
    confidence: 0.7,
    needsReview: true,
    reviewReason: 'Template-based draft needs personalization',
  };
}

function findBestTemplate(classification: EmailClassification): EmailTemplate | null {
  // First, try to find by exact intent
  if (classification.intent) {
    const intentMatch = ALL_TEMPLATES.find(t => t.intent === classification.intent);
    if (intentMatch) return intentMatch;
  }

  // Fall back to category-based templates
  const categoryTemplates = getTemplatesByCategory(classification.category);
  if (categoryTemplates.length > 0) {
    // Return the first (most generic) template for the category
    return categoryTemplates[0];
  }

  return null;
}

function textToHtml(text: string): string {
  // Convert markdown-like formatting to HTML
  let html = text
    // Escape HTML
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Line breaks
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');

  // Wrap in paragraphs
  html = `<p>${html}</p>`;

  // Add styling wrapper
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: Georgia, 'Times New Roman', serif; background-color: #f9fafb;">
  <div style="max-width: 520px; margin: 0 auto; padding: 40px 20px;">
    <div style="background: white; border-radius: 16px; padding: 32px; box-shadow: 0 4px 16px rgba(0,0,0,0.08);">
      <div style="font-size: 18px; color: #374151; line-height: 1.8;">
        ${html}
      </div>
    </div>
    <div style="text-align: center; margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;">
      <p style="font-size: 13px; color: #9ca3af; margin: 0;">
        ✨ Curious Kelly • <a href="https://curiouskelly.com" style="color: #d4a574;">curiouskelly.com</a>
      </p>
    </div>
  </div>
</body>
</html>`.trim();
}

/**
 * Quick auto-responses for common support queries
 * These can be sent immediately without AI
 */
export const QUICK_RESPONSES: Record<string, (name: string) => GeneratedResponse> = {
  password_reset: (name: string) => ({
    subject: 'Let\'s get you back in! 🔑',
    bodyText: `Hi ${name}!

I'm so sorry you're locked out — that's always frustrating! Let me help you get back to learning.

Here's your password reset link:
https://curiouskelly.com/reset-password

This link expires in 1 hour, so use it soon!

💡 Fun fact: The average person forgets 3 passwords per month. You're in very good company!

If you didn't request this reset, just ignore this email — your account is safe.

Stay curious,
Kelly ✨`,
    bodyHtml: '',
    templateUsed: 'password_reset',
    confidence: 0.95,
    needsReview: false,
  }),
  
  unsubscribe_confirm: (name: string) => ({
    subject: 'You\'re unsubscribed 💌',
    bodyText: `Hi ${name}!

I've removed you from daily lesson emails. You won't hear from me unless you ask to come back.

I'll miss our daily learning moments, but I understand! If you ever want to rejoin, just visit curiouskelly.com/settings.

Thank you for the time we had together.

Stay curious (even without me!),
Kelly ✨`,
    bodyHtml: '',
    templateUsed: 'unsubscribe_confirm',
    confidence: 0.95,
    needsReview: false,
  }),
};
