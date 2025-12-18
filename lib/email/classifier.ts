/**
 * Kelly's Email Classifier
 * 
 * Uses Claude API to intelligently classify incoming emails:
 * - Category (support, enterprise, press, etc.)
 * - Sentiment (positive, neutral, negative, angry)
 * - Urgency (low, normal, high, critical)
 * - Intent detection (password_reset, refund_request, etc.)
 * - Escalation triggers
 */

export interface EmailClassification {
  category: 'support' | 'billing' | 'enterprise' | 'press' | 'family' | 'partner' | 'feedback' | 'spam' | 'other';
  sentiment: 'positive' | 'neutral' | 'negative' | 'angry';
  urgency: 'low' | 'normal' | 'high' | 'critical';
  confidence: number; // 0.0 to 1.0
  intent: string | null; // password_reset, refund_request, pricing_inquiry, etc.
  summary: string; // Brief AI summary
  entities: Record<string, string>; // Extracted entities
  requiresHuman: boolean;
  escalationTriggers: string[]; // Reasons for escalation
  suggestedResponseType: 'auto' | 'draft' | 'escalate';
}

export interface EmailInput {
  from: string;
  fromName?: string;
  subject: string;
  bodyText: string;
  bodyHtml?: string;
}

const CLASSIFICATION_PROMPT = `You are Kelly's email classification system. Analyze the incoming email and classify it.

CATEGORIES:
- support: Password resets, how-to questions, technical issues, account help
- billing: Refunds, charges, subscription changes, payment issues
- enterprise: Schools, businesses, organizations, volume licensing, custom solutions
- press: Journalists, media outlets, interview requests, publication inquiries
- family: Parents asking about children under 13, parental controls, family accounts
- partner: Affiliates, integration requests, business partnerships, collaborations
- feedback: Product suggestions, compliments, feature requests, general opinions
- spam: Marketing emails, scams, unrelated solicitations
- other: Anything that doesn't fit above

SENTIMENT:
- positive: Happy, thankful, enthusiastic
- neutral: Matter-of-fact, informational
- negative: Disappointed, frustrated, complaining
- angry: Furious, threatening, hostile

URGENCY:
- low: General inquiry, no time pressure
- normal: Standard request, reasonable timeframe
- high: Time-sensitive, frustrated customer, business inquiry
- critical: Legal threats, media deadline, major issue

ESCALATION TRIGGERS (any of these = requires human):
- Mentions of: lawyer, legal, lawsuit, attorney, sue
- Refund requests over $50
- Enterprise deals (any school/business inquiry)
- Press/media from any outlet
- Angry or hostile tone
- Mentions of: cancel, unsubscribe with frustration
- Complaints about Kelly herself
- Anything involving children under 13

RESPONSE TYPE:
- auto: Can be auto-responded (support, positive feedback)
- draft: Need human approval before sending (enterprise, press, billing)
- escalate: Immediately notify nicoletterankin@gmail.com (critical issues)

Respond in JSON format only:
{
  "category": "support",
  "sentiment": "neutral",
  "urgency": "normal",
  "confidence": 0.95,
  "intent": "password_reset",
  "summary": "User forgot password and needs reset link",
  "entities": { "issue": "password", "urgency_reason": null },
  "requiresHuman": false,
  "escalationTriggers": [],
  "suggestedResponseType": "auto"
}`;

export async function classifyEmail(email: EmailInput): Promise<EmailClassification> {
  const apiKey = process.env.ANT_API_KEY || process.env.ANTHROPIC_API_KEY;
  
  if (!apiKey) {
    console.error('Missing ANTHROPIC_API_KEY or ANT_API_KEY');
    // Return safe default that requires human review
    return getDefaultClassification('Missing AI API key');
  }

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-3-haiku-20240307', // Fast and cheap for classification
        max_tokens: 500,
        messages: [
          {
            role: 'user',
            content: `${CLASSIFICATION_PROMPT}

EMAIL TO CLASSIFY:
From: ${email.from}${email.fromName ? ` (${email.fromName})` : ''}
Subject: ${email.subject}
Body:
${email.bodyText.slice(0, 3000)}` // Limit body length
          }
        ]
      }),
    });

    if (!response.ok) {
      console.error('Claude API error:', response.status, await response.text());
      return getDefaultClassification('AI API error');
    }

    const data = await response.json();
    const content = data.content?.[0]?.text;

    if (!content) {
      return getDefaultClassification('Empty AI response');
    }

    // Parse JSON from response
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return getDefaultClassification('Invalid AI response format');
    }

    const classification = JSON.parse(jsonMatch[0]) as EmailClassification;
    
    // Validate and sanitize
    return sanitizeClassification(classification);

  } catch (error) {
    console.error('Email classification error:', error);
    return getDefaultClassification('Classification error');
  }
}

function getDefaultClassification(reason: string): EmailClassification {
  return {
    category: 'other',
    sentiment: 'neutral',
    urgency: 'normal',
    confidence: 0,
    intent: null,
    summary: `Could not classify: ${reason}`,
    entities: {},
    requiresHuman: true, // When in doubt, ask a human
    escalationTriggers: [reason],
    suggestedResponseType: 'draft',
  };
}

function sanitizeClassification(input: Partial<EmailClassification>): EmailClassification {
  const validCategories = ['support', 'billing', 'enterprise', 'press', 'family', 'partner', 'feedback', 'spam', 'other'];
  const validSentiments = ['positive', 'neutral', 'negative', 'angry'];
  const validUrgencies = ['low', 'normal', 'high', 'critical'];
  const validResponseTypes = ['auto', 'draft', 'escalate'];

  return {
    category: validCategories.includes(input.category || '') 
      ? input.category as EmailClassification['category'] 
      : 'other',
    sentiment: validSentiments.includes(input.sentiment || '') 
      ? input.sentiment as EmailClassification['sentiment'] 
      : 'neutral',
    urgency: validUrgencies.includes(input.urgency || '') 
      ? input.urgency as EmailClassification['urgency'] 
      : 'normal',
    confidence: typeof input.confidence === 'number' 
      ? Math.min(1, Math.max(0, input.confidence)) 
      : 0.5,
    intent: typeof input.intent === 'string' ? input.intent : null,
    summary: typeof input.summary === 'string' ? input.summary.slice(0, 500) : 'No summary',
    entities: typeof input.entities === 'object' && input.entities !== null 
      ? input.entities 
      : {},
    requiresHuman: input.requiresHuman === true,
    escalationTriggers: Array.isArray(input.escalationTriggers) 
      ? input.escalationTriggers.filter(t => typeof t === 'string') 
      : [],
    suggestedResponseType: validResponseTypes.includes(input.suggestedResponseType || '') 
      ? input.suggestedResponseType as EmailClassification['suggestedResponseType'] 
      : 'draft',
  };
}

/**
 * Quick keyword-based pre-classification for obvious cases
 * Runs before AI classification to catch spam and obvious patterns
 */
export function quickClassify(email: EmailInput): Partial<EmailClassification> | null {
  const lowerSubject = email.subject.toLowerCase();
  const lowerBody = email.bodyText.toLowerCase();
  const combined = `${lowerSubject} ${lowerBody}`;
  const fromDomain = email.from.split('@')[1]?.toLowerCase() || '';

  // Obvious spam patterns
  const spamPatterns = [
    /\bunsubscribe\s+from\s+all/i,
    /\bcongratulations.*won/i,
    /\bnigerian\s+prince/i,
    /\bclick\s+here\s+to\s+claim/i,
    /\bact\s+now.*limited\s+time/i,
    /\bdear\s+valued\s+customer/i,
  ];

  if (spamPatterns.some(p => p.test(combined))) {
    return {
      category: 'spam',
      sentiment: 'neutral',
      urgency: 'low',
      confidence: 0.95,
      requiresHuman: false,
      suggestedResponseType: 'auto', // Auto-ignore
    };
  }

  // Known press domains
  const pressDomains = [
    'nytimes.com', 'wsj.com', 'washingtonpost.com', 'bbc.com', 'cnn.com',
    'techcrunch.com', 'wired.com', 'theverge.com', 'engadget.com', 'mashable.com',
    'forbes.com', 'businessinsider.com', 'bloomberg.com', 'reuters.com',
  ];

  if (pressDomains.includes(fromDomain)) {
    return {
      category: 'press',
      urgency: 'high',
      requiresHuman: true,
      escalationTriggers: ['Press inquiry from recognized outlet'],
      suggestedResponseType: 'escalate',
    };
  }

  // Education domains
  if (fromDomain.endsWith('.edu') || fromDomain.endsWith('.k12.')) {
    return {
      category: 'enterprise',
      urgency: 'high',
      requiresHuman: true,
      escalationTriggers: ['Educational institution inquiry'],
      suggestedResponseType: 'draft',
    };
  }

  // Legal keywords - immediate escalation
  const legalKeywords = ['lawyer', 'attorney', 'legal action', 'lawsuit', 'sue you', 'court'];
  if (legalKeywords.some(k => combined.includes(k))) {
    return {
      category: 'billing', // Often billing-related
      sentiment: 'angry',
      urgency: 'critical',
      requiresHuman: true,
      escalationTriggers: ['Legal language detected'],
      suggestedResponseType: 'escalate',
    };
  }

  return null; // Let AI classify
}

/**
 * Determines if an email should be escalated based on rules
 */
export function shouldEscalate(
  classification: EmailClassification,
  escalationRules?: Array<{
    match_category?: string[];
    match_sentiment?: string[];
    match_keywords?: string[];
  }>
): boolean {
  // Always escalate these
  if (classification.suggestedResponseType === 'escalate') return true;
  if (classification.urgency === 'critical') return true;
  if (classification.sentiment === 'angry') return true;
  if (classification.escalationTriggers.length > 0) return true;
  
  // Categories that always escalate
  const alwaysEscalateCategories = ['enterprise', 'press', 'billing', 'partner'];
  if (alwaysEscalateCategories.includes(classification.category)) return true;

  return false;
}
