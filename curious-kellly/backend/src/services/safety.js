/**
 * Safety & Moderation Service
 * Uses OpenAI Moderation API + custom rules
 * Target: ≥98% precision, ≥95% recall
 */

const OpenAI = require('openai');

class SafetyService {
  constructor() {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY not found in environment variables');
    }

    this.client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });

    // Custom safety rules for children's content
    this.bannedWords = [
      // Add specific words that should never appear in content
      // This list would be expanded based on testing
    ];

    // Categories we care about for kids' education
    this.criticalCategories = [
      'sexual',
      'hate',
      'harassment',
      'self-harm',
      'sexual/minors',
      'hate/threatening',
      'violence/graphic'
    ];
  }

  /**
   * Moderate content using OpenAI Moderation API
   */
  async moderateContent(text) {
    try {
      const startTime = Date.now();

      // Call OpenAI Moderation API
      const moderation = await this.client.moderations.create({
        input: text
      });

      const result = moderation.results[0];
      const latency = Date.now() - startTime;

      // Check if any critical categories are flagged
      const flaggedCategories = [];
      let isFlagged = false;

      for (const category of this.criticalCategories) {
        if (result.categories[category]) {
          flaggedCategories.push({
            category,
            score: result.category_scores[category],
            flagged: true
          });
          isFlagged = true;
        }
      }

      // Apply custom rules
      const customViolations = this.checkCustomRules(text);
      if (customViolations.length > 0) {
        isFlagged = true;
      }

      return {
        safe: !isFlagged,
        flagged: isFlagged,
        categories: flaggedCategories,
        customViolations,
        allCategories: result.categories,
        allScores: result.category_scores,
        latency,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Moderation error:', error);
      // Fail safe: if moderation fails, block the content
      return {
        safe: false,
        flagged: true,
        error: error.message,
        failsafe: true,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Check custom safety rules
   */
  checkCustomRules(text) {
    const violations = [];
    const lowerText = text.toLowerCase();

    // Check for banned words
    for (const word of this.bannedWords) {
      if (lowerText.includes(word)) {
        violations.push({
          rule: 'banned-word',
          match: word,
          severity: 'high'
        });
      }
    }

    // Check for personal information patterns
    const patterns = {
      email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/,
      phone: /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/,
      ssn: /\b\d{3}-\d{2}-\d{4}\b/,
      creditCard: /\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b/
    };

    for (const [type, pattern] of Object.entries(patterns)) {
      if (pattern.test(text)) {
        violations.push({
          rule: 'personal-info',
          type,
          severity: 'medium'
        });
      }
    }

    // Check for requests to ignore safety
    const jailbreakPatterns = [
      'ignore previous instructions',
      'forget your rules',
      'act as if',
      'pretend you are',
      'roleplay as',
      'disregard safety'
    ];

    for (const pattern of jailbreakPatterns) {
      if (lowerText.includes(pattern)) {
        violations.push({
          rule: 'jailbreak-attempt',
          match: pattern,
          severity: 'high'
        });
      }
    }

    return violations;
  }

  /**
   * Safe-completion: Rewrite unsafe content to be safe
   * (For use when AI generates something borderline)
   */
  async safeCompletion(unsafeText) {
    try {
      const systemPrompt = `You are a content safety assistant. 
Rewrite the following text to make it appropriate for children ages 2-102.
Remove any inappropriate content while preserving the core educational message.
If the content cannot be made safe, respond with: "I can't help with that topic."`;

      const completion = await this.client.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: unsafeText }
        ],
        max_tokens: 200,
        temperature: 0.3
      });

      return {
        success: true,
        safeVersion: completion.choices[0].message.content,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        safeVersion: "I can't help with that topic.",
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Age-appropriate content check
   * Additional layer for children's content
   */
  isAgeAppropriate(text, age) {
    // Topics that are only appropriate for older learners
    const matureTopics = [
      'war', 'violence', 'death', 'politics', 'religion',
      'drugs', 'alcohol', 'weapons', 'crime'
    ];

    const lowerText = text.toLowerCase();
    
    // Very young children (2-5)
    if (age <= 5) {
      for (const topic of matureTopics) {
        if (lowerText.includes(topic)) {
          return {
            appropriate: false,
            reason: `Topic "${topic}" not suitable for age ${age}`,
            recommendedMinAge: 13
          };
        }
      }
    }

    // Children (6-12)
    if (age <= 12) {
      const sensitiveTopics = ['violence', 'death', 'drugs', 'weapons', 'crime'];
      for (const topic of sensitiveTopics) {
        if (lowerText.includes(topic)) {
          return {
            appropriate: false,
            reason: `Topic "${topic}" requires careful framing for age ${age}`,
            recommendedMinAge: 13
          };
        }
      }
    }

    return {
      appropriate: true,
      age
    };
  }
}

module.exports = SafetyService;














