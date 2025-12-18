/**
 * Kelly's Email System
 * 
 * Centralized exports for the agentic email infrastructure.
 */

// Classification
export { 
  classifyEmail, 
  quickClassify, 
  shouldEscalate,
  type EmailClassification,
  type EmailInput 
} from './classifier';

// Templates
export { 
  ALL_TEMPLATES,
  SUPPORT_TEMPLATES,
  FEEDBACK_TEMPLATES,
  ENTERPRISE_TEMPLATES,
  PRESS_TEMPLATES,
  BILLING_TEMPLATES,
  PARTNER_TEMPLATES,
  FAMILY_TEMPLATES,
  KELLY_VOICE,
  getTemplateByIntent,
  getTemplatesByCategory,
  fillTemplate,
  getRandomFunFact,
  type EmailTemplate 
} from './kelly-templates';

// Response Generation
export { 
  generateResponse,
  QUICK_RESPONSES,
  type GeneratedResponse,
  type ResponseContext 
} from './response-generator';

// Escalation
export { 
  sendEscalationNotification,
  sendDailyDigest,
  type EscalationContext,
  type EscalationResult 
} from './escalation';
