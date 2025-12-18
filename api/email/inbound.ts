/**
 * Inbound Email Webhook
 * 
 * Receives emails from Resend Inbound and processes them through
 * Kelly's Agentic Email System:
 * 
 * 1. Verify webhook signature
 * 2. Store the email
 * 3. Classify with AI
 * 4. Generate response (auto or draft)
 * 5. Escalate if needed
 * 6. Send response (auto) or queue for approval (draft)
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import { classifyEmail, quickClassify, shouldEscalate, EmailClassification } from '../../lib/email/classifier';
import { generateResponse, GeneratedResponse } from '../../lib/email/response-generator';
import { sendEscalationNotification } from '../../lib/email/escalation';

const RESEND_API_URL = 'https://api.resend.com/emails';

// Initialize Supabase
const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

interface ResendInboundEmail {
  from: string;
  to: string;
  subject: string;
  text: string;
  html?: string;
  reply_to?: string;
  cc?: string[];
  attachments?: Array<{
    filename: string;
    content: string;
    content_type: string;
  }>;
  headers?: Record<string, string>;
  created_at?: string;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, x-resend-signature');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Verify Resend webhook signature (optional but recommended)
  const signature = req.headers['x-resend-signature'];
  const webhookSecret = process.env.RESEND_INBOUND_WEBHOOK_SECRET;
  
  if (webhookSecret && signature) {
    // TODO: Implement signature verification when Resend provides it
    // For now, we accept all requests
  }

  if (!supabaseUrl || !supabaseKey) {
    console.error('Supabase not configured');
    return res.status(500).json({ error: 'Database not configured' });
  }

  const supabase = createClient(supabaseUrl, supabaseKey);
  const email = req.body as ResendInboundEmail;

  // Validate required fields
  if (!email.from || !email.subject || !email.text) {
    return res.status(400).json({ 
      error: 'Missing required fields',
      required: ['from', 'subject', 'text']
    });
  }

  console.log(`[inbound] Received email from ${email.from}: ${email.subject}`);

  try {
    // Extract sender info
    const fromMatch = email.from.match(/^(?:"?([^"]*)"?\s)?<?([^>]+)>?$/);
    const fromName = fromMatch?.[1] || null;
    const fromEmail = fromMatch?.[2] || email.from;

    // Generate or find thread ID
    const inReplyTo = email.headers?.['in-reply-to'] || email.headers?.['In-Reply-To'];
    const messageId = email.headers?.['message-id'] || email.headers?.['Message-ID'] || crypto.randomUUID();
    const threadId = inReplyTo || messageId;

    // Check if this is a reply to an existing thread
    const { data: existingThread } = await supabase
      .from('email_threads')
      .select('id, category, status')
      .eq('thread_id', threadId)
      .single();

    // ============================================
    // STEP 1: Quick pre-classification
    // ============================================
    const quickResult = quickClassify({
      from: fromEmail,
      fromName: fromName || undefined,
      subject: email.subject,
      bodyText: email.text,
    });

    // If it's obvious spam, handle quickly
    if (quickResult?.category === 'spam') {
      console.log(`[inbound] Quick-classified as spam, ignoring: ${email.subject}`);
      
      // Still store it for auditing
      const { data: thread } = await supabase
        .from('email_threads')
        .insert({
          thread_id: threadId,
          from_email: fromEmail,
          from_name: fromName,
          subject: email.subject,
          category: 'spam',
          status: 'spam',
          urgency: 'low',
          requires_human: false,
        })
        .select()
        .single();

      if (thread) {
        await supabase.from('email_messages').insert({
          thread_id: thread.id,
          direction: 'inbound',
          from_email: fromEmail,
          from_name: fromName,
          to_email: email.to,
          subject: email.subject,
          body_text: email.text,
          body_html: email.html,
        });

        await logAction(supabase, thread.id, 'classified', 'kelly_ai', { 
          quick: true, 
          category: 'spam' 
        });
      }

      return res.status(200).json({ success: true, action: 'ignored_spam' });
    }

    // ============================================
    // STEP 2: Full AI Classification
    // ============================================
    let classification: EmailClassification;
    
    if (quickResult && quickResult.category) {
      // Merge quick result with defaults
      classification = {
        category: quickResult.category as EmailClassification['category'],
        sentiment: quickResult.sentiment || 'neutral',
        urgency: quickResult.urgency || 'normal',
        confidence: quickResult.confidence || 0.8,
        intent: null,
        summary: 'Quick-classified based on patterns',
        entities: {},
        requiresHuman: quickResult.requiresHuman || false,
        escalationTriggers: quickResult.escalationTriggers || [],
        suggestedResponseType: quickResult.suggestedResponseType || 'draft',
      };
    } else {
      // Full AI classification
      classification = await classifyEmail({
        from: fromEmail,
        fromName: fromName || undefined,
        subject: email.subject,
        bodyText: email.text,
      });
    }

    console.log(`[inbound] Classified as ${classification.category} (${classification.sentiment}, ${classification.urgency})`);

    // ============================================
    // STEP 3: Create or update thread
    // ============================================
    let threadDbId: string;

    if (existingThread) {
      // Update existing thread
      threadDbId = existingThread.id;
      
      await supabase
        .from('email_threads')
        .update({
          last_message_at: new Date().toISOString(),
          status: 'open', // Reopen if closed
        })
        .eq('id', threadDbId);

    } else {
      // Create new thread
      const { data: newThread, error: threadError } = await supabase
        .from('email_threads')
        .insert({
          thread_id: threadId,
          from_email: fromEmail,
          from_name: fromName,
          subject: email.subject,
          category: classification.category,
          urgency: classification.urgency,
          sentiment: classification.sentiment,
          status: classification.suggestedResponseType === 'auto' ? 'open' : 'pending_approval',
          requires_human: classification.requiresHuman,
          classification_confidence: classification.confidence,
          ai_summary: classification.summary,
          detected_intent: classification.intent,
          detected_entities: classification.entities,
        })
        .select()
        .single();

      if (threadError || !newThread) {
        console.error('Failed to create thread:', threadError);
        return res.status(500).json({ error: 'Failed to store email' });
      }

      threadDbId = newThread.id;
    }

    // Store the inbound message
    const { error: messageError } = await supabase
      .from('email_messages')
      .insert({
        thread_id: threadDbId,
        direction: 'inbound',
        from_email: fromEmail,
        from_name: fromName,
        to_email: email.to,
        reply_to: email.reply_to,
        cc: email.cc,
        subject: email.subject,
        body_text: email.text,
        body_html: email.html,
        attachments: email.attachments?.map(a => ({
          filename: a.filename,
          content_type: a.content_type,
          size: a.content.length,
        })),
      });

    if (messageError) {
      console.error('Failed to store message:', messageError);
    }

    await logAction(supabase, threadDbId, 'received', 'system', { 
      from: fromEmail 
    });

    await logAction(supabase, threadDbId, 'classified', 'kelly_ai', {
      category: classification.category,
      sentiment: classification.sentiment,
      urgency: classification.urgency,
      confidence: classification.confidence,
    });

    // ============================================
    // STEP 4: Determine action
    // ============================================
    const needsEscalation = shouldEscalate(classification);
    const canAutoRespond = 
      !needsEscalation && 
      classification.suggestedResponseType === 'auto' &&
      ['support', 'feedback'].includes(classification.category);

    // Generate response
    const response = await generateResponse({
      originalEmail: {
        from: fromEmail,
        fromName: fromName || undefined,
        subject: email.subject,
        bodyText: email.text,
      },
      classification,
      additionalContext: {
        help_url: 'https://curiouskelly.com/help',
        reset_password_url: 'https://curiouskelly.com/reset-password',
      },
    });

    console.log(`[inbound] Generated response (needsReview: ${response.needsReview}, canAutoRespond: ${canAutoRespond})`);

    // ============================================
    // STEP 5: Execute action
    // ============================================

    if (needsEscalation) {
      // Escalate immediately
      console.log(`[inbound] Escalating to ${process.env.ESCALATION_EMAIL || 'nicoletterankin@gmail.com'}`);
      
      await sendEscalationNotification({
        threadId: threadDbId,
        originalEmail: {
          from: fromEmail,
          fromName: fromName || undefined,
          subject: email.subject,
          bodyText: email.text,
          receivedAt: new Date(),
        },
        classification,
        draftResponse: {
          subject: response.subject,
          bodyText: response.bodyText,
        },
      });

      await supabase
        .from('email_threads')
        .update({
          escalated_to: process.env.ESCALATION_EMAIL || 'nicoletterankin@gmail.com',
          escalation_reason: classification.escalationTriggers.join(', ') || classification.category,
          escalated_at: new Date().toISOString(),
          status: 'escalated',
        })
        .eq('id', threadDbId);

      await logAction(supabase, threadDbId, 'escalated', 'kelly_ai', {
        reason: classification.escalationTriggers,
        to: process.env.ESCALATION_EMAIL,
      });

      // Also save the draft for review
      await supabase.from('email_messages').insert({
        thread_id: threadDbId,
        direction: 'draft',
        from_email: 'hello@curiouskelly.com',
        to_email: fromEmail,
        subject: response.subject,
        body_text: response.bodyText,
        body_html: response.bodyHtml,
        is_approved: false,
      });

      await logAction(supabase, threadDbId, 'drafted', 'kelly_ai', {
        template: response.templateUsed,
      });

      return res.status(200).json({
        success: true,
        action: 'escalated',
        threadId: threadDbId,
        classification: {
          category: classification.category,
          urgency: classification.urgency,
          sentiment: classification.sentiment,
        },
      });
    }

    if (canAutoRespond && !response.needsReview) {
      // Auto-respond immediately
      console.log(`[inbound] Auto-responding to ${fromEmail}`);
      
      const sendResult = await sendEmail(fromEmail, response.subject, response.bodyText, response.bodyHtml);

      if (sendResult.success) {
        await supabase.from('email_messages').insert({
          thread_id: threadDbId,
          direction: 'outbound',
          from_email: 'hello@curiouskelly.com',
          to_email: fromEmail,
          subject: response.subject,
          body_text: response.bodyText,
          body_html: response.bodyHtml,
          resend_message_id: sendResult.messageId,
          resend_status: 'sent',
          sent_at: new Date().toISOString(),
          is_approved: true,
        });

        await supabase
          .from('email_threads')
          .update({
            status: 'responded',
            responded_at: new Date().toISOString(),
          })
          .eq('id', threadDbId);

        await logAction(supabase, threadDbId, 'sent', 'kelly_ai', {
          template: response.templateUsed,
          messageId: sendResult.messageId,
        });

        return res.status(200).json({
          success: true,
          action: 'auto_responded',
          threadId: threadDbId,
          messageId: sendResult.messageId,
        });
      } else {
        console.error(`[inbound] Failed to send auto-response:`, sendResult.error);
        // Fall through to draft
      }
    }

    // Default: Save as draft for review
    console.log(`[inbound] Saving draft for review`);
    
    await supabase.from('email_messages').insert({
      thread_id: threadDbId,
      direction: 'draft',
      from_email: 'hello@curiouskelly.com',
      to_email: fromEmail,
      subject: response.subject,
      body_text: response.bodyText,
      body_html: response.bodyHtml,
      is_approved: false,
    });

    await supabase
      .from('email_threads')
      .update({ status: 'pending_approval' })
      .eq('id', threadDbId);

    await logAction(supabase, threadDbId, 'drafted', 'kelly_ai', {
      template: response.templateUsed,
      needsReview: response.needsReview,
      reviewReason: response.reviewReason,
    });

    return res.status(200).json({
      success: true,
      action: 'drafted',
      threadId: threadDbId,
      classification: {
        category: classification.category,
        urgency: classification.urgency,
        sentiment: classification.sentiment,
      },
    });

  } catch (error) {
    console.error('[inbound] Error processing email:', error);
    return res.status(500).json({
      error: 'Failed to process email',
      details: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}

// ============================================
// HELPER FUNCTIONS
// ============================================

async function sendEmail(
  to: string,
  subject: string,
  text: string,
  html: string
): Promise<{ success: boolean; messageId?: string; error?: string }> {
  const apiKey = process.env.RESEND_API_KEY;
  
  if (!apiKey) {
    return { success: false, error: 'RESEND_API_KEY not configured' };
  }

  try {
    const response = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly <hello@curiouskelly.com>',
        to,
        subject,
        text,
        html,
        reply_to: 'hello@curiouskelly.com',
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      return { success: false, error: data.message || 'Failed to send' };
    }

    return { success: true, messageId: data.id };

  } catch (error) {
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    };
  }
}

async function logAction(
  supabase: ReturnType<typeof createClient>,
  threadId: string,
  action: string,
  actor: string,
  details: Record<string, unknown>
): Promise<void> {
  try {
    await supabase.from('email_actions').insert({
      thread_id: threadId,
      action,
      actor,
      details,
    });
  } catch (error) {
    console.error('Failed to log action:', error);
  }
}
