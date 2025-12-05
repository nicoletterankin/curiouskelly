/**
 * Daily Lesson Email API
 * 
 * Sends daily lesson reminder emails to subscribed users.
 * Can be triggered by:
 * - Cron job (recommended: daily at 7am in user's timezone)
 * - Manual trigger for testing
 * - Batch processing for all users
 * 
 * Environment Variables:
 * - RESEND_API_KEY: Your Resend API key
 * - DAILY_EMAIL_API_KEY: Secret key to authorize daily email sends
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { sendEmail, sendBatchEmails, EMAIL_TAGS } from './lib/resend';
import { dailyLessonEmail } from './lib/email-templates';

interface DailyLessonRequest {
  // For single email
  email?: string;
  name?: string;
  
  // Lesson details
  lessonTitle: string;
  lessonEmoji: string;
  lessonCategory: string;
  dayNumber: number;
  lessonUrl?: string;
  
  // For batch emails
  batch?: Array<{
    email: string;
    name?: string;
  }>;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Verify API key
  const authHeader = req.headers.authorization;
  const expectedKey = process.env.DAILY_EMAIL_API_KEY;
  
  if (expectedKey && authHeader !== `Bearer ${expectedKey}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  try {
    const body = req.body as DailyLessonRequest;
    
    const { lessonTitle, lessonEmoji, lessonCategory, dayNumber, lessonUrl } = body;
    
    // Validate required lesson fields
    if (!lessonTitle || !lessonEmoji || !lessonCategory || !dayNumber) {
      return res.status(400).json({ 
        error: 'Missing required lesson fields',
        required: ['lessonTitle', 'lessonEmoji', 'lessonCategory', 'dayNumber']
      });
    }
    
    const finalLessonUrl = lessonUrl || `https://curiouskelly.com/day/${dayNumber}`;

    // Handle batch emails
    if (body.batch && Array.isArray(body.batch)) {
      if (body.batch.length === 0) {
        return res.status(400).json({ error: 'Batch array is empty' });
      }
      
      if (body.batch.length > 100) {
        return res.status(400).json({ 
          error: 'Batch size exceeds maximum of 100',
          tip: 'Split into multiple requests for larger batches'
        });
      }

      const emails = body.batch.map(user => {
        const emailContent = dailyLessonEmail(
          user.name || 'friend',
          lessonTitle,
          lessonEmoji,
          lessonCategory,
          dayNumber,
          finalLessonUrl
        );
        
        return {
          to: user.email,
          subject: emailContent.subject,
          html: emailContent.html,
          text: emailContent.text,
          tags: [EMAIL_TAGS.DAILY_LESSON, { name: 'day', value: String(dayNumber) }],
        };
      });

      const result = await sendBatchEmails(emails);
      
      if (!result.success) {
        return res.status(500).json({ 
          error: 'Failed to send batch emails',
          details: result.details 
        });
      }

      return res.status(200).json({ 
        success: true,
        message: `Sent ${body.batch.length} daily lesson emails`,
        data: result.data,
      });
    }

    // Handle single email
    if (!body.email) {
      return res.status(400).json({ 
        error: 'Email is required (or provide batch array)' 
      });
    }

    const emailContent = dailyLessonEmail(
      body.name || 'friend',
      lessonTitle,
      lessonEmoji,
      lessonCategory,
      dayNumber,
      finalLessonUrl
    );

    const result = await sendEmail({
      to: body.email,
      subject: emailContent.subject,
      html: emailContent.html,
      text: emailContent.text,
      tags: [EMAIL_TAGS.DAILY_LESSON, { name: 'day', value: String(dayNumber) }],
    });

    if (!result.success) {
      return res.status(500).json({ 
        error: 'Failed to send email',
        details: result.details 
      });
    }

    return res.status(200).json({ 
      success: true,
      message: 'Daily lesson email sent',
      id: result.id,
    });

  } catch (error) {
    console.error('Error sending daily lesson email:', error);
    return res.status(500).json({ 
      error: 'Failed to send email',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

