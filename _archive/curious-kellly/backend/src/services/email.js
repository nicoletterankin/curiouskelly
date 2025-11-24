const sgMail = require('@sendgrid/mail');

sgMail.setApiKey(process.env.SENDGRID_API_KEY);

const EMAIL_TEMPLATES = {
  WAITLIST: process.env.TEMPLATE_WAITLIST,
  EARLY_BIRD: process.env.TEMPLATE_EARLY_BIRD,
  LAST_CHANCE: process.env.TEMPLATE_LAST_CHANCE,
  GIFT_RECIPIENT: process.env.TEMPLATE_GIFT_RECIPIENT,
  GIFTER_CONFIRM: process.env.TEMPLATE_GIFTER_CONFIRM,
  CALENDAR_EXPLORE: process.env.TEMPLATE_CALENDAR_EXPLORE,
  GET_READY: process.env.TEMPLATE_GET_READY,
  DAY1: process.env.TEMPLATE_DAY1,
  WELCOME: process.env.TEMPLATE_WELCOME,
  DAILY_REMINDER: process.env.TEMPLATE_DAILY_REMINDER,
  STREAK: process.env.TEMPLATE_STREAK,
  WEEK1: process.env.TEMPLATE_WEEK1,
  MISSED: process.env.TEMPLATE_MISSED,
  REENGAGE: process.env.TEMPLATE_REENGAGE
};

/**
 * Send email using SendGrid dynamic template
 * @param {Object} options - Email options
 * @param {string} options.to - Recipient email
 * @param {string} options.templateId - SendGrid template ID
 * @param {Object} options.dynamicTemplateData - Data for template
 * @param {Date} options.sendAt - Optional: Schedule email for future send
 */
async function sendEmail({ to, templateId, dynamicTemplateData, sendAt }) {
  try {
    const msg = {
      to,
      from: {
        email: process.env.FROM_EMAIL,
        name: process.env.FROM_NAME
      },
      templateId,
      dynamicTemplateData,
      ...(sendAt && { sendAt: Math.floor(sendAt.getTime() / 1000) })
    };

    const result = await sgMail.send(msg);
    console.log('✓ Email sent:', { to, templateId, scheduled: !!sendAt });
    return result;
  } catch (error) {
    console.error('✗ Email error:', error.response?.body || error.message);
    throw error;
  }
}

/**
 * Send gift recipient email (Christmas morning delivery)
 */
async function sendGiftRecipientEmail({ recipientEmail, recipientName, gifterName, giftMessage, giftCode, calendarUrl }) {
  return sendEmail({
    to: recipientEmail,
    templateId: EMAIL_TEMPLATES.GIFT_RECIPIENT,
    dynamicTemplateData: {
      recipient_name: recipientName || extractFirstName(recipientEmail),
      gifter_name: gifterName,
      gift_message: giftMessage || '',
      gift_code: giftCode,
      calendar_url: calendarUrl || `${process.env.FRONTEND_URL}/calendar`,
      redeem_url: `${process.env.FRONTEND_URL}/redeem?code=${giftCode}`
    },
    sendAt: new Date('2025-12-25T06:00:00Z') // Christmas morning, 6am UTC
  });
}

/**
 * Send immediate confirmation to gift purchaser
 */
async function sendGifterConfirmationEmail({ gifterEmail, gifterName, recipientEmail, orderNumber, amount }) {
  return sendEmail({
    to: gifterEmail,
    templateId: EMAIL_TEMPLATES.GIFTER_CONFIRM,
    dynamicTemplateData: {
      gifter_name: gifterName || extractFirstName(gifterEmail),
      recipient_email: recipientEmail,
      order_number: orderNumber,
      amount: amount
    }
  });
}

/**
 * Send Day 1 lesson notification (January 1st)
 */
async function sendDay1LessonEmail({ userEmail, userName, lessonTitle }) {
  return sendEmail({
    to: userEmail,
    templateId: EMAIL_TEMPLATES.DAY1,
    dynamicTemplateData: {
      recipient_name: userName || extractFirstName(userEmail),
      lesson_title: lessonTitle || 'The Sun - Our Magnificent Life-Giving Star',
      lesson_day: 1,
      lesson_url: `${process.env.FRONTEND_URL}/player?day=1`
    },
    sendAt: new Date('2026-01-01T06:00:00Z') // Jan 1, 6am UTC
  });
}

/**
 * Send daily reminder email
 */
async function sendDailyReminderEmail({ userEmail, userName, lessonDay, lessonTitle, currentStreak, lessonsCompleted }) {
  return sendEmail({
    to: userEmail,
    templateId: EMAIL_TEMPLATES.DAILY_REMINDER,
    dynamicTemplateData: {
      recipient_name: userName || extractFirstName(userEmail),
      lesson_day: lessonDay,
      lesson_title: lessonTitle,
      current_streak: currentStreak,
      lessons_completed: lessonsCompleted,
      total_lessons: 365,
      lesson_url: `${process.env.FRONTEND_URL}/player?day=${lessonDay}`
    }
  });
}

/**
 * Send streak milestone celebration email
 */
async function sendStreakMilestoneEmail({ userEmail, userName, streakDays, lessonsCompleted }) {
  return sendEmail({
    to: userEmail,
    templateId: EMAIL_TEMPLATES.STREAK,
    dynamicTemplateData: {
      recipient_name: userName || extractFirstName(userEmail),
      streak_days: streakDays,
      lessons_completed: lessonsCompleted,
      total_lessons: 365,
      progress_percentage: Math.round((lessonsCompleted / 365) * 100)
    }
  });
}

/**
 * Extract first name from email address
 */
function extractFirstName(email) {
  const name = email.split('@')[0];
  return name.charAt(0).toUpperCase() + name.slice(1);
}

module.exports = {
  sendEmail,
  sendGiftRecipientEmail,
  sendGifterConfirmationEmail,
  sendDay1LessonEmail,
  sendDailyReminderEmail,
  sendStreakMilestoneEmail,
  EMAIL_TEMPLATES
};






