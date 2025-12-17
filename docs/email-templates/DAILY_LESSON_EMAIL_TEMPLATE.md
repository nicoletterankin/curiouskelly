# Daily Lesson Email Template

**Purpose:** Full lesson delivery via email for email-first users  
**Format:** HTML email (dark mode, mobile responsive)  
**Use:** Replace template variables with lesson content from PhaseDNA

---

## Template Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{DAY_NUMBER}}` | Day of year (1-365) | `42` |
| `{{DATE_FORMATTED}}` | Human-readable date | `February 11, 2025` |
| `{{FIRST_NAME}}` | Learner's first name | `Alex` |
| `{{LEARN_TOPIC}}` | Today's LEARN lesson title | `Why the Sky is Blue` |
| `{{LEARN_IMAGE_URL}}` | LEARN lesson hero image | `https://cdn...` |
| `{{LEARN_CONTENT_PARAGRAPH_1}}` | First paragraph of LEARN | Full text |
| `{{LEARN_CONTENT_PARAGRAPH_2}}` | Second paragraph of LEARN | Full text |
| `{{LEARN_QUESTION}}` | Question from the lesson | `What causes...` |
| `{{LEARN_ANSWER}}` | Answer to the question | `Light scatters...` |
| `{{GROW_TOPIC}}` | Today's GROW skill title | `Active Listening` |
| `{{GROW_IMAGE_URL}}` | GROW lesson image | `https://cdn...` |
| `{{GROW_CONTENT_PARAGRAPH_1}}` | First paragraph of GROW | Full text |
| `{{GROW_CONTENT_PARAGRAPH_2}}` | Second paragraph of GROW | Full text |
| `{{GROW_ACTIVITY}}` | Practice activity | `Try this today...` |
| `{{WISDOM_QUOTE}}` | Daily wisdom quote | `"Stay curious..."` |
| `{{STREAK_COUNT}}` | Current streak (optional) | `7` |
| `{{UNSUBSCRIBE_URL}}` | Unsubscribe link | `https://...` |

---

## HTML Template

Copy the code below and replace variables with actual content:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <title>✨ Curious Kelly — Day {{DAY_NUMBER}}</title>
  <!--[if mso]>
  <noscript>
    <xml>
      <o:OfficeDocumentSettings>
        <o:PixelsPerInch>96</o:PixelsPerInch>
      </o:OfficeDocumentSettings>
    </xml>
  </noscript>
  <![endif]-->
  <style>
    /* Reset */
    body, table, td, p, a, li, blockquote {
      -webkit-text-size-adjust: 100%;
      -ms-text-size-adjust: 100%;
    }
    table, td { mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
    img { -ms-interpolation-mode: bicubic; border: 0; height: auto; line-height: 100%; outline: none; text-decoration: none; }
    
    /* Base */
    body {
      margin: 0 !important;
      padding: 0 !important;
      background-color: #0a0a0b;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      .email-bg { background-color: #0a0a0b !important; }
      .email-card { background-color: #18181b !important; }
      .email-text { color: #ffffff !important; }
      .email-text-secondary { color: #a1a1aa !important; }
    }
    
    /* Responsive */
    @media only screen and (max-width: 600px) {
      .email-container { width: 100% !important; padding: 16px !important; }
      .email-card { padding: 20px !important; }
      .hero-image { width: 100% !important; height: auto !important; }
      .cta-button { width: 100% !important; display: block !important; }
    }
  </style>
</head>
<body class="email-bg" style="margin: 0; padding: 0; background-color: #0a0a0b;">
  
  <!-- Preheader (hidden preview text) -->
  <div style="display: none; max-height: 0; overflow: hidden; mso-hide: all;">
    Today's LEARN: {{LEARN_TOPIC}} • Today's GROW: {{GROW_TOPIC}} ✨
  </div>
  
  <!-- Email Container -->
  <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background-color: #0a0a0b;">
    <tr>
      <td align="center" style="padding: 24px 16px;">
        <table role="presentation" cellpadding="0" cellspacing="0" width="600" class="email-container" style="max-width: 600px; width: 100%;">
          
          <!-- Header -->
          <tr>
            <td align="center" style="padding-bottom: 24px;">
              <table role="presentation" cellpadding="0" cellspacing="0">
                <tr>
                  <td style="font-size: 28px; padding-right: 8px;">✨</td>
                  <td style="color: #ffffff; font-size: 20px; font-weight: 700; letter-spacing: 0.5px;">
                    Curious Kelly
                  </td>
                </tr>
              </table>
              <p style="color: #71717a; font-size: 14px; margin: 8px 0 0;">
                {{DATE_FORMATTED}} • Day {{DAY_NUMBER}} of 365
              </p>
            </td>
          </tr>
          
          <!-- Greeting -->
          <tr>
            <td style="color: #ffffff; font-size: 16px; padding-bottom: 24px;">
              Hey {{FIRST_NAME}}! 👋
            </td>
          </tr>
          
          <!-- ═══════════════════════════════════════════════════
               LEARN LESSON SECTION
               ═══════════════════════════════════════════════════ -->
          <tr>
            <td class="email-card" style="background-color: #18181b; border-radius: 16px; padding: 24px; margin-bottom: 24px;">
              
              <!-- Section Header -->
              <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
                <tr>
                  <td style="padding-bottom: 16px;">
                    <span style="background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                      📚 Learn
                    </span>
                  </td>
                </tr>
              </table>
              
              <!-- Topic Title -->
              <h2 style="color: #ffffff; font-size: 22px; font-weight: 700; margin: 0 0 16px; line-height: 1.3;">
                {{LEARN_TOPIC}}
              </h2>
              
              <!-- Hero Image -->
              <img src="{{LEARN_IMAGE_URL}}" alt="{{LEARN_TOPIC}}" class="hero-image" 
                   style="width: 100%; max-width: 552px; height: auto; border-radius: 12px; margin-bottom: 16px;">
              
              <!-- Lesson Content -->
              <div style="color: #e4e4e7; font-size: 16px; line-height: 1.7; margin-bottom: 20px;">
                {{LEARN_CONTENT_PARAGRAPH_1}}
              </div>
              
              <div style="color: #e4e4e7; font-size: 16px; line-height: 1.7; margin-bottom: 20px;">
                {{LEARN_CONTENT_PARAGRAPH_2}}
              </div>
              
              <!-- Question Box -->
              <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
                <tr>
                  <td style="background-color: rgba(59, 130, 246, 0.15); border-left: 4px solid #3b82f6; padding: 16px; border-radius: 8px;">
                    <p style="color: #93c5fd; font-size: 14px; font-weight: 600; margin: 0 0 8px;">
                      🤔 Quick Question
                    </p>
                    <p style="color: #e4e4e7; font-size: 15px; margin: 0; line-height: 1.5;">
                      {{LEARN_QUESTION}}
                    </p>
                  </td>
                </tr>
              </table>
              
              <!-- Answer Reveal -->
              <div style="margin-top: 16px; padding: 16px; background-color: rgba(255,255,255,0.05); border-radius: 8px;">
                <p style="color: #a1a1aa; font-size: 13px; margin: 0 0 8px; font-weight: 600;">
                  💡 Answer
                </p>
                <p style="color: #e4e4e7; font-size: 15px; margin: 0; line-height: 1.5;">
                  {{LEARN_ANSWER}}
                </p>
              </div>
              
            </td>
          </tr>
          
          <!-- Spacer -->
          <tr><td style="height: 24px;"></td></tr>
          
          <!-- ═══════════════════════════════════════════════════
               GROW LESSON SECTION
               ═══════════════════════════════════════════════════ -->
          <tr>
            <td class="email-card" style="background-color: #18181b; border-radius: 16px; padding: 24px; margin-bottom: 24px;">
              
              <!-- Section Header -->
              <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
                <tr>
                  <td style="padding-bottom: 16px;">
                    <span style="background: linear-gradient(135deg, #22c55e, #10b981); color: white; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                      🧠 Grow
                    </span>
                  </td>
                </tr>
              </table>
              
              <!-- Skill Title -->
              <h2 style="color: #ffffff; font-size: 22px; font-weight: 700; margin: 0 0 16px; line-height: 1.3;">
                {{GROW_TOPIC}}
              </h2>
              
              <!-- Skill Image -->
              <img src="{{GROW_IMAGE_URL}}" alt="{{GROW_TOPIC}}" class="hero-image" 
                   style="width: 100%; max-width: 552px; height: auto; border-radius: 12px; margin-bottom: 16px;">
              
              <!-- Skill Content -->
              <div style="color: #e4e4e7; font-size: 16px; line-height: 1.7; margin-bottom: 20px;">
                {{GROW_CONTENT_PARAGRAPH_1}}
              </div>
              
              <div style="color: #e4e4e7; font-size: 16px; line-height: 1.7; margin-bottom: 20px;">
                {{GROW_CONTENT_PARAGRAPH_2}}
              </div>
              
              <!-- Try This Activity -->
              <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
                <tr>
                  <td style="background-color: rgba(34, 197, 94, 0.15); border-left: 4px solid #22c55e; padding: 16px; border-radius: 8px;">
                    <p style="color: #86efac; font-size: 14px; font-weight: 600; margin: 0 0 8px;">
                      🎯 Try This
                    </p>
                    <p style="color: #e4e4e7; font-size: 15px; margin: 0; line-height: 1.5;">
                      {{GROW_ACTIVITY}}
                    </p>
                  </td>
                </tr>
              </table>
              
            </td>
          </tr>
          
          <!-- Spacer -->
          <tr><td style="height: 24px;"></td></tr>
          
          <!-- ═══════════════════════════════════════════════════
               DAILY WISDOM
               ═══════════════════════════════════════════════════ -->
          <tr>
            <td align="center" style="padding: 24px;">
              <table role="presentation" cellpadding="0" cellspacing="0">
                <tr>
                  <td style="font-size: 24px; padding-bottom: 12px;">✨</td>
                </tr>
                <tr>
                  <td style="color: #a1a1aa; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; padding-bottom: 8px;">
                    Daily Wisdom
                  </td>
                </tr>
                <tr>
                  <td style="color: #ffffff; font-size: 18px; font-style: italic; line-height: 1.5; text-align: center; max-width: 450px;">
                    "{{WISDOM_QUOTE}}"
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Spacer -->
          <tr><td style="height: 24px;"></td></tr>
          
          <!-- ═══════════════════════════════════════════════════
               CTA - Experience with Kelly
               ═══════════════════════════════════════════════════ -->
          <tr>
            <td align="center" style="padding: 24px 0;">
              <table role="presentation" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center" style="background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 12px;">
                    <a href="https://curiouskelly.com/learn.html?day={{DAY_NUMBER}}" 
                       class="cta-button"
                       style="display: inline-block; padding: 16px 32px; color: #ffffff; font-size: 16px; font-weight: 600; text-decoration: none;">
                      ▶ Experience with Kelly
                    </a>
                  </td>
                </tr>
              </table>
              <p style="color: #71717a; font-size: 13px; margin: 12px 0 0;">
                Watch today's lesson with voice & video
              </p>
            </td>
          </tr>
          
          <!-- Streak Reminder (conditional) -->
          <!-- {{#if STREAK_COUNT}} -->
          <tr>
            <td align="center" style="padding: 16px; background-color: rgba(251, 191, 36, 0.1); border-radius: 12px;">
              <p style="color: #fbbf24; font-size: 14px; margin: 0;">
                🔥 {{STREAK_COUNT}} day streak! Keep it going!
              </p>
            </td>
          </tr>
          <!-- {{/if}} -->
          
          <!-- Spacer -->
          <tr><td style="height: 32px;"></td></tr>
          
          <!-- Footer -->
          <tr>
            <td align="center" style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 24px;">
              <p style="color: #71717a; font-size: 13px; margin: 0 0 8px;">
                Stay curious! ✨
              </p>
              <p style="color: #52525b; font-size: 12px; margin: 0 0 16px;">
                — Kelly
              </p>
              
              <p style="color: #52525b; font-size: 11px; margin: 0;">
                <a href="{{UNSUBSCRIBE_URL}}" style="color: #52525b; text-decoration: underline;">Unsubscribe</a>
                &nbsp;•&nbsp;
                <a href="https://curiouskelly.com/learn.html?tab=settings" style="color: #52525b; text-decoration: underline;">Preferences</a>
                &nbsp;•&nbsp;
                <a href="https://curiouskelly.com" style="color: #52525b; text-decoration: underline;">curiouskelly.com</a>
              </p>
              
              <p style="color: #3f3f46; font-size: 10px; margin: 16px 0 0;">
                Lesson of the Day, PBC • hello@curiouskelly.com
              </p>
            </td>
          </tr>
          
        </table>
      </td>
    </tr>
  </table>
  
</body>
</html>
```

---

## Email Client Compatibility

| Client | Status | Notes |
|--------|--------|-------|
| Gmail (web) | ✅ | Full support |
| Gmail (mobile) | ✅ | Full support |
| Apple Mail | ✅ | Full support + dark mode |
| Outlook 365 | ✅ | MSO conditionals included |
| Outlook desktop | ⚠️ | Gradients may not render |
| Yahoo Mail | ✅ | Full support |

---

## Usage Notes

1. **Image hosting:** All images must be hosted on CDN with HTTPS
2. **Dark mode:** Template auto-adapts to system preference
3. **Mobile:** Fully responsive, stacks properly on mobile
4. **Unsubscribe:** Required by CAN-SPAM — always include
5. **Preheader:** Shows in email preview — keep concise

---

*Stay curious. ✨*
