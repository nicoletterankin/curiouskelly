import type { LocaleDictionary } from './types';

export const enUS: LocaleDictionary = {
  code: 'en-US',
  languageName: 'English (US)',
  meta: {
    title: 'The Daily Lesson by Curious Kelly — Learn something new every day',
    description:
      'Join thousands of curious learners with 8-minute daily lessons for adults, children, and teachers. Age-adaptive. Three languages. Just $4.99/month.',
    keywords: [
      'daily learning',
      'AI learning companion',
      'Kelly',
      'online education',
      'learn every day',
      'curiosity'
    ]
  },
  hero: {
    headline: 'Learn something new every day with Kelly',
    subheadline:
      '8-minute daily lessons for adults, children, and teachers. Age-adaptive. Three languages. One universal topic.',
    ctaLabel: 'Start your 7-day free trial'
  },
  countdown: {
    labelActive: 'Holiday gift special ends in',
    labelEnded: 'Give 365 days of curiosity for 2026',
    offerEndedCta: 'Buy gift subscription',
    units: {
      days: 'Days',
      hours: 'Hours',
      minutes: 'Minutes',
      seconds: 'Seconds'
    }
  },
  nav: [
    { key: 'home', href: '/', label: 'Home' },
    { key: 'adults', href: '/adults/', label: 'Adults' },
    { key: 'children', href: '/children/', label: 'Children' },
    { key: 'teachers', href: '/teachers/', label: 'Teachers' },
    { key: 'privacy', href: '/privacy/', label: 'Privacy' },
    { key: 'cookies', href: '/cookies/', label: 'Cookies' }
  ],
  leadForm: {
    title: 'Start your 7-day free trial',
    subtitle:
      'No credit card required. Cancel anytime.',
    submitLabel: 'Start learning free',
    submittingLabel: 'Starting your trial…',
    successHeading: 'Welcome! You're all set.',
    successBody:
      'Check your email to start your first lesson. We're excited to have you!',
    successCta: 'Back to home',
    errors: {
      generic: 'Something went wrong. Please try again or email support@curiouskelly.com',
      turnstile: 'Please complete the verification to continue.'
    },
    fields: {
      firstName: {
        label: 'Name',
        placeholder: 'Kelly',
        errors: {
          required: 'Name is required.',
          invalid: 'Please enter a valid name.'
        }
      },
      lastName: {
        label: 'Last name',
        placeholder: 'Rivera',
        errors: {
          required: 'Last name is required.',
          invalid: 'Please enter a valid last name.'
        }
      },
      email: {
        label: 'Email',
        placeholder: 'you@example.com',
        errors: {
          required: 'Email is required.',
          invalid: 'Please enter a valid email address.'
        }
      },
      phone: {
        label: 'Mobile number (optional)',
        placeholder: '+1 555 010 2026',
        helpText: 'For lesson reminders via text',
        errors: {
          required: 'Mobile number is required.',
          invalid: 'Please enter a valid phone number.'
        }
      },
      country: {
        label: 'Country',
        placeholder: 'Select your country',
        errors: {
          required: 'Please select a country.'
        }
      },
      region: {
        label: 'State / Province',
        placeholder: 'Select your region',
        errors: {
          required: 'Please select a region.'
        }
      },
      marketingOptIn: {
        label: 'Keep me updated',
        description: 'Send me tips, new lessons, and special offers (optional)'
      }
    }
  },
  testimonials: {
    title: 'Join thousands of daily learners',
    items: [
      { quote: 'I look forward to my daily lesson with Kelly every morning. It's like coffee for my brain.', author: 'Sarah M.', role: 'Adult learner' },
      { quote: 'My 7-year-old asks for her "Kelly time" every day after school.', author: 'Marcus T.', role: 'Parent' },
      { quote: 'Perfect conversation starter for my students. Same topic, every level can engage.', author: 'Jamie R.', role: 'Middle school teacher' }
    ]
  },
  features: {
    title: 'Why people choose The Daily Lesson',
    items: [
      {
        title: '8 minutes a day',
        description: 'Fits any schedule. Builds lasting habits. No overwhelm.',
        icon: 'clock'
      },
      {
        title: 'Privacy-first',
        description: 'No ads. No data selling. No tracking. Your learning is yours alone.',
        icon: 'shield'
      },
      {
        title: 'Three languages',
        description: 'English, Spanish, and Portuguese. Every lesson. Every day.',
        icon: 'globe'
      },
      {
        title: 'Whole family',
        description: 'One subscription. Up to 5 profiles. Everyone learns together.',
        icon: 'people'
      }
    ]
  },
  pricing: {
    title: 'Simple, honest pricing',
    subtitle: 'One subscription. Unlimited lessons. Three languages.',
    options: [
      {
        title: 'Monthly',
        description: '$4.99/month · Cancel anytime · Try 7 days free'
      },
      {
        title: 'Annual',
        description: '$49.99/year · Save $10 · Perfect for gifting'
      }
    ],
    legal: '7-day free trial. No credit card required to start. Cancel anytime.'
  },
  faq: {
    title: 'Common questions',
    items: [
      {
        question: 'How does the free trial work?',
        answer:
          '7 days free, no credit card required. Try unlimited lessons in all three languages. If you love it, choose monthly ($4.99) or annual ($49.99). Cancel anytime with one click.'
      },
      {
        question: 'Can my whole family use one account?',
        answer:
          'Yes! Create up to 5 profiles per subscription. Each person gets age-adaptive lessons and tracks their own progress.'
      },
      {
        question: 'What languages do you support?',
        answer:
          'Every lesson is available in English, Spanish, and Brazilian Portuguese. Switch languages anytime.'
      },
      {
        question: 'Is it really for ages 2 to 102?',
        answer:
          'Yes! The same universal topic adapts to your age. A 6-year-old and a 60-year-old can discuss the same lesson at dinner.'
      },
      {
        question: 'Can I gift this for Christmas?',
        answer:
          'Absolutely! Annual subscriptions ($49.99) make perfect gifts. We provide email delivery and a personalized message option for last-minute gifting.'
      },
      {
        question: 'Is my data safe?',
        answer:
          'We never sell your data, show ads, or track you across the web. Privacy-first means privacy always. Your learning stays private.'
      }
    ]
  },
  trust: {
    title: 'Trusted by curious learners in 47 countries',
    items: [
      'Ad-free learning',
      'Privacy guaranteed',
      '365 universal lessons',
      'Three languages included'
    ]
  },
  footer: {
    rights: '© 2025 The Daily Lesson by Curious Kelly. All rights reserved.',
    privacy: 'Privacy',
    cookies: 'Cookies',
    storeHeading: 'Download our app (coming soon)'
  },
  consent: {
    title: 'Manage your privacy',
    description:
      'We use minimal analytics to improve your experience. Marketing cookies only load if you opt in.',
    acceptAll: 'Accept all',
    rejectAll: 'Reject optional cookies',
    manageLabel: 'Manage preferences',
    modalTitle: 'Cookie preferences',
    save: 'Save my preferences',
    categories: {
      strictlyNecessary: {
        label: 'Essential',
        description: 'Required for security and basic site functionality.'
      },
      analytics: {
        label: 'Analytics',
        description: 'Helps us understand what works and improve lessons.'
      },
      marketing: {
        label: 'Marketing',
        description: 'Optional tracking pixels (Google, Meta, etc.)'
      }
    }
  },
  analytics: {
    viewEvent: 'page_view',
    leadSubmittedEvent: 'trial_started',
    leadErrorEvent: 'trial_error',
    consentChangedEvent: 'consent_changed',
    localePromptShown: 'language_prompt_shown',
    localePromptAccepted: 'language_switched',
    localePromptDismissed: 'language_prompt_dismissed'
  },
  localePrompt: {
    message: 'Would you like to switch to {{language}}?',
    confirm: 'Switch language',
    dismiss: 'Stay in English'
  },
  thankYou: {
    heading: 'Welcome to The Daily Lesson!',
    body: 'Check your email for your first lesson. We're excited to learn with you.',
    checklist: [
      'Start your first 8-minute lesson',
      'Create profiles for your family',
      'Explore lessons in English, Spanish, or Portuguese'
    ],
    back: 'Back to homepage'
  }
};

