import type { LocaleDictionary } from './types';

export const enUS: LocaleDictionary = {
  code: 'en-US',
  languageName: 'English (US)',
  meta: {
    title: 'Curious Kelly — The privacy-first AI learning companion',
    description:
      'Curious Kelly delivers daily micro-lessons that build joyful learning habits for adults, children, and teachers—now enrolling for 2026.',
    keywords: [
      'Curious Kelly',
      'AI tutor',
      'daily learning',
      'education technology',
      'personalized lessons'
    ]
  },
  hero: {
    headline: 'Learning with heart, for every learner.',
    subheadline:
      'Curious Kelly blends story, science, and human warmth so your family keeps learning every single day.',
    ctaLabel: 'Register for 2026 access'
  },
  countdown: {
    labelActive: 'Offer ends in',
    labelEnded: 'The 2026 cohort is now full.',
    offerEndedCta: 'Join the waitlist',
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
    { key: 'schools', href: '/schools/', label: 'Schools' },
    { key: 'demo', href: '/demo/avatar/', label: 'Live Avatar' },
    { key: 'privacy', href: '/privacy/', label: 'Privacy' },
    { key: 'cookies', href: '/cookies/', label: 'Cookies' }
  ],
  leadForm: {
    title: 'Tell us who should meet Curious Kelly first',
    subtitle:
      'Our concierge team will confirm your enrollment and schedule an onboarding session tailored to your goals.',
    submitLabel: 'Submit interest',
    submittingLabel: 'Submitting…',
    successHeading: 'Thank you! You’re on the list.',
    successBody:
      'We just pinged the concierge team. Expect a follow-up within one business day with your access kit.',
    successCta: 'Return to home',
    errors: {
      generic: 'We could not save your information. Please retry or contact hello@curiouskelly.com.',
      turnstile: 'Please complete the verification to prove you’re human.'
    },
    fields: {
      firstName: {
        label: 'First name',
        placeholder: 'Kelly',
        errors: {
          required: 'First name is required.',
          invalid: 'Only letters, spaces, hyphens, and apostrophes are allowed.'
        }
      },
      lastName: {
        label: 'Last name',
        placeholder: 'Rivera',
        errors: {
          required: 'Last name is required.',
          invalid: 'Only letters, spaces, hyphens, and apostrophes are allowed.'
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
        label: 'Mobile number',
        placeholder: '+1 555 010 2026',
        helpText: 'Include your country code for WhatsApp or SMS updates.',
        errors: {
          required: 'Mobile number is required.',
          invalid: 'Please enter a valid international phone number.'
        }
      },
      country: {
        label: 'Country / Region',
        placeholder: 'Select a country',
        errors: {
          required: 'Please select a country.'
        }
      },
      region: {
        label: 'State / Province',
        placeholder: 'Select a region',
        errors: {
          required: 'Please select a region for the chosen country.'
        }
      },
      marketingOptIn: {
        label: 'Stay in the loop',
        description: 'I agree to receive updates about Curious Kelly launch events and new features.'
      }
    }
  },
  testimonials: {
    title: 'Voices from our pilot schools',
    items: [
      { quote: 'Best digital teacher of the year — 2025.', author: 'EduTech Awards Jury' },
      { quote: 'Finally the learning has a face.', author: 'Aria T.', role: 'Parent of a 9-year-old' },
      { quote: 'Kelly sparks curiosity in our adult learners every morning.', author: 'Jamal R.', role: 'Community college dean' }
    ]
  },
  features: {
    title: 'Why families choose Curious Kelly',
    items: [
      {
        title: 'Daily micro-lessons',
        description: '8-minute experiences that fit any schedule and keep streaks alive.',
        icon: 'clock'
      },
      {
        title: 'Privacy-first design',
        description: 'No ads, no dark patterns—your household data stays encrypted and in your control.',
        icon: 'shield'
      },
      {
        title: 'Trilingual from day one',
        description: 'Every lesson is authored in English, Spanish, and Brazilian Portuguese before delivery.',
        icon: 'globe'
      },
      {
        title: 'Human concierge',
        description: 'A real person checks every onboarding plan to align with your goals.',
        icon: 'people'
      }
    ]
  },
  pricing: {
    title: 'Founding cohort benefits',
    subtitle: 'Reserve your 2026 seat now and lock in lifetime pricing.',
    options: [
      {
        title: 'Household access',
        description: 'Unlimited profiles across adults and children with daily streak coaching.'
      },
      {
        title: 'Teacher toolkit',
        description: 'Lesson exports, consent workflows, and ready-to-run group guides.'
      },
      {
        title: 'School partnerships',
        description: 'District-level analytics, privacy frameworks, and concierge onboarding.'
      }
    ],
    legal: 'No payment is collected today. We will confirm eligibility and pricing during onboarding.'
  },
  faq: {
    title: 'Frequently asked questions',
    items: [
      {
        question: 'How does Curious Kelly protect learner privacy?',
        answer:
          'We precompute all lesson content and never profile learners. Content is cached per locale and device, consent is required for every marketing tag, and we never sell data.'
      },
      {
        question: 'Will Kelly replace teachers?',
        answer:
          'No. Kelly is designed as a co-teacher who handles repetition and reminders so educators can focus on human connection.'
      },
      {
        question: 'Can we try Kelly before committing?',
        answer:
          'Yes. The concierge team will share a sample lesson plan and facilitate a live walkthrough tailored to your audience.'
      }
    ]
  },
  trust: {
    title: 'Trusted by teams who care about daily learning habits',
    items: [
      '2025 EdTech Rising Stars',
      'European Privacy Design Council',
      'Global Learning Collective',
      'Equity in AI Education Fund'
    ]
  },
  footer: {
    rights: '© 2025 Curious Kelly. All rights reserved.',
    privacy: 'Privacy',
    cookies: 'Cookies',
    storeHeading: 'Available soon'
  },
  consent: {
    title: 'Manage your privacy choices',
    description:
      'We use minimal, privacy-friendly analytics to improve the Curious Kelly experience. Marketing tags only load after you opt in.',
    acceptAll: 'Accept all',
    rejectAll: 'Reject non-essential',
    manageLabel: 'Manage cookies',
    modalTitle: 'Consent settings',
    save: 'Save preferences',
    categories: {
      strictlyNecessary: {
        label: 'Essential',
        description: 'Required for security, consent storage, and service delivery.'
      },
      analytics: {
        label: 'Analytics',
        description: 'Helps us understand performance and improve lessons.'
      },
      marketing: {
        label: 'Marketing',
        description: 'Allows optional pixels such as GTM, Meta, and TikTok.'
      }
    }
  },
  analytics: {
    viewEvent: 'page_view',
    leadSubmittedEvent: 'lead_submitted',
    leadErrorEvent: 'lead_error',
    consentChangedEvent: 'consent_changed',
    localePromptShown: 'locale_prompt_shown',
    localePromptAccepted: 'locale_prompt_accepted',
    localePromptDismissed: 'locale_prompt_dismissed',
    unityDemoEvent: 'unity_demo_event'
  },
  localePrompt: {
    message: 'Switch to {{language}} for a localized Curious Kelly experience?',
    confirm: 'Switch language',
    dismiss: 'Stay in English'
  },
  thankYou: {
    heading: 'You’re officially on Kelly’s radar.',
    body: 'We sent a confirmation email with next steps. A concierge will reach out within 24 hours.',
    checklist: [
      'Add hello@curiouskelly.com to your safe senders list.',
      'Invite a colleague to join the onboarding session.',
      'Prepare your top learning goals for 2026.'
    ],
    back: 'Back to homepage'
  },
  demoPage: {
    title: 'Live Avatar Preview',
    subtitle: 'Stream the Unity WebGL build directly in your browser and trigger a real lesson payload.',
    actions: {
      load: 'Play sample lesson',
      stop: 'Stop playback',
      reload: 'Reload player'
    },
    checklistTitle: 'What to expect',
    checklist: [
      'Unity boots inside the iframe in ~5 seconds (Brotli, hash-named files).',
      'Lesson audio and visemes stream from secure CDN endpoints.',
      'The postMessage bridge emits readiness, playback, and error events for analytics.'
    ],
    status: {
      waiting: 'Waiting for the Unity player…',
      ready: 'Kelly is ready. Load a lesson when you are.',
      loading: 'Loading lesson assets…',
      playing: 'Lesson is playing.',
      error: 'Unable to reach the player. Confirm the WebGL build is deployed.',
      assetsMissing: 'Set PUBLIC_UNITY_SAMPLE_JSON/AUDIO to enable playback.'
    },
    supportCta: 'View deployment playbook',
    fallback: {
      title: 'Need help?',
      body: 'If the player keeps failing after two attempts, ping the concierge team and attach the browser console log.',
      ctaLabel: 'Contact concierge'
    }
  }
};





