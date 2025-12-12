export type LocaleCode = 'en-US' | 'es-ES' | 'pt-BR';

export type RouteKey =
  | 'home'
  | 'adults'
  | 'children'
  | 'teachers'
  | 'schools'
  | 'demo'
  | 'privacy'
  | 'cookies'
  | 'thank-you';

export interface NavItem {
  key: RouteKey;
  href: string;
  label: string;
}

export interface FaqItem {
  question: string;
  answer: string;
}

export interface FeatureItem {
  title: string;
  description: string;
  icon: string;
}

export interface TestimonialItem {
  quote: string;
  author: string;
  role?: string;
}

export interface PricingOption {
  title: string;
  description: string;
}

export interface FormFieldCopy {
  label: string;
  placeholder?: string;
  helpText?: string;
  errors: {
    required?: string;
    invalid?: string;
  };
}

export interface LeadFormCopy {
  title: string;
  subtitle: string;
  submitLabel: string;
  submittingLabel: string;
  successHeading: string;
  successBody: string;
  successCta: string;
  errors: {
    generic: string;
    turnstile: string;
  };
  fields: {
    firstName: FormFieldCopy;
    lastName: FormFieldCopy;
    email: FormFieldCopy;
    phone: FormFieldCopy;
    country: FormFieldCopy;
    region: FormFieldCopy;
    marketingOptIn: {
      label: string;
      description: string;
    };
  };
}

export interface ConsentCopy {
  title: string;
  description: string;
  acceptAll: string;
  rejectAll: string;
  manageLabel: string;
  modalTitle: string;
  save: string;
  categories: {
    strictlyNecessary: {
      label: string;
      description: string;
    };
    analytics: {
      label: string;
      description: string;
    };
    marketing: {
      label: string;
      description: string;
    };
  };
}

export interface CountdownCopy {
  labelActive: string;
  labelEnded: string;
  offerEndedCta: string;
  units: {
    days: string;
    hours: string;
    minutes: string;
    seconds: string;
  };
}

export interface LocalePromptCopy {
  message: string;
  confirm: string;
  dismiss: string;
}

export interface ThankYouCopy {
  heading: string;
  body: string;
  checklist: string[];
  back: string;
}

export interface LocaleDictionary {
  code: LocaleCode;
  languageName: string;
  meta: {
    title: string;
    description: string;
    keywords: string[];
  };
  hero: {
    headline: string;
    subheadline: string;
    ctaLabel: string;
  };
  countdown: CountdownCopy;
  nav: NavItem[];
  leadForm: LeadFormCopy;
  testimonials: {
    title: string;
    items: TestimonialItem[];
  };
  features: {
    title: string;
    items: FeatureItem[];
  };
  pricing: {
    title: string;
    subtitle: string;
    options: PricingOption[];
    legal: string;
  };
  faq: {
    title: string;
    items: FaqItem[];
  };
  trust: {
    title: string;
    items: string[];
  };
  footer: {
    rights: string;
    privacy: string;
    cookies: string;
    storeHeading: string;
  };
  consent: ConsentCopy;
  analytics: {
    viewEvent: string;
    leadSubmittedEvent: string;
    leadErrorEvent: string;
    consentChangedEvent: string;
    localePromptShown: string;
    localePromptAccepted: string;
    localePromptDismissed: string;
    unityDemoEvent: string;
  };
  localePrompt: LocalePromptCopy;
  thankYou: ThankYouCopy;
  demoPage: {
    title: string;
    subtitle: string;
    actions: {
      load: string;
      stop: string;
      reload: string;
    };
    checklistTitle: string;
    checklist: string[];
    status: {
      waiting: string;
      ready: string;
      loading: string;
      playing: string;
      error: string;
      assetsMissing: string;
    };
    supportCta: string;
    fallback: {
      title: string;
      body: string;
      ctaLabel: string;
    };
  };
}





