import type { LocaleDictionary } from './types';

export const esES: LocaleDictionary = {
  code: 'es-ES',
  languageName: 'Español (España)',
  meta: {
    title: 'The Daily Lesson by Curious Kelly — Aprende algo nuevo cada día',
    description:
      'Únete a miles de estudiantes curiosos con lecciones diarias de 8 minutos para adultos, niños y docentes. Adaptado por edad. Tres idiomas. Solo $4.99/mes.',
    keywords: [
      'aprendizaje diario',
      'compañera de aprendizaje IA',
      'Kelly',
      'educación en línea',
      'aprender cada día',
      'curiosidad'
    ]
  },
  hero: {
    headline: 'Aprende algo nuevo cada día con Kelly',
    subheadline:
      'Lecciones diarias de 8 minutos para adultos, niños y docentes. Adaptado por edad. Tres idiomas. Un tema universal.',
    ctaLabel: 'Comienza tu prueba gratuita de 7 días'
  },
  countdown: {
    labelActive: 'Oferta especial de vacaciones termina en',
    labelEnded: 'Regala 365 días de curiosidad para 2026',
    offerEndedCta: 'Comprar suscripción de regalo',
    units: {
      days: 'Días',
      hours: 'Horas',
      minutes: 'Minutos',
      seconds: 'Segundos'
    }
  },
  nav: [
    { key: 'home', href: '/es-es/', label: 'Inicio' },
    { key: 'adults', href: '/es-es/adults/', label: 'Adultos' },
    { key: 'children', href: '/es-es/children/', label: 'Niños' },
    { key: 'teachers', href: '/es-es/teachers/', label: 'Docentes' },
    { key: 'privacy', href: '/es-es/privacy/', label: 'Privacidad' },
    { key: 'cookies', href: '/es-es/cookies/', label: 'Cookies' }
  ],
  leadForm: {
    title: 'Comienza tu prueba gratuita de 7 días',
    subtitle:
      'Sin tarjeta de crédito. Cancela cuando quieras.',
    submitLabel: 'Empezar a aprender gratis',
    submittingLabel: 'Iniciando tu prueba…',
    successHeading: '¡Bienvenido! Todo listo.',
    successBody:
      'Revisa tu correo para comenzar tu primera lección. ¡Estamos emocionados de tenerte!',
    successCta: 'Volver al inicio',
    errors: {
      generic: 'Algo salió mal. Inténtalo de nuevo o escribe a support@curiouskelly.com',
      turnstile: 'Completa la verificación para continuar.'
    },
    fields: {
      firstName: {
        label: 'Nombre',
        placeholder: 'Kelly',
        errors: {
          required: 'El nombre es obligatorio.',
          invalid: 'Introduce un nombre válido.'
        }
      },
      lastName: {
        label: 'Apellidos',
        placeholder: 'Rivera',
        errors: {
          required: 'Los apellidos son obligatorios.',
          invalid: 'Introduce apellidos válidos.'
        }
      },
      email: {
        label: 'Correo electrónico',
        placeholder: 'tu@ejemplo.com',
        errors: {
          required: 'El correo es obligatorio.',
          invalid: 'Introduce una dirección válida.'
        }
      },
      phone: {
        label: 'Número de móvil (opcional)',
        placeholder: '+34 600 000 000',
        helpText: 'Para recordatorios de lecciones por mensaje',
        errors: {
          required: 'El móvil es obligatorio.',
          invalid: 'Introduce un número válido.'
        }
      },
      country: {
        label: 'País',
        placeholder: 'Selecciona tu país',
        errors: {
          required: 'Selecciona un país.'
        }
      },
      region: {
        label: 'Provincia / Estado',
        placeholder: 'Selecciona tu región',
        errors: {
          required: 'Selecciona una región.'
        }
      },
      marketingOptIn: {
        label: 'Mantenerme informado',
        description: 'Envíame consejos, nuevas lecciones y ofertas especiales (opcional)'
      }
    }
  },
  testimonials: {
    title: 'Únete a miles de estudiantes diarios',
    items: [
      { quote: 'Espero mi lección diaria con Kelly cada mañana. Es como café para mi cerebro.', author: 'Sarah M.', role: 'Estudiante adulta' },
      { quote: 'Mi hijo de 7 años pide su "tiempo con Kelly" todos los días después de la escuela.', author: 'Marcus T.', role: 'Padre' },
      { quote: 'Perfecto iniciador de conversación para mis estudiantes. Mismo tema, todos los niveles pueden participar.', author: 'Jamie R.', role: 'Docente de secundaria' }
    ]
  },
  features: {
    title: 'Por qué la gente elige The Daily Lesson',
    items: [
      {
        title: '8 minutos al día',
        description: 'Se adapta a cualquier horario. Crea hábitos duraderos. Sin abrumar.',
        icon: 'clock'
      },
      {
        title: 'Privacidad primero',
        description: 'Sin anuncios. Sin venta de datos. Sin rastreo. Tu aprendizaje es solo tuyo.',
        icon: 'shield'
      },
      {
        title: 'Tres idiomas',
        description: 'Inglés, español y portugués. Cada lección. Cada día.',
        icon: 'globe'
      },
      {
        title: 'Toda la familia',
        description: 'Una suscripción. Hasta 5 perfiles. Todos aprenden juntos.',
        icon: 'people'
      }
    ]
  },
  pricing: {
    title: 'Precios simples y honestos',
    subtitle: 'Una suscripción. Lecciones ilimitadas. Tres idiomas.',
    options: [
      {
        title: 'Mensual',
        description: '$4.99/mes · Cancela cuando quieras · Prueba 7 días gratis'
      },
      {
        title: 'Anual',
        description: '$49.99/año · Ahorra $10 · Perfecto para regalar'
      }
    ],
    legal: 'Prueba gratuita de 7 días. Sin tarjeta de crédito para empezar. Cancela cuando quieras.'
  },
  faq: {
    title: 'Preguntas frecuentes',
    items: [
      {
        question: '¿Cómo funciona la prueba gratuita?',
        answer:
          '7 días gratis, sin tarjeta de crédito. Prueba lecciones ilimitadas en los tres idiomas. Si te encanta, elige mensual ($4.99) o anual ($49.99). Cancela cuando quieras con un clic.'
      },
      {
        question: '¿Puede toda mi familia usar una cuenta?',
        answer:
          '¡Sí! Crea hasta 5 perfiles por suscripción. Cada persona obtiene lecciones adaptadas a su edad y rastrea su propio progreso.'
      },
      {
        question: '¿Qué idiomas admiten?',
        answer:
          'Cada lección está disponible en inglés, español y portugués brasileño. Cambia de idioma cuando quieras.'
      },
      {
        question: '¿Realmente es para edades de 2 a 102?',
        answer:
          '¡Sí! El mismo tema universal se adapta a tu edad. Un niño de 6 años y uno de 60 pueden discutir la misma lección en la cena.'
      },
      {
        question: '¿Puedo regalar esto para Navidad?',
        answer:
          '¡Por supuesto! Las suscripciones anuales ($49.99) son regalos perfectos. Ofrecemos entrega por correo y opción de mensaje personalizado para regalos de último minuto.'
      },
      {
        question: '¿Están seguros mis datos?',
        answer:
          'Nunca vendemos tus datos, mostramos anuncios o te rastreamos por la web. Privacidad primero significa privacidad siempre. Tu aprendizaje permanece privado.'
      }
    ]
  },
  trust: {
    title: 'Confiado por estudiantes curiosos en 47 países',
    items: [
      'Aprendizaje sin anuncios',
      'Privacidad garantizada',
      '365 lecciones universales',
      'Tres idiomas incluidos'
    ]
  },
  footer: {
    rights: '© 2025 The Daily Lesson by Curious Kelly. Todos los derechos reservados.',
    privacy: 'Privacidad',
    cookies: 'Cookies',
    storeHeading: 'Descarga nuestra app (próximamente)'
  },
  consent: {
    title: 'Gestiona tu privacidad',
    description:
      'Usamos análisis mínimos para mejorar tu experiencia. Las cookies de marketing solo se cargan si aceptas.',
    acceptAll: 'Aceptar todo',
    rejectAll: 'Rechazar cookies opcionales',
    manageLabel: 'Gestionar preferencias',
    modalTitle: 'Preferencias de cookies',
    save: 'Guardar mis preferencias',
    categories: {
      strictlyNecessary: {
        label: 'Esenciales',
        description: 'Necesarias para seguridad y funcionalidad básica del sitio.'
      },
      analytics: {
        label: 'Análisis',
        description: 'Nos ayuda a entender qué funciona y mejorar las lecciones.'
      },
      marketing: {
        label: 'Marketing',
        description: 'Píxeles de seguimiento opcionales (Google, Meta, etc.)'
      }
    }
  },
  analytics: {
    viewEvent: 'vista_pagina',
    leadSubmittedEvent: 'prueba_iniciada',
    leadErrorEvent: 'error_prueba',
    consentChangedEvent: 'consentimiento_cambiado',
    localePromptShown: 'aviso_idioma_mostrado',
    localePromptAccepted: 'idioma_cambiado',
    localePromptDismissed: 'aviso_idioma_descartado'
  },
  localePrompt: {
    message: '¿Te gustaría cambiar a {{language}}?',
    confirm: 'Cambiar idioma',
    dismiss: 'Seguir en español'
  },
  thankYou: {
    heading: '¡Bienvenido a The Daily Lesson!',
    body: 'Revisa tu correo para tu primera lección. Estamos emocionados de aprender contigo.',
    checklist: [
      'Comienza tu primera lección de 8 minutos',
      'Crea perfiles para tu familia',
      'Explora lecciones en inglés, español o portugués'
    ],
    back: 'Volver al inicio'
  }
};





