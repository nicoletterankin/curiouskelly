import type { LocaleDictionary } from './types';

export const esES: LocaleDictionary = {
  code: 'es-ES',
  languageName: 'Español (España)',
  meta: {
    title: 'Curious Kelly — La compañera de aprendizaje con privacidad primero',
    description:
      'Curious Kelly ofrece microlecciones diarias que construyen hábitos de aprendizaje alegres para adultos, niños y docentes. Inscripciones para 2026.',
    keywords: [
      'Curious Kelly',
      'tutora IA',
      'aprendizaje diario',
      'tecnología educativa',
      'hábitos de estudio'
    ]
  },
  hero: {
    headline: 'Aprender con corazón, para cada persona.',
    subheadline:
      'Curious Kelly une narrativa, ciencia y calidez humana para que tu familia aprenda todos los días.',
    ctaLabel: 'Reserva tu acceso 2026'
  },
  countdown: {
    labelActive: 'La oferta termina en',
    labelEnded: 'El grupo 2026 ya está completo.',
    offerEndedCta: 'Únete a la lista de espera',
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
    { key: 'schools', href: '/es-es/schools/', label: 'Centros' },
    { key: 'demo', href: '/es-es/demo/avatar/', label: 'Avatar en vivo' },
    { key: 'privacy', href: '/es-es/privacy/', label: 'Privacidad' },
    { key: 'cookies', href: '/es-es/cookies/', label: 'Cookies' }
  ],
  leadForm: {
    title: 'Cuéntanos quién debe conocer a Curious Kelly primero',
    subtitle:
      'Nuestro equipo de concierge confirmará tu inscripción y programará una sesión de lanzamiento adaptada a tus objetivos.',
    submitLabel: 'Enviar interés',
    submittingLabel: 'Enviando…',
    successHeading: '¡Gracias! Ya estás en la lista.',
    successBody:
      'Acabamos de avisar al equipo de concierge. Recibirás un mensaje en menos de un día hábil con tu kit de acceso.',
    successCta: 'Volver al inicio',
    errors: {
      generic: 'No pudimos guardar tu información. Inténtalo de nuevo o escribe a hello@curiouskelly.com.',
      turnstile: 'Completa la verificación para confirmar que eres humano.'
    },
    fields: {
      firstName: {
        label: 'Nombre',
        placeholder: 'Kelly',
        errors: {
          required: 'El nombre es obligatorio.',
          invalid: 'Solo se permiten letras, espacios, guiones y apóstrofes.'
        }
      },
      lastName: {
        label: 'Apellidos',
        placeholder: 'Rivera',
        errors: {
          required: 'Los apellidos son obligatorios.',
          invalid: 'Solo se permiten letras, espacios, guiones y apóstrofes.'
        }
      },
      email: {
        label: 'Correo electrónico',
        placeholder: 'tú@ejemplo.com',
        errors: {
          required: 'El correo electrónico es obligatorio.',
          invalid: 'Introduce una dirección de correo válida.'
        }
      },
      phone: {
        label: 'Móvil',
        placeholder: '+34 600 000 000',
        helpText: 'Incluye tu prefijo internacional para WhatsApp o SMS.',
        errors: {
          required: 'El móvil es obligatorio.',
          invalid: 'Introduce un número internacional válido.'
        }
      },
      country: {
        label: 'País / Región',
        placeholder: 'Selecciona un país',
        errors: {
          required: 'Selecciona un país.'
        }
      },
      region: {
        label: 'Provincia / Estado',
        placeholder: 'Selecciona una región',
        errors: {
          required: 'Selecciona una región según el país elegido.'
        }
      },
      marketingOptIn: {
        label: 'Quiero recibir novedades',
        description: 'Acepto recibir información sobre eventos y nuevas funciones de Curious Kelly.'
      }
    }
  },
  testimonials: {
    title: 'Voces de nuestros centros piloto',
    items: [
      { quote: 'Mejor docente digital del año — 2025.', author: 'Jurado EdTech Awards' },
      { quote: 'Por fin el aprendizaje tiene rostro.', author: 'Aria T.', role: 'Madre de una niña de 9 años' },
      { quote: 'Kelly despierta curiosidad en nuestro alumnado adulto cada mañana.', author: 'Jamal R.', role: 'Director de formación continua' }
    ]
  },
  features: {
    title: 'Por qué las familias eligen a Curious Kelly',
    items: [
      {
        title: 'Microlecciones diarias',
        description: 'Experiencias de 8 minutos que caben en cualquier agenda y mantienen la racha.',
        icon: 'clock'
      },
      {
        title: 'Privacidad garantizada',
        description: 'Sin anuncios ni patrones oscuros: tus datos permanecen cifrados y bajo tu control.',
        icon: 'shield'
      },
      {
        title: 'Trilingüe desde el inicio',
        description: 'Cada lección se redacta en inglés, español y portugués brasileño antes de publicarse.',
        icon: 'globe'
      },
      {
        title: 'Concierge humano',
        description: 'Una persona real revisa cada plan de incorporación para alinearlo con tus metas.',
        icon: 'people'
      }
    ]
  },
  pricing: {
    title: 'Ventajas del grupo fundador',
    subtitle: 'Reserva tu plaza 2026 ahora y asegura el precio de por vida.',
    options: [
      {
        title: 'Acceso familiar',
        description: 'Perfiles ilimitados para adultos y niños con coaching diario de rachas.'
      },
      {
        title: 'Kit docente',
        description: 'Exportaciones de lecciones, flujos de consentimiento y guías para grupos.'
      },
      {
        title: 'Alianzas con centros',
        description: 'Analítica a nivel distrito, marcos de privacidad y onboarding asistido.'
      }
    ],
    legal: 'Hoy no se realiza ningún pago. Confirmaremos elegibilidad y precio durante la incorporación.'
  },
  faq: {
    title: 'Preguntas frecuentes',
    items: [
      {
        question: '¿Cómo protege la privacidad Curious Kelly?',
        answer:
          'Precalculamos todo el contenido y nunca perfilamos a las personas. El contenido se cachea por idioma y dispositivo, pedimos consentimiento para cada tag y jamás vendemos datos.'
      },
      {
        question: '¿Kelly sustituirá al profesorado?',
        answer:
          'No. Kelly actúa como co-docente y se ocupa de la repetición y los recordatorios para que los educadores se centren en la conexión humana.'
      },
      {
        question: '¿Podemos probar Kelly antes de decidirnos?',
        answer:
          'Sí. El equipo de concierge compartirá un plan de muestra y conducirá una sesión guiada adaptada a tu audiencia.'
      }
    ]
  },
  trust: {
    title: 'Organizaciones que confían en los hábitos diarios de aprendizaje',
    items: [
      'Rising Stars EdTech 2025',
      'Consejo Europeo de Diseño de Privacidad',
      'Global Learning Collective',
      'Equity in AI Education Fund'
    ]
  },
  footer: {
    rights: '© 2025 Curious Kelly. Todos los derechos reservados.',
    privacy: 'Privacidad',
    cookies: 'Cookies',
    storeHeading: 'Disponible pronto'
  },
  consent: {
    title: 'Gestiona tus preferencias de privacidad',
    description:
      'Usamos analítica mínima y respetuosa para mejorar la experiencia. Las etiquetas de marketing solo cargan tras tu aceptación.',
    acceptAll: 'Aceptar todo',
    rejectAll: 'Rechazar no esenciales',
    manageLabel: 'Gestionar cookies',
    modalTitle: 'Ajustes de consentimiento',
    save: 'Guardar preferencias',
    categories: {
      strictlyNecessary: {
        label: 'Esenciales',
        description: 'Necesarias para seguridad, almacenamiento de consentimiento y servicio.'
      },
      analytics: {
        label: 'Analítica',
        description: 'Nos ayuda a entender el rendimiento y mejorar las lecciones.'
      },
      marketing: {
        label: 'Marketing',
        description: 'Permite píxeles opcionales como GTM, Meta o TikTok.'
      }
    }
  },
  analytics: {
    viewEvent: 'vista_pagina',
    leadSubmittedEvent: 'lead_enviado',
    leadErrorEvent: 'lead_error',
    consentChangedEvent: 'consentimiento_cambiado',
    localePromptShown: 'sugerencia_idioma_mostrada',
    localePromptAccepted: 'sugerencia_idioma_aceptada',
    localePromptDismissed: 'sugerencia_idioma_descartada',
    unityDemoEvent: 'evento_avatar_vivo'
  },
  localePrompt: {
    message: '¿Cambiar a {{language}} para disfrutar de Curious Kelly en tu idioma?',
    confirm: 'Cambiar idioma',
    dismiss: 'Seguir en inglés'
  },
  thankYou: {
    heading: 'Kelly ya tiene tu solicitud.',
    body: 'En breve recibirás un correo con los próximos pasos. Un concierge te contactará en 24 horas.',
    checklist: [
      'Añade hello@curiouskelly.com a tu lista segura.',
      'Invita a una persona colega a la sesión de onboarding.',
      'Prepara tus principales objetivos de aprendizaje para 2026.'
    ],
    back: 'Volver al inicio'
  },
  demoPage: {
    title: 'Vista previa del avatar en vivo',
    subtitle: 'Reproduce la compilación WebGL de Unity directamente en tu navegador y activa una lección real.',
    actions: {
      load: 'Reproducir lección de prueba',
      stop: 'Detener reproducción',
      reload: 'Recargar visor'
    },
    checklistTitle: 'Lo que verás',
    checklist: [
      'Unity se carga en el iframe en unos 5 segundos (archivos comprimidos).',
      'El audio y los visemas se sirven desde endpoints seguros.',
      'El puente postMessage informa estado, reproducción y errores en tiempo real.'
    ],
    status: {
      waiting: 'Esperando al reproductor de Unity…',
      ready: 'Kelly está lista. Carga la lección cuando quieras.',
      loading: 'Cargando recursos de la lección…',
      playing: 'La lección se está reproduciendo.',
      error: 'No pudimos conectar con el reproductor. Comprueba que el build esté desplegado.',
      assetsMissing: 'Configura PUBLIC_UNITY_SAMPLE_JSON/AUDIO para habilitar la reproducción.'
    },
    supportCta: 'Ver guía de despliegue',
    fallback: {
      title: '¿Necesitas ayuda?',
      body: 'Si el visor falla dos veces seguidas, escribe al equipo concierge y adjunta la consola del navegador.',
      ctaLabel: 'Contactar concierge'
    }
  }
};





