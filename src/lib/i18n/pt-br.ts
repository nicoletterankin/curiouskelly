import type { LocaleDictionary } from './types';

export const ptBR: LocaleDictionary = {
  code: 'pt-BR',
  languageName: 'Português (Brasil)',
  meta: {
    title: 'Curious Kelly — Companheira de aprendizagem com privacidade em primeiro lugar',
    description:
      'Curious Kelly entrega microlições diárias que constroem hábitos alegres para adultos, crianças e educadores. Inscrições abertas para 2026.',
    keywords: [
      'Curious Kelly',
      'tutora IA',
      'aprendizado diário',
      'tecnologia educacional',
      'hábitos de estudo'
    ]
  },
  hero: {
    headline: 'Aprender com afeto, para todo mundo.',
    subheadline:
      'Curious Kelly une história, ciência e calor humano para que sua família aprenda todos os dias.',
    ctaLabel: 'Garanta o acesso em 2026'
  },
  countdown: {
    labelActive: 'A oferta termina em',
    labelEnded: 'A turma 2026 está completa.',
    offerEndedCta: 'Entrar na lista de espera',
    units: {
      days: 'Dias',
      hours: 'Horas',
      minutes: 'Minutos',
      seconds: 'Segundos'
    }
  },
  nav: [
    { key: 'home', href: '/pt-br/', label: 'Início' },
    { key: 'adults', href: '/pt-br/adults/', label: 'Adultos' },
    { key: 'children', href: '/pt-br/children/', label: 'Crianças' },
    { key: 'teachers', href: '/pt-br/teachers/', label: 'Educadores' },
    { key: 'schools', href: '/pt-br/schools/', label: 'Escolas' },
    { key: 'demo', href: '/pt-br/demo/avatar/', label: 'Avatar ao vivo' },
    { key: 'privacy', href: '/pt-br/privacy/', label: 'Privacidade' },
    { key: 'cookies', href: '/pt-br/cookies/', label: 'Cookies' }
  ],
  leadForm: {
    title: 'Conte quem deve conhecer a Curious Kelly primeiro',
    subtitle:
      'Nossa equipe concierge confirma sua inscrição e agenda uma sessão de onboarding alinhada aos seus objetivos.',
    submitLabel: 'Enviar interesse',
    submittingLabel: 'Enviando…',
    successHeading: 'Obrigada! Você entrou para a lista.',
    successBody:
      'Avisamos a equipe concierge agora mesmo. Em até um dia útil você recebe um retorno com o kit de acesso.',
    successCta: 'Voltar para a página inicial',
    errors: {
      generic: 'Não foi possível registrar suas informações. Tente novamente ou escreva para concierge@curiouskelly.com.',
      turnstile: 'Complete a verificação para mostrar que você é humana.'
    },
    fields: {
      firstName: {
        label: 'Nome',
        placeholder: 'Kelly',
        errors: {
          required: 'O nome é obrigatório.',
          invalid: 'Use apenas letras, espaços, hifens e apóstrofos.'
        }
      },
      lastName: {
        label: 'Sobrenome',
        placeholder: 'Rivera',
        errors: {
          required: 'O sobrenome é obrigatório.',
          invalid: 'Use apenas letras, espaços, hifens e apóstrofos.'
        }
      },
      email: {
        label: 'E-mail',
        placeholder: 'voce@exemplo.com',
        errors: {
          required: 'O e-mail é obrigatório.',
          invalid: 'Informe um endereço de e-mail válido.'
        }
      },
      phone: {
        label: 'Telefone móvel',
        placeholder: '+55 11 90000-0000',
        helpText: 'Inclua o código do país para receber contato por WhatsApp ou SMS.',
        errors: {
          required: 'O telefone é obrigatório.',
          invalid: 'Informe um telefone internacional válido.'
        }
      },
      country: {
        label: 'País / Região',
        placeholder: 'Selecione um país',
        errors: {
          required: 'Selecione um país.'
        }
      },
      region: {
        label: 'Estado / Província',
        placeholder: 'Selecione uma região',
        errors: {
          required: 'Selecione uma região de acordo com o país escolhido.'
        }
      },
      marketingOptIn: {
        label: 'Quero receber novidades',
        description: 'Aceito receber novidades sobre eventos de lançamento e novas funcionalidades.'
      }
    }
  },
  testimonials: {
    title: 'Vozes das escolas piloto',
    items: [
      { quote: 'Melhor professora digital do ano — 2025.', author: 'Júri EdTech Awards' },
      { quote: 'Finalmente o aprendizado tem rosto.', author: 'Aria T.', role: 'Mãe de uma menina de 9 anos' },
      { quote: 'Kelly desperta curiosidade nas turmas adultas toda manhã.', author: 'Jamal R.', role: 'Diretor de educação continuada' }
    ]
  },
  features: {
    title: 'Por que famílias escolhem a Curious Kelly',
    items: [
      {
        title: 'Microlições diárias',
        description: 'Experiências de 8 minutos que cabem em qualquer rotina e mantêm a sequência ativa.',
        icon: 'clock'
      },
      {
        title: 'Privacidade garantida',
        description: 'Sem anúncios nem artifícios — os dados da sua família ficam criptografados e sob seu controle.',
        icon: 'shield'
      },
      {
        title: 'Trilíngue desde o primeiro dia',
        description: 'Cada lição é criada em inglês, espanhol e português brasileiro antes de ser entregue.',
        icon: 'globe'
      },
      {
        title: 'Concierge humano',
        description: 'Uma pessoa real revisa cada plano de onboarding para alinhar com as suas metas.',
        icon: 'people'
      }
    ]
  },
  pricing: {
    title: 'Benefícios da turma fundadora',
    subtitle: 'Reserve sua vaga 2026 agora e garanta o preço vitalício.',
    options: [
      {
        title: 'Acesso familiar',
        description: 'Perfis ilimitados para adultos e crianças com acompanhamento diário de hábitos.'
      },
      {
        title: 'Kit para educadores',
        description: 'Exportação de lições, fluxos de consentimento e guias para grupos.'
      },
      {
        title: 'Parcerias com escolas',
        description: 'Analíticos em nível de rede, políticas de privacidade e onboarding assistido.'
      }
    ],
    legal: 'Nenhum pagamento é cobrado agora. Confirmaremos elegibilidade e preço durante o onboarding.'
  },
  faq: {
    title: 'Perguntas frequentes',
    items: [
      {
        question: 'Como a Curious Kelly protege a privacidade?',
        answer:
          'Pré-calculamos todo o conteúdo e nunca criamos perfis. As lições são armazenadas por idioma e dispositivo, pedimos consentimento para cada tag e nunca vendemos dados.'
      },
      {
        question: 'Kelly vai substituir professores?',
        answer:
          'Não. Kelly é uma coeducadora que cuida da repetição e dos lembretes para que educadores foquem na conexão humana.'
      },
      {
        question: 'Podemos testar Kelly antes?',
        answer:
          'Sim. A equipe concierge envia um plano de amostra e conduz uma sessão ao vivo adaptada ao seu público.'
      }
    ]
  },
  trust: {
    title: 'Quem aposta em hábitos diários de aprendizagem',
    items: [
      'EdTech Rising Stars 2025',
      'Conselho Europeu de Design de Privacidade',
      'Global Learning Collective',
      'Equity in AI Education Fund'
    ]
  },
  footer: {
    rights: '© 2025 Curious Kelly. Todos os direitos reservados.',
    privacy: 'Privacidade',
    cookies: 'Cookies',
    storeHeading: 'Disponível em breve'
  },
  consent: {
    title: 'Gerencie suas escolhas de privacidade',
    description:
      'Usamos apenas analíticas essenciais para melhorar a experiência. Tags de marketing carregam somente após o seu consentimento.',
    acceptAll: 'Aceitar tudo',
    rejectAll: 'Recusar não essenciais',
    manageLabel: 'Gerenciar cookies',
    modalTitle: 'Configurações de consentimento',
    save: 'Salvar preferências',
    categories: {
      strictlyNecessary: {
        label: 'Essenciais',
        description: 'Necessárias para segurança, registro de consentimento e entrega do serviço.'
      },
      analytics: {
        label: 'Analíticas',
        description: 'Ajudam a entender desempenho e aprimorar as lições.'
      },
      marketing: {
        label: 'Marketing',
        description: 'Permitem pixels opcionais como GTM, Meta e TikTok.'
      }
    }
  },
  analytics: {
    viewEvent: 'visualizacao_pagina',
    leadSubmittedEvent: 'lead_enviado',
    leadErrorEvent: 'lead_erro',
    consentChangedEvent: 'consentimento_alterado',
    localePromptShown: 'sugestao_idioma_exibida',
    localePromptAccepted: 'sugestao_idioma_aceita',
    localePromptDismissed: 'sugestao_idioma_descartada',
    unityDemoEvent: 'evento_avatar_ao_vivo'
  },
  localePrompt: {
    message: 'Deseja mudar para {{language}} e viver a Curious Kelly no seu idioma?',
    confirm: 'Mudar idioma',
    dismiss: 'Continuar em inglês'
  },
  thankYou: {
    heading: 'Kelly registrou seu interesse.',
    body: 'Você receberá um e-mail com os próximos passos. Um concierge faz contato em até 24 horas.',
    checklist: [
      'Adicione concierge@curiouskelly.com aos contatos confiáveis.',
      'Convide alguém da sua equipe para a sessão de onboarding.',
      'Prepare seus principais objetivos de aprendizagem para 2026.'
    ],
    back: 'Voltar para o início'
  },
  demoPage: {
    title: 'Prévia do avatar ao vivo',
    subtitle: 'Carregue o build WebGL do Unity direto no navegador e envie uma lição real para Kelly.',
    actions: {
      load: 'Reproduzir lição de teste',
      stop: 'Parar reprodução',
      reload: 'Recarregar player'
    },
    checklistTitle: 'O que você verá',
    checklist: [
      'Unity inicializa no iframe em ~5 segundos (arquivos comprimidos).',
      'Áudio e visemas são servidos por endpoints seguros.',
      'A ponte postMessage sinaliza prontidão, reprodução e erros em tempo real.'
    ],
    status: {
      waiting: 'Aguardando o player do Unity…',
      ready: 'Kelly está pronta. Carregue a lição quando desejar.',
      loading: 'Carregando os recursos da lição…',
      playing: 'A lição está em reprodução.',
      error: 'Não conseguimos alcançar o player. Confirme se o build foi publicado.',
      assetsMissing: 'Defina PUBLIC_UNITY_SAMPLE_JSON/AUDIO para habilitar a reprodução.'
    },
    supportCta: 'Ver playbook de implantação',
    fallback: {
      title: 'Precisa de suporte?',
      body: 'Se o player falhar duas vezes, envie um e-mail para o concierge com o log do console do navegador.',
      ctaLabel: 'Falar com o concierge'
    }
  }
};





