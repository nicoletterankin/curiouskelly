import type { LocaleDictionary } from './types';

export const ptBR: LocaleDictionary = {
  code: 'pt-BR',
  languageName: 'Português (Brasil)',
  meta: {
    title: 'Curious Kelly — A companheira de aprendizagem com privacidade primeiro',
    description:
      'Curious Kelly oferece microlições diárias que criam hábitos de aprendizado felizes para adultos, crianças e educadores. Inscrições para 2026.',
    keywords: [
      'Curious Kelly',
      'tutora IA',
      'aprendizado diário',
      'tecnologia educacional',
      'hábitos de estudo'
    ]
  },
  hero: {
    headline: 'Aprender com carinho, para cada pessoa.',
    subheadline:
      'Curious Kelly une narrativa, ciência e calor humano para que sua comunidade aprenda todos os dias.',
    ctaLabel: 'Garanta acesso em 2026'
  },
  countdown: {
    labelActive: 'A oferta termina em',
    labelEnded: 'O grupo de 2026 já está completo.',
    offerEndedCta: 'Entre na lista de espera',
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
    { key: 'privacy', href: '/pt-br/privacy/', label: 'Privacidade' },
    { key: 'cookies', href: '/pt-br/cookies/', label: 'Cookies' }
  ],
  leadForm: {
    title: 'Conte quem deve conhecer a Curious Kelly primeiro',
    subtitle:
      'Nossa equipe concierge vai confirmar sua inscrição e marcar uma sessão de lançamento alinhada às suas metas.',
    submitLabel: 'Enviar interesse',
    submittingLabel: 'Enviando…',
    successHeading: 'Obrigadx! Você está na lista.',
    successBody:
      'Acabamos de avisar a equipe concierge. Você receberá contato em até um dia útil com seu kit de acesso.',
    successCta: 'Voltar ao início',
    errors: {
      generic: 'Não conseguimos salvar suas informações. Tente novamente ou escreva para hello@curiouskelly.com.',
      turnstile: 'Conclua a verificação para confirmar que você é humano.'
    },
    fields: {
      firstName: {
        label: 'Nome',
        placeholder: 'Kelly',
        errors: {
          required: 'O nome é obrigatório.',
          invalid: 'Apenas letras, espaços, hífens e apóstrofos são permitidos.'
        }
      },
      lastName: {
        label: 'Sobrenome',
        placeholder: 'Rivera',
        errors: {
          required: 'O sobrenome é obrigatório.',
          invalid: 'Apenas letras, espaços, hífens e apóstrofos são permitidos.'
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
        label: 'Celular',
        placeholder: '+55 11 99999-2026',
        helpText: 'Inclua seu código do país para receber WhatsApp ou SMS.',
        errors: {
          required: 'O celular é obrigatório.',
          invalid: 'Informe um número internacional válido.'
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
        description: 'Aceito receber atualizações sobre eventos e novos recursos da Curious Kelly.'
      }
    }
  },
  testimonials: {
    title: 'Vozes dos nossos parceiros piloto',
    items: [
      { quote: 'Melhor professora digital do ano — 2025.', author: 'Júri EdTech Awards' },
      { quote: 'Finalmente o aprendizado tem um rosto.', author: 'Aria T.', role: 'Mãe de uma criança de 9 anos' },
      { quote: 'Kelly desperta curiosidade em nossos estudantes adultos todas as manhãs.', author: 'Jamal R.', role: 'Diretor de faculdade comunitária' }
    ]
  },
  features: {
    title: 'Por que as comunidades escolhem a Curious Kelly',
    items: [
      {
        title: 'Microlições diárias',
        description: 'Experiências de 8 minutos que cabem em qualquer agenda e mantêm a sequência ativa.',
        icon: 'clock'
      },
      {
        title: 'Privacidade em primeiro lugar',
        description: 'Sem anúncios, sem padrões escuros — seus dados permanecem criptografados e sob seu controle.',
        icon: 'shield'
      },
      {
        title: 'Trilíngue desde o início',
        description: 'Cada lição é criada em inglês, espanhol e português brasileiro antes da entrega.',
        icon: 'globe'
      },
      {
        title: 'Concierge humano',
        description: 'Uma pessoa real revisa cada plano de onboarding para alinhar com suas metas.',
        icon: 'people'
      }
    ]
  },
  pricing: {
    title: 'Benefícios do grupo fundador',
    subtitle: 'Garanta sua vaga para 2026 agora e congele o preço para sempre.',
    options: [
      {
        title: 'Acesso para famílias',
        description: 'Perfis ilimitados para adultos e crianças com coaching diário de sequência.'
      },
      {
        title: 'Kit para educadores',
        description: 'Exportação de lições, fluxos de consentimento e guias prontos para grupos.'
      },
      {
        title: 'Parcerias com escolas',
        description: 'Analytics em nível de rede, estruturas de privacidade e onboarding assistido.'
      }
    ],
    legal: 'Nenhum pagamento é cobrado hoje. Confirmaremos elegibilidade e preço durante o onboarding.'
  },
  faq: {
    title: 'Perguntas frequentes',
    items: [
      {
        question: 'Como a Curious Kelly protege a privacidade dos aprendizes?',
        answer:
          'Pré-calculamos todo o conteúdo e nunca criamos perfis. O conteúdo é armazenado em cache por idioma e dispositivo, pedimos consentimento para cada tag e nunca vendemos dados.'
      },
      {
        question: 'A Kelly vai substituir professores?',
        answer:
          'Não. A Kelly foi criada como co-professora, cuidando da repetição e dos lembretes para que os educadores foquem na conexão humana.'
      },
      {
        question: 'Podemos experimentar a Kelly antes de decidir?',
        answer:
          'Sim. A equipe concierge compartilhará um plano de lição e fará uma apresentação guiada adaptada ao seu público.'
      }
    ]
  },
  trust: {
    title: 'Confiada por equipes que cuidam de hábitos diários de aprendizado',
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
      'Usamos análise mínima e respeitosa para melhorar a experiência. Tags de marketing só carregam após sua autorização.',
    acceptAll: 'Aceitar tudo',
    rejectAll: 'Rejeitar não essenciais',
    manageLabel: 'Gerenciar cookies',
    modalTitle: 'Configurações de consentimento',
    save: 'Salvar preferências',
    categories: {
      strictlyNecessary: {
        label: 'Essenciais',
        description: 'Necessárias para segurança, armazenamento de consentimento e entrega do serviço.'
      },
      analytics: {
        label: 'Analytics',
        description: 'Ajuda a entender desempenho e melhorar as lições.'
      },
      marketing: {
        label: 'Marketing',
        description: 'Permite pixels opcionais como GTM, Meta e TikTok.'
      }
    }
  },
  analytics: {
    viewEvent: 'visualizacao_pagina',
    leadSubmittedEvent: 'lead_enviado',
    leadErrorEvent: 'lead_erro',
    consentChangedEvent: 'consentimento_alterado',
    localePromptShown: 'aviso_idioma_exibido',
    localePromptAccepted: 'aviso_idioma_aceito',
    localePromptDismissed: 'aviso_idioma_dispensado'
  },
  localePrompt: {
    message: 'Alterar para {{language}} e aproveitar a Curious Kelly no seu idioma?',
    confirm: 'Alterar idioma',
    dismiss: 'Permanecer em inglês'
  },
  thankYou: {
    heading: 'Kelly já recebeu sua solicitação.',
    body: 'Você receberá um e-mail com os próximos passos. Um concierge entrará em contato em até 24 horas.',
    checklist: [
      'Adicione hello@curiouskelly.com à sua lista de contatos confiáveis.',
      'Convide uma pessoa colega para a sessão de onboarding.',
      'Prepare seus principais objetivos de aprendizado para 2026.'
    ],
    back: 'Voltar ao início'
  }
};








