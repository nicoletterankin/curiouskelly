import type { LocaleDictionary } from './types';

export const ptBR: LocaleDictionary = {
  code: 'pt-BR',
  languageName: 'Português (Brasil)',
  meta: {
    title: 'The Daily Lesson by Curious Kelly — Aprenda algo novo todos os dias',
    description:
      'Junte-se a milhares de estudantes curiosos com lições diárias de 8 minutos para adultos, crianças e educadores. Adaptado por idade. Três idiomas. Apenas $4.99/mês.',
    keywords: [
      'aprendizado diário',
      'companheira de aprendizado IA',
      'Kelly',
      'educação online',
      'aprender todo dia',
      'curiosidade'
    ]
  },
  hero: {
    headline: 'Aprenda algo novo todos os dias com Kelly',
    subheadline:
      'Lições diárias de 8 minutos para adultos, crianças e educadores. Adaptado por idade. Três idiomas. Um tópico universal.',
    ctaLabel: 'Comece seu teste gratuito de 7 dias'
  },
  countdown: {
    labelActive: 'Oferta especial de férias termina em',
    labelEnded: 'Presenteie 365 dias de curiosidade para 2026',
    offerEndedCta: 'Comprar assinatura de presente',
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
    { key: 'privacy', href: '/pt-br/privacy/', label: 'Privacidade' },
    { key: 'cookies', href: '/pt-br/cookies/', label: 'Cookies' }
  ],
  leadForm: {
    title: 'Comece seu teste gratuito de 7 dias',
    subtitle:
      'Sem cartão de crédito. Cancele quando quiser.',
    submitLabel: 'Começar a aprender grátis',
    submittingLabel: 'Iniciando seu teste…',
    successHeading: 'Bem-vindo! Tudo pronto.',
    successBody:
      'Confira seu e-mail para começar sua primeira lição. Estamos animados por ter você!',
    successCta: 'Voltar ao início',
    errors: {
      generic: 'Algo deu errado. Tente novamente ou escreva para support@curiouskelly.com',
      turnstile: 'Complete a verificação para continuar.'
    },
    fields: {
      firstName: {
        label: 'Nome',
        placeholder: 'Kelly',
        errors: {
          required: 'O nome é obrigatório.',
          invalid: 'Digite um nome válido.'
        }
      },
      lastName: {
        label: 'Sobrenome',
        placeholder: 'Rivera',
        errors: {
          required: 'O sobrenome é obrigatório.',
          invalid: 'Digite um sobrenome válido.'
        }
      },
      email: {
        label: 'E-mail',
        placeholder: 'voce@exemplo.com',
        errors: {
          required: 'O e-mail é obrigatório.',
          invalid: 'Digite um endereço válido.'
        }
      },
      phone: {
        label: 'Número de celular (opcional)',
        placeholder: '+55 11 99999-2026',
        helpText: 'Para lembretes de lições por mensagem',
        errors: {
          required: 'O celular é obrigatório.',
          invalid: 'Digite um número válido.'
        }
      },
      country: {
        label: 'País',
        placeholder: 'Selecione seu país',
        errors: {
          required: 'Selecione um país.'
        }
      },
      region: {
        label: 'Estado / Província',
        placeholder: 'Selecione sua região',
        errors: {
          required: 'Selecione uma região.'
        }
      },
      marketingOptIn: {
        label: 'Manter-me atualizado',
        description: 'Envie-me dicas, novas lições e ofertas especiais (opcional)'
      }
    }
  },
  testimonials: {
    title: 'Junte-se a milhares de estudantes diários',
    items: [
      { quote: 'Espero minha lição diária com Kelly todas as manhãs. É como café para meu cérebro.', author: 'Sarah M.', role: 'Estudante adulta' },
      { quote: 'Meu filho de 7 anos pede seu "tempo com Kelly" todos os dias depois da escola.', author: 'Marcus T.', role: 'Pai' },
      { quote: 'Perfeito iniciador de conversa para meus alunos. Mesmo tópico, todos os níveis podem participar.', author: 'Jamie R.', role: 'Professor do ensino médio' }
    ]
  },
  features: {
    title: 'Por que as pessoas escolhem The Daily Lesson',
    items: [
      {
        title: '8 minutos por dia',
        description: 'Se encaixa em qualquer horário. Cria hábitos duradouros. Sem sobrecarga.',
        icon: 'clock'
      },
      {
        title: 'Privacidade primeiro',
        description: 'Sem anúncios. Sem venda de dados. Sem rastreamento. Seu aprendizado é só seu.',
        icon: 'shield'
      },
      {
        title: 'Três idiomas',
        description: 'Inglês, espanhol e português. Toda lição. Todo dia.',
        icon: 'globe'
      },
      {
        title: 'Toda a família',
        description: 'Uma assinatura. Até 5 perfis. Todos aprendem juntos.',
        icon: 'people'
      }
    ]
  },
  pricing: {
    title: 'Preços simples e honestos',
    subtitle: 'Uma assinatura. Lições ilimitadas. Três idiomas.',
    options: [
      {
        title: 'Mensal',
        description: '$4.99/mês · Cancele quando quiser · Teste 7 dias grátis'
      },
      {
        title: 'Anual',
        description: '$49.99/ano · Economize $10 · Perfeito para presentear'
      }
    ],
    legal: 'Teste gratuito de 7 dias. Sem cartão de crédito para começar. Cancele quando quiser.'
  },
  faq: {
    title: 'Perguntas frequentes',
    items: [
      {
        question: 'Como funciona o teste gratuito?',
        answer:
          '7 dias grátis, sem cartão de crédito. Experimente lições ilimitadas nos três idiomas. Se você gostar, escolha mensal ($4.99) ou anual ($49.99). Cancele quando quiser com um clique.'
      },
      {
        question: 'Toda a minha família pode usar uma conta?',
        answer:
          'Sim! Crie até 5 perfis por assinatura. Cada pessoa recebe lições adaptadas à sua idade e acompanha seu próprio progresso.'
      },
      {
        question: 'Quais idiomas vocês oferecem?',
        answer:
          'Cada lição está disponível em inglês, espanhol e português brasileiro. Mude de idioma quando quiser.'
      },
      {
        question: 'É realmente para idades de 2 a 102?',
        answer:
          'Sim! O mesmo tópico universal se adapta à sua idade. Uma criança de 6 anos e uma de 60 podem discutir a mesma lição no jantar.'
      },
      {
        question: 'Posso presentear isso no Natal?',
        answer:
          'Com certeza! Assinaturas anuais ($49.99) são presentes perfeitos. Oferecemos entrega por e-mail e opção de mensagem personalizada para presentes de última hora.'
      },
      {
        question: 'Meus dados estão seguros?',
        answer:
          'Nunca vendemos seus dados, mostramos anúncios ou rastreamos você pela web. Privacidade primeiro significa privacidade sempre. Seu aprendizado permanece privado.'
      }
    ]
  },
  trust: {
    title: 'Confiado por estudantes curiosos em 47 países',
    items: [
      'Aprendizado sem anúncios',
      'Privacidade garantida',
      '365 lições universais',
      'Três idiomas incluídos'
    ]
  },
  footer: {
    rights: '© 2025 The Daily Lesson by Curious Kelly. Todos os direitos reservados.',
    privacy: 'Privacidade',
    cookies: 'Cookies',
    storeHeading: 'Baixe nosso aplicativo (em breve)'
  },
  consent: {
    title: 'Gerencie sua privacidade',
    description:
      'Usamos análise mínima para melhorar sua experiência. Cookies de marketing só carregam se você aceitar.',
    acceptAll: 'Aceitar tudo',
    rejectAll: 'Rejeitar cookies opcionais',
    manageLabel: 'Gerenciar preferências',
    modalTitle: 'Preferências de cookies',
    save: 'Salvar minhas preferências',
    categories: {
      strictlyNecessary: {
        label: 'Essenciais',
        description: 'Necessários para segurança e funcionalidade básica do site.'
      },
      analytics: {
        label: 'Análise',
        description: 'Nos ajuda a entender o que funciona e melhorar as lições.'
      },
      marketing: {
        label: 'Marketing',
        description: 'Pixels de rastreamento opcionais (Google, Meta, etc.)'
      }
    }
  },
  analytics: {
    viewEvent: 'visualizacao_pagina',
    leadSubmittedEvent: 'teste_iniciado',
    leadErrorEvent: 'erro_teste',
    consentChangedEvent: 'consentimento_alterado',
    localePromptShown: 'aviso_idioma_exibido',
    localePromptAccepted: 'idioma_alterado',
    localePromptDismissed: 'aviso_idioma_dispensado'
  },
  localePrompt: {
    message: 'Gostaria de mudar para {{language}}?',
    confirm: 'Mudar idioma',
    dismiss: 'Continuar em português'
  },
  thankYou: {
    heading: 'Bem-vindo ao The Daily Lesson!',
    body: 'Confira seu e-mail para sua primeira lição. Estamos animados para aprender com você.',
    checklist: [
      'Comece sua primeira lição de 8 minutos',
      'Crie perfis para sua família',
      'Explore lições em inglês, espanhol ou português'
    ],
    back: 'Voltar ao início'
  }
};





