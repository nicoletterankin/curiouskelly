# 📋 LESSON SCHEMA v5.0 - CANONICAL REFERENCE

## Single Source of Truth for All Lesson Data

**Version:** 5.0-full-choices-i18n  
**Created:** December 17, 2025  
**Status:** ✅ LOCKED — All systems must implement this schema

---

## 🎯 Key Principles

1. **ALL phases have choices** (A/B options with feedback)
2. **Multilingual from the start** (EN + ES + PT embedded)
3. **Phases renamed to q1/q2/q3** (not fact1/fact2/fact3)
4. **JSON is source of truth** — Supabase syncs FROM JSON files

---

## 📊 Schema Overview

```
lesson.json
├── meta                          # Lesson metadata
│   ├── day: number               # 1-365
│   ├── date: string              # "2025-01-01"
│   ├── topic: { en, es, pt }     # Translated topic name
│   ├── emoji: string             # Universal emoji
│   ├── category: string          # English key (e.g., "Beginnings")
│   ├── version: string           # "v5.0-full-choices-i18n"
│   └── languages: string[]       # ["en", "es", "pt"]
│
├── headline: { en, es, pt }      # Marketing headline
├── universal_truth: { en, es, pt }
├── fun_facts: [{ en, es, pt }]   # Array of 3
├── discussion_questions: [{ en, es, pt }]
│
├── phases                        # 7 phases
│   ├── hook                      # Opening
│   ├── cliff                     # Choice point
│   ├── q1                        # Fact 1 (renamed from fact1)
│   ├── q2                        # Fact 2 (renamed from fact2)
│   ├── q3                        # Fact 3 (renamed from fact3)
│   ├── wisdom                    # Reflection
│   └── outro                     # Closing
│
├── phaseOrder: string[]          # ["hook", "cliff", "q1", "q2", "q3", "wisdom", "outro"]
├── totalDuration: { en, es, pt } # Duration varies by language
│
└── growTrack                     # Optional activity
    ├── title: { en, es, pt }
    ├── learning_objective: { en, es, pt }
    └── activity: { en, es, pt }
```

---

## 📍 Phase Schema (EVERY phase has this structure)

```json
{
  "title": {
    "en": "Welcome",
    "es": "Bienvenida",
    "pt": "Boas-vindas"
  },
  "script": {
    "en": "Welcome to Day One...",
    "es": "Bienvenidos al Día Uno...",
    "pt": "Bem-vindos ao Dia Um..."
  },
  "duration": {
    "en": 12,
    "es": 14,
    "pt": 13
  },
  "prompt": {
    "en": "What brings you here today?",
    "es": "¿Qué te trae aquí hoy?",
    "pt": "O que te traz aqui hoje?"
  },
  "options": [
    {
      "letter": "A",
      "text": {
        "en": "I want to learn something new",
        "es": "Quiero aprender algo nuevo",
        "pt": "Quero aprender algo novo"
      },
      "quality": "best",
      "response": {
        "en": "Perfect. Curiosity is the spark...",
        "es": "Perfecto. La curiosidad es la chispa...",
        "pt": "Perfeito. A curiosidade é a faísca..."
      }
    },
    {
      "letter": "B",
      "text": {
        "en": "I'm looking for a change",
        "es": "Estoy buscando un cambio",
        "pt": "Estou buscando uma mudança"
      },
      "quality": "best",
      "response": {
        "en": "That feeling of wanting something different...",
        "es": "Ese sentimiento de querer algo diferente...",
        "pt": "Esse sentimento de querer algo diferente..."
      }
    }
  ]
}
```

---

## 🏗️ Complete Example: Day 1

```json
{
  "meta": {
    "day": 1,
    "date": "2025-01-01",
    "topic": {
      "en": "Starting Fresh",
      "es": "Empezando de Nuevo",
      "pt": "Começando de Novo"
    },
    "emoji": "🍁",
    "category": "Beginnings",
    "version": "v5.0-full-choices-i18n",
    "target_audience": "adult",
    "voice_id": "wAdymQH5YucAkXwmrdL0",
    "languages": ["en", "es", "pt"]
  },
  "headline": {
    "en": "Every ending holds the seed of a new beginning",
    "es": "Cada final contiene la semilla de un nuevo comienzo",
    "pt": "Todo fim contém a semente de um novo começo"
  },
  "universal_truth": {
    "en": "Fresh starts are available in every moment, not just on special days",
    "es": "Los nuevos comienzos están disponibles en cada momento, no solo en días especiales",
    "pt": "Recomeços estão disponíveis em cada momento, não apenas em dias especiais"
  },
  "fun_facts": [
    {
      "en": "Research shows it takes about 66 days on average to form a new habit—not the commonly cited 21",
      "es": "Las investigaciones muestran que se necesitan unos 66 días en promedio para formar un nuevo hábito, no los 21 comúnmente citados",
      "pt": "Pesquisas mostram que leva cerca de 66 dias em média para formar um novo hábito—não os 21 comumente citados"
    },
    {
      "en": "The brain's neuroplasticity means you can literally rewire your thinking patterns at any age",
      "es": "La neuroplasticidad del cerebro significa que puedes reconectar tus patrones de pensamiento a cualquier edad",
      "pt": "A neuroplasticidade do cérebro significa que você pode literalmente reconectar seus padrões de pensamento em qualquer idade"
    },
    {
      "en": "People who write down their goals are 42% more likely to achieve them",
      "es": "Las personas que escriben sus metas tienen un 42% más de probabilidades de lograrlas",
      "pt": "Pessoas que escrevem seus objetivos têm 42% mais probabilidade de alcançá-los"
    }
  ],
  "phases": {
    "hook": {
      "title": { "en": "Welcome", "es": "Bienvenida", "pt": "Boas-vindas" },
      "script": {
        "en": "Welcome to Day One. Not just of this journey, but of something bigger...",
        "es": "Bienvenidos al Día Uno. No solo de este viaje, sino de algo más grande...",
        "pt": "Bem-vindos ao Dia Um. Não apenas desta jornada, mas de algo maior..."
      },
      "duration": { "en": 12, "es": 14, "pt": 13 },
      "prompt": {
        "en": "What brings you here today?",
        "es": "¿Qué te trae aquí hoy?",
        "pt": "O que te traz aqui hoje?"
      },
      "options": [
        {
          "letter": "A",
          "text": { "en": "I want to learn something new", "es": "Quiero aprender algo nuevo", "pt": "Quero aprender algo novo" },
          "quality": "best",
          "response": { "en": "Perfect. Curiosity is the first step...", "es": "Perfecto. La curiosidad es el primer paso...", "pt": "Perfeito. A curiosidade é o primeiro passo..." }
        },
        {
          "letter": "B",
          "text": { "en": "I'm curious about this topic", "es": "Tengo curiosidad sobre este tema", "pt": "Estou curioso sobre este tema" },
          "quality": "best",
          "response": { "en": "That curiosity is exactly what we need...", "es": "Esa curiosidad es exactamente lo que necesitamos...", "pt": "Essa curiosidade é exatamente o que precisamos..." }
        }
      ]
    },
    "cliff": {
      "title": { "en": "The Question", "es": "La Pregunta", "pt": "A Pergunta" },
      "script": {
        "en": "Here's what's fascinating: our brains are wired to love new beginnings...",
        "es": "Esto es lo fascinante: nuestros cerebros están programados para amar los nuevos comienzos...",
        "pt": "Eis o fascinante: nossos cérebros são programados para amar novos começos..."
      },
      "duration": { "en": 12, "es": 14, "pt": 13 },
      "prompt": {
        "en": "What makes a fresh start actually work?",
        "es": "¿Qué hace que un nuevo comienzo realmente funcione?",
        "pt": "O que faz um recomeço realmente funcionar?"
      },
      "options": [
        {
          "letter": "A",
          "text": { "en": "Strong willpower and discipline", "es": "Fuerza de voluntad y disciplina", "pt": "Força de vontade e disciplina" },
          "quality": "good",
          "response": { "en": "Willpower helps, but research shows it's not the whole story...", "es": "La fuerza de voluntad ayuda, pero las investigaciones muestran que no es toda la historia...", "pt": "A força de vontade ajuda, mas pesquisas mostram que não é toda a história..." }
        },
        {
          "letter": "B",
          "text": { "en": "Small, consistent actions over time", "es": "Pequeñas acciones consistentes con el tiempo", "pt": "Pequenas ações consistentes ao longo do tempo" },
          "quality": "best",
          "response": { "en": "Exactly. Tiny changes compound into transformation.", "es": "Exacto. Los pequeños cambios se acumulan en transformación.", "pt": "Exatamente. Pequenas mudanças se acumulam em transformação." }
        }
      ]
    },
    "q1": {
      "title": { "en": "The 66-Day Truth", "es": "La Verdad de los 66 Días", "pt": "A Verdade dos 66 Dias" },
      "script": {
        "en": "You've probably heard it takes 21 days to form a habit. That number? Made up in the 1960s...",
        "es": "Probablemente has escuchado que se necesitan 21 días para formar un hábito. ¿Ese número? Inventado en los años 60...",
        "pt": "Você provavelmente ouviu que leva 21 dias para formar um hábito. Esse número? Inventado nos anos 60..."
      },
      "duration": { "en": 18, "es": 21, "pt": 19 },
      "prompt": {
        "en": "If you miss a day on a new habit, what happens?",
        "es": "Si pierdes un día en un nuevo hábito, ¿qué pasa?",
        "pt": "Se você perde um dia em um novo hábito, o que acontece?"
      },
      "options": [
        {
          "letter": "A",
          "text": { "en": "You have to start over from day one", "es": "Tienes que empezar de nuevo desde el día uno", "pt": "Você tem que começar de novo do dia um" },
          "quality": "misconception",
          "response": { "en": "This is a common myth! Research shows missing one day barely affects habit formation...", "es": "¡Este es un mito común! Las investigaciones muestran que perder un día apenas afecta la formación de hábitos...", "pt": "Este é um mito comum! Pesquisas mostram que perder um dia mal afeta a formação de hábitos..." }
        },
        {
          "letter": "B",
          "text": { "en": "It's a small bump—you keep going", "es": "Es un pequeño tropiezo—continúas adelante", "pt": "É um pequeno tropeço—você continua" },
          "quality": "best",
          "response": { "en": "Exactly right. One missed day doesn't erase your progress...", "es": "Exactamente. Un día perdido no borra tu progreso...", "pt": "Exatamente. Um dia perdido não apaga seu progresso..." }
        }
      ]
    },
    "q2": {
      "title": { "en": "Your Brain Can Change", "es": "Tu Cerebro Puede Cambiar", "pt": "Seu Cérebro Pode Mudar" },
      "script": {
        "en": "Neuroplasticity—your brain's ability to rewire itself—doesn't expire with age...",
        "es": "La neuroplasticidad—la capacidad de tu cerebro para reconectarse—no expira con la edad...",
        "pt": "Neuroplasticidade—a capacidade do seu cérebro de se reconectar—não expira com a idade..."
      },
      "duration": { "en": 15, "es": 18, "pt": 16 },
      "prompt": {
        "en": "At what age does your brain stop being able to change?",
        "es": "¿A qué edad tu cerebro deja de poder cambiar?",
        "pt": "Em que idade seu cérebro para de poder mudar?"
      },
      "options": [
        {
          "letter": "A",
          "text": { "en": "Around age 25 when the brain fully develops", "es": "Alrededor de los 25 años cuando el cerebro se desarrolla completamente", "pt": "Por volta dos 25 anos quando o cérebro se desenvolve completamente" },
          "quality": "misconception",
          "response": { "en": "While the brain does mature around 25, neuroplasticity continues your entire life...", "es": "Aunque el cerebro madura alrededor de los 25, la neuroplasticidad continúa toda tu vida...", "pt": "Embora o cérebro amadureça por volta dos 25, a neuroplasticidade continua por toda a vida..." }
        },
        {
          "letter": "B",
          "text": { "en": "It never stops—change is always possible", "es": "Nunca se detiene—el cambio siempre es posible", "pt": "Nunca para—a mudança sempre é possível" },
          "quality": "best",
          "response": { "en": "Yes! Your brain maintains the ability to change throughout your entire life.", "es": "¡Sí! Tu cerebro mantiene la capacidad de cambiar durante toda tu vida.", "pt": "Sim! Seu cérebro mantém a capacidade de mudar durante toda a vida." }
        }
      ]
    },
    "q3": {
      "title": { "en": "Write It Down", "es": "Escríbelo", "pt": "Escreva" },
      "script": {
        "en": "Dr. Gail Matthews found that people who wrote down their goals were 42% more likely to achieve them...",
        "es": "La Dra. Gail Matthews descubrió que las personas que escribieron sus metas tenían un 42% más de probabilidades de lograrlas...",
        "pt": "A Dra. Gail Matthews descobriu que pessoas que escreveram seus objetivos tinham 42% mais probabilidade de alcançá-los..."
      },
      "duration": { "en": 13, "es": 15, "pt": 14 },
      "prompt": {
        "en": "Why does writing down goals make them more achievable?",
        "es": "¿Por qué escribir las metas las hace más alcanzables?",
        "pt": "Por que escrever os objetivos os torna mais alcançáveis?"
      },
      "options": [
        {
          "letter": "A",
          "text": { "en": "It forces you to be specific and clear", "es": "Te obliga a ser específico y claro", "pt": "Te obriga a ser específico e claro" },
          "quality": "best",
          "response": { "en": "That's a big part of it. Writing requires clarity...", "es": "Esa es una gran parte. Escribir requiere claridad...", "pt": "Essa é uma grande parte. Escrever requer clareza..." }
        },
        {
          "letter": "B",
          "text": { "en": "It creates accountability", "es": "Crea responsabilidad", "pt": "Cria responsabilidade" },
          "quality": "best",
          "response": { "en": "Also true! There's something powerful about seeing your commitment in writing.", "es": "¡También es verdad! Hay algo poderoso en ver tu compromiso por escrito.", "pt": "Também verdade! Há algo poderoso em ver seu compromisso por escrito." }
        }
      ]
    },
    "wisdom": {
      "title": { "en": "Today's Wisdom", "es": "La Sabiduría de Hoy", "pt": "A Sabedoria de Hoje" },
      "script": {
        "en": "Here's today's wisdom: You don't need January 1st to start fresh. Every morning—every moment—is a chance to begin again...",
        "es": "Esta es la sabiduría de hoy: No necesitas el 1 de enero para empezar de nuevo. Cada mañana—cada momento—es una oportunidad de comenzar de nuevo...",
        "pt": "Eis a sabedoria de hoje: Você não precisa de 1º de janeiro para recomeçar. Cada manhã—cada momento—é uma chance de começar de novo..."
      },
      "duration": { "en": 14, "es": 16, "pt": 15 },
      "prompt": {
        "en": "What resonates more with you?",
        "es": "¿Qué resuena más contigo?",
        "pt": "O que ressoa mais com você?"
      },
      "options": [
        {
          "letter": "A",
          "text": { "en": "Fresh starts can happen any moment", "es": "Los nuevos comienzos pueden ocurrir en cualquier momento", "pt": "Recomeços podem acontecer a qualquer momento" },
          "quality": "best",
          "response": { "en": "This insight frees you from waiting for the 'perfect time.'", "es": "Esta idea te libera de esperar el 'momento perfecto.'", "pt": "Essa percepção te liberta de esperar o 'momento perfeito.'" }
        },
        {
          "letter": "B",
          "text": { "en": "The second best time is now", "es": "El segundo mejor momento es ahora", "pt": "O segundo melhor momento é agora" },
          "quality": "best",
          "response": { "en": "This ancient wisdom reminds us that regret about the past is wasted energy.", "es": "Esta sabiduría antigua nos recuerda que el arrepentimiento por el pasado es energía desperdiciada.", "pt": "Esta sabedoria antiga nos lembra que arrependimento sobre o passado é energia desperdiçada." }
        }
      ]
    },
    "outro": {
      "title": { "en": "See You Tomorrow", "es": "Hasta Mañana", "pt": "Até Amanhã" },
      "script": {
        "en": "That's Day 1. Tomorrow, we'll explore the three states of water and what they teach us about change...",
        "es": "Eso es el Día 1. Mañana exploraremos los tres estados del agua y lo que nos enseñan sobre el cambio...",
        "pt": "Esse é o Dia 1. Amanhã exploraremos os três estados da água e o que eles nos ensinam sobre mudança..."
      },
      "duration": { "en": 10, "es": 12, "pt": 11 },
      "prompt": {
        "en": "Before you go—what will you take from today?",
        "es": "Antes de irte—¿qué te llevarás de hoy?",
        "pt": "Antes de ir—o que você vai levar de hoje?"
      },
      "options": [
        {
          "letter": "A",
          "text": { "en": "I'll write down one goal today", "es": "Voy a escribir una meta hoy", "pt": "Vou escrever um objetivo hoje" },
          "quality": "best",
          "response": { "en": "Powerful choice. That 42% boost starts with this single action.", "es": "Elección poderosa. Ese impulso del 42% comienza con esta única acción.", "pt": "Escolha poderosa. Esse impulso de 42% começa com esta única ação." }
        },
        {
          "letter": "B",
          "text": { "en": "I'll give myself permission to start fresh", "es": "Me daré permiso para empezar de nuevo", "pt": "Vou me dar permissão para recomeçar" },
          "quality": "best",
          "response": { "en": "Beautiful. Sometimes the biggest shift is simply allowing yourself to begin again.", "es": "Hermoso. A veces el mayor cambio es simplemente permitirte comenzar de nuevo.", "pt": "Lindo. Às vezes a maior mudança é simplesmente se permitir começar de novo." }
        }
      ]
    }
  },
  "phaseOrder": ["hook", "cliff", "q1", "q2", "q3", "wisdom", "outro"],
  "totalDuration": { "en": 94, "es": 110, "pt": 100 },
  "growTrack": {
    "title": { "en": "Fresh Start - Setting Intentions", "es": "Nuevo Comienzo - Estableciendo Intenciones", "pt": "Recomeço - Definindo Intenções" },
    "emoji": "🍁",
    "learning_objective": { "en": "Create a meaningful intention for your learning journey", "es": "Crear una intención significativa para tu viaje de aprendizaje", "pt": "Criar uma intenção significativa para sua jornada de aprendizado" },
    "activity": { "en": "Write down one thing you want to learn...", "es": "Escribe una cosa que quieras aprender...", "pt": "Escreva uma coisa que você quer aprender..." }
  }
}
```

---

## 🗄️ Supabase Schema Mapping

### core_lessons Table

| Field | Type | Source |
|-------|------|--------|
| `id` | uuid | Auto-generated |
| `day_number` | integer | `meta.day` |
| `topic` | text | `meta.topic.en` (legacy) |
| `topic_i18n` | jsonb | `meta.topic` (new) |
| `universal_truth` | text | `universal_truth.en` (legacy) |
| `universal_truth_i18n` | jsonb | `universal_truth` (new) |
| `marketing_headline` | text | `headline.en` |
| `headline_i18n` | jsonb | `headline` |
| `icon_emoji` | text | `meta.emoji` |

### lesson_atoms Table

| Field | Type | Source |
|-------|------|--------|
| `id` | uuid | Auto-generated |
| `core_lesson_id` | uuid | FK to core_lessons |
| `day_number` | integer | `meta.day` |
| `phase` | text | Phase key (hook, cliff, q1, etc.) |
| `archetype` | text | "The Scientist" (default) |
| `content` | jsonb | Phase object with i18n structure |

### content JSONB Structure

```json
{
  "title": { "en": "...", "es": "...", "pt": "..." },
  "script": { "en": "...", "es": "...", "pt": "..." },
  "duration": { "en": 12, "es": 14, "pt": 13 },
  "prompt": { "en": "...", "es": "...", "pt": "..." },
  "options": [
    {
      "letter": "A",
      "text": { "en": "...", "es": "...", "pt": "..." },
      "quality": "best",
      "response": { "en": "...", "es": "...", "pt": "..." }
    },
    {
      "letter": "B",
      "text": { "en": "...", "es": "...", "pt": "..." },
      "quality": "good",
      "response": { "en": "...", "es": "...", "pt": "..." }
    }
  ],
  // Legacy compatibility (read-only)
  "choice_intro": "...",
  "option_a": "...",
  "option_b": "...",
  "success_response": "...",
  "alt_response": "..."
}
```

---

## 🔌 API Contract

### GET /api/lessons/:day

**Request:**
```
GET /api/lessons/1?lang=es&archetype=The+Scientist
```

**Response:**
```json
{
  "lesson": {
    "day_number": 1,
    "topic": "Empezando de Nuevo",
    "headline": "Cada final contiene la semilla de un nuevo comienzo",
    "universal_truth": "Los nuevos comienzos están disponibles en cada momento..."
  },
  "atoms": [
    {
      "phase": "hook",
      "script": "Bienvenidos al Día Uno...",
      "prompt": "¿Qué te trae aquí hoy?",
      "option_a": "Quiero aprender algo nuevo",
      "option_b": "Tengo curiosidad sobre este tema",
      "success_response": "Perfecto. La curiosidad es el primer paso...",
      "alt_response": "Esa curiosidad es exactamente lo que necesitamos..."
    }
  ]
}
```

---

## 📍 Visual Commons Mapping

Each phase now requires:

| Visual Type | Count per Phase | Purpose |
|-------------|-----------------|---------|
| `scene` | 1 | Background/wallpaper |
| `choice_a` | 1 | Option A visual |
| `choice_b` | 1 | Option B visual |

**Total per lesson: 21 visuals** (7 phases × 3 visual types)

---

## 🔄 Migration Path

### Current State (v4.0)
- English only
- fact1/fact2/fact3 phase names
- Only cliff has choices

### Target State (v5.0)
- EN/ES/PT embedded
- q1/q2/q3 phase names
- ALL phases have choices

### Migration Scripts

1. `migrate-lessons-full-choices.js` - Add choices to all phases (DONE)
2. `migrate-lessons-i18n.js` - Add ES/PT translations (NEEDED)
3. `sync-lessons-to-supabase.js` - Push to database (DONE, needs i18n update)

---

## ✅ Validation Checklist

A lesson is COMPLETE when:

- [ ] All 7 phases present (hook, cliff, q1, q2, q3, wisdom, outro)
- [ ] Each phase has: title, script, duration, prompt, options[A,B]
- [ ] Each field has: en, es, pt translations
- [ ] Options have: letter, text, quality, response
- [ ] Version is "v5.0-full-choices-i18n"
- [ ] JSON is valid and parseable
- [ ] Synced to Supabase

---

## 🚫 Forbidden

1. **Never** use fact1/fact2/fact3 - use q1/q2/q3
2. **Never** have phases without choices
3. **Never** have string values where i18n object expected
4. **Never** skip a language (EN/ES/PT all required)
5. **Never** translate technical fields (day, emoji, letter, quality)

---

## 📚 Reference Documents

- `INTERNATIONALIZATION_MASTER_PLAN.md` - i18n architecture
- `LESSON_TRANSLATION_RULES.md` - Translation guidelines
- `PHASE_ARTIFACT_MATRIX.md` - Visual requirements per phase
- `VISUAL_INFRASTRUCTURE_BLUEPRINT.md` - Visual placement

---

*This document is the SINGLE SOURCE OF TRUTH for lesson data structure. All systems (frontend, backend, content generation, translations) must align to this schema.*
