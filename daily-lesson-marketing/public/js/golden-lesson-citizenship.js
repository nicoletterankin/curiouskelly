/**
 * GOLDEN LESSON: Citizenship (Day 333)
 *
 * This is the perfected template with:
 * - ALL 6 age variants with unique text
 * - ALL 3 language translations
 * - ALL tone variants (encouraging, direct, playful)
 * - ALL difficulty levels (2 or 3 choices)
 * - Kelly responses for each choice
 *
 * Use this as the model for all 365 lessons.
 */

const GOLDEN_LESSON_333 = {
  id: 'citizenship-333',
  dayNumber: 333,
  date: 'November 28',
  topic: 'Citizenship',
  topicEmoji: '🏛️',
  objective: 'Understand what it means to participate in and contribute to your community.',

  // Audio files per phase/age/language (ElevenLabs generated)
  audioBase: '/audio/lessons/333/',

  phases: [
    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: WELCOME
    // ═══════════════════════════════════════════════════════════════════
    {
      order: 1,
      type: 'welcome',
      name: 'Welcome',
      expression: 'curious',

      // Text varies by AGE and LANGUAGE
      text: {
        '2-5': {
          en: "Hi friend! 👋 Today we're going to learn about being a good neighbor! Do you know what a neighbor is? It's everyone around you — your family, your friends, the people on your street!",
          es: '¡Hola amiguito! 👋 ¡Hoy vamos a aprender sobre ser un buen vecino! ¿Sabes qué es un vecino? ¡Son todas las personas a tu alrededor: tu familia, tus amigos, la gente de tu calle!',
          fr: "Salut mon ami ! 👋 Aujourd'hui, on va apprendre à être un bon voisin ! Tu sais ce que c'est un voisin ? C'est tous ceux autour de toi : ta famille, tes amis, les gens de ta rue !"
        },
        '6-12': {
          en: "Hey there, explorer! 🌟 Today we're diving into something super important: citizenship! It's like being a superhero for your community. Ready to discover your powers?",
          es: '¡Hola, explorador! 🌟 ¡Hoy vamos a sumergirnos en algo súper importante: la ciudadanía! Es como ser un superhéroe para tu comunidad. ¿Listo para descubrir tus poderes?',
          fr: "Salut, explorateur ! 🌟 Aujourd'hui, on plonge dans quelque chose de super important : la citoyenneté ! C'est comme être un super-héros pour ta communauté. Prêt à découvrir tes pouvoirs ?"
        },
        '13-17': {
          en: "What's up! 🎯 Ever wondered what citizenship actually means beyond what's in textbooks? Today we're breaking it down — and trust me, it's way more relevant to your life than you might think.",
          es: '¡Qué tal! 🎯 ¿Alguna vez te preguntaste qué significa realmente la ciudadanía más allá de los libros de texto? Hoy lo vamos a desglosar — y créeme, es mucho más relevante para tu vida de lo que piensas.',
          fr: "Salut ! 🎯 Tu t'es déjà demandé ce que signifie vraiment la citoyenneté au-delà des manuels ? Aujourd'hui, on décortique tout ça — et crois-moi, c'est bien plus pertinent pour ta vie que tu ne le penses."
        },
        '18-35': {
          en: "Welcome! 🏛️ Citizenship — it's a word we hear often, but what does it really mean in practice? Today we're exploring how being an engaged citizen shapes not just your community, but your career and relationships too.",
          es: '¡Bienvenido! 🏛️ Ciudadanía — es una palabra que escuchamos a menudo, pero ¿qué significa realmente en la práctica? Hoy exploraremos cómo ser un ciudadano comprometido moldea no solo tu comunidad, sino también tu carrera y relaciones.',
          fr: "Bienvenue ! 🏛️ La citoyenneté — un mot qu'on entend souvent, mais que signifie-t-il vraiment en pratique ? Aujourd'hui, on explore comment être un citoyen engagé façonne non seulement ta communauté, mais aussi ta carrière et tes relations."
        },
        '36-60': {
          en: "Hello there! 🌍 As someone who's likely building a family or leading in your community, citizenship takes on deeper meaning. Today, let's explore how your actions ripple outward — to your kids, your neighbors, your world.",
          es: '¡Hola! 🌍 Como alguien que probablemente está formando una familia o liderando en su comunidad, la ciudadanía adquiere un significado más profundo. Hoy, exploremos cómo tus acciones se extienden — a tus hijos, tus vecinos, tu mundo.',
          fr: "Bonjour ! 🌍 En tant que personne qui construit probablement une famille ou qui dirige dans sa communauté, la citoyenneté prend un sens plus profond. Aujourd'hui, explorons comment tes actions se propagent — vers tes enfants, tes voisins, ton monde."
        },
        '61+': {
          en: "Welcome, friend. 🕊️ You've seen communities change, countries transform. Today we're reflecting on citizenship through the lens of wisdom — what it meant, what it means now, and the legacy we leave behind.",
          es: 'Bienvenido, amigo. 🕊️ Has visto comunidades cambiar, países transformarse. Hoy reflexionamos sobre la ciudadanía a través del lente de la sabiduría — qué significaba, qué significa ahora, y el legado que dejamos.',
          fr: "Bienvenue, ami. 🕊️ Tu as vu des communautés changer, des pays se transformer. Aujourd'hui, nous réfléchissons à la citoyenneté à travers le prisme de la sagesse — ce qu'elle signifiait, ce qu'elle signifie maintenant, et l'héritage que nous laissons."
        }
      },

      hint: null,
      choices: null
    },

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: Q1 - What is citizenship?
    // ═══════════════════════════════════════════════════════════════════
    {
      order: 2,
      type: 'question',
      name: 'Q1',
      expression: 'explaining',

      text: {
        '2-5': {
          en: "When you share your toys with a friend, you're being a good... what? 🤔",
          es: 'Cuando compartes tus juguetes con un amigo, estás siendo un buen... ¿qué? 🤔',
          fr: 'Quand tu partages tes jouets avec un ami, tu es un bon... quoi ? 🤔'
        },
        '6-12': {
          en: 'What do you think makes someone a REAL citizen? Not just on paper — but in their actions? 🦸',
          es: '¿Qué crees que hace a alguien un ciudadano DE VERDAD? No solo en papel — ¡sino en sus acciones! 🦸',
          fr: "Qu'est-ce qui fait de quelqu'un un VRAI citoyen selon toi ? Pas juste sur le papier — mais dans ses actions ! 🦸"
        },
        '13-17': {
          en: 'Real talk: What does citizenship mean to YOU? Is it about papers, or something bigger? 🎯',
          es: 'Hablemos en serio: ¿Qué significa la ciudadanía PARA TI? ¿Se trata de papeles o de algo más grande? 🎯',
          fr: "Parlons vrai : Que signifie la citoyenneté POUR TOI ? C'est une question de papiers, ou quelque chose de plus grand ? 🎯"
        },
        '18-35': {
          en: 'In your daily life, how do you define active citizenship? Is it political engagement, or something more personal? 🏛️',
          es: 'En tu vida diaria, ¿cómo defines la ciudadanía activa? ¿Es compromiso político o algo más personal? 🏛️',
          fr: 'Dans ta vie quotidienne, comment définis-tu la citoyenneté active ? Est-ce un engagement politique, ou quelque chose de plus personnel ? 🏛️'
        },
        '36-60': {
          en: 'How has your understanding of citizenship evolved over the years? What matters most now? 🌱',
          es: '¿Cómo ha evolucionado tu comprensión de la ciudadanía a lo largo de los años? ¿Qué importa más ahora? 🌱',
          fr: "Comment ta compréhension de la citoyenneté a-t-elle évolué au fil des années ? Qu'est-ce qui compte le plus maintenant ? 🌱"
        },
        '61+': {
          en: 'Looking back, what acts of citizenship are you most proud of? What would you tell younger generations? 🕊️',
          es: 'Mirando atrás, ¿de qué actos de ciudadanía te sientes más orgulloso? ¿Qué le dirías a las generaciones más jóvenes? 🕊️',
          fr: 'En regardant en arrière, de quels actes de citoyenneté es-tu le plus fier ? Que dirais-tu aux jeunes générations ? 🕊️'
        }
      },

      hint: {
        '2-5': {
          en: 'Think about being kind...',
          es: 'Piensa en ser amable...',
          fr: 'Pense à être gentil...'
        },
        '6-12': {
          en: 'Think about your school and neighborhood...',
          es: 'Piensa en tu escuela y vecindario...',
          fr: 'Pense à ton école et ton quartier...'
        },
        '13-17': {
          en: 'Think beyond just voting...',
          es: 'Piensa más allá de solo votar...',
          fr: 'Pense au-delà du simple vote...'
        },
        '18-35': {
          en: 'Consider your daily impact...',
          es: 'Considera tu impacto diario...',
          fr: 'Considère ton impact quotidien...'
        },
        '36-60': {
          en: 'Think about legacy...',
          es: 'Piensa en el legado...',
          fr: "Pense à l'héritage..."
        },
        '61+': {
          en: 'Reflect on your journey...',
          es: 'Reflexiona sobre tu camino...',
          fr: 'Réfléchis à ton parcours...'
        }
      },

      choices: {
        '2-5': {
          en: [
            {
              letter: 'A',
              text: 'A good helper!',
              response:
                "Yes! Helpers are SO important! And you know what? When you help, you're being a good citizen too! 🌟"
            },
            {
              letter: 'B',
              text: 'A good friend and neighbor!',
              response:
                "Exactly right! 🎉 When you're a good friend and neighbor, you're being the BEST kind of citizen! That's what makes communities happy!"
            },
            {
              letter: 'C',
              text: 'A good person who cares!',
              response:
                'Wow, you really get it! 💫 Caring about others IS what being a citizen is all about!'
            }
          ],
          es: [
            {
              letter: 'A',
              text: '¡Un buen ayudante!',
              response:
                '¡Sí! ¡Los ayudantes son MUY importantes! ¿Y sabes qué? ¡Cuando ayudas, también estás siendo un buen ciudadano! 🌟'
            },
            {
              letter: 'B',
              text: '¡Un buen amigo y vecino!',
              response:
                '¡Exactamente! 🎉 ¡Cuando eres un buen amigo y vecino, estás siendo el MEJOR tipo de ciudadano! ¡Eso es lo que hace felices a las comunidades!'
            },
            {
              letter: 'C',
              text: '¡Una buena persona que se preocupa!',
              response:
                '¡Guau, realmente lo entiendes! 💫 ¡Preocuparse por los demás ES de lo que se trata ser ciudadano!'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: 'Un bon aide !',
              response:
                'Oui ! Les aides sont TELLEMENT importants ! Et tu sais quoi ? Quand tu aides, tu es aussi un bon citoyen ! 🌟'
            },
            {
              letter: 'B',
              text: 'Un bon ami et voisin !',
              response:
                "Exactement ! 🎉 Quand tu es un bon ami et voisin, tu es le MEILLEUR type de citoyen ! C'est ce qui rend les communautés heureuses !"
            },
            {
              letter: 'C',
              text: 'Une bonne personne qui se soucie !',
              response:
                "Wow, tu comprends vraiment ! 💫 Se soucier des autres, C'EST ce que signifie être citoyen !"
            }
          ]
        },
        '6-12': {
          en: [
            {
              letter: 'A',
              text: 'Someone born in a country',
              response:
                "That's part of it! But here's a secret: you can be a citizen of your school, your team, your neighborhood — even if you weren't born there! 🌟"
            },
            {
              letter: 'B',
              text: 'Someone who helps make their community better',
              response:
                "BOOM! 💥 You nailed it! A real citizen isn't just someone with papers — it's someone who ACTS to make things better. You're already one!"
            },
            {
              letter: 'C',
              text: 'Both! Being part of a place AND helping it',
              response:
                "Whoa, big brain thinking! 🧠 You're absolutely right — it's both where you belong AND what you DO. That's citizenship level 100!"
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Alguien nacido en un país',
              response:
                '¡Eso es parte de ello! Pero aquí hay un secreto: puedes ser ciudadano de tu escuela, tu equipo, tu vecindario — ¡aunque no hayas nacido allí! 🌟'
            },
            {
              letter: 'B',
              text: 'Alguien que ayuda a mejorar su comunidad',
              response:
                '¡BOOM! 💥 ¡Lo clavaste! Un ciudadano real no es solo alguien con papeles — es alguien que ACTÚA para mejorar las cosas. ¡Ya eres uno!'
            },
            {
              letter: 'C',
              text: '¡Ambos! Ser parte de un lugar Y ayudarlo',
              response:
                '¡Guau, pensamiento de gran cerebro! 🧠 Tienes toda la razón — es tanto dónde perteneces COMO lo que HACES. ¡Eso es ciudadanía nivel 100!'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: "Quelqu'un né dans un pays",
              response:
                "C'est une partie ! Mais voici un secret : tu peux être citoyen de ton école, ton équipe, ton quartier — même si tu n'y es pas né ! 🌟"
            },
            {
              letter: 'B',
              text: "Quelqu'un qui aide à améliorer sa communauté",
              response:
                "BOOM ! 💥 Tu as tout compris ! Un vrai citoyen n'est pas juste quelqu'un avec des papiers — c'est quelqu'un qui AGIT pour améliorer les choses. Tu en es déjà un !"
            },
            {
              letter: 'C',
              text: "Les deux ! Faire partie d'un endroit ET l'aider",
              response:
                "Wow, réflexion de génie ! 🧠 Tu as absolument raison — c'est à la fois où tu appartiens ET ce que tu FAIS. C'est la citoyenneté niveau 100 !"
            }
          ]
        },
        '13-17': {
          en: [
            {
              letter: 'A',
              text: 'Having legal status and rights',
              response:
                "True, that's the legal definition. But here's the thing — citizenship is also a mindset. You can be legally a citizen but not really engaged. Let's dig deeper..."
            },
            {
              letter: 'B',
              text: 'Being engaged and making a difference',
              response:
                "THIS. 🎯 You get it. Citizenship isn't just a status — it's action. Every time you speak up, help out, or stand for something, you're being a citizen. Papers or not."
            },
            {
              letter: 'C',
              text: 'A mix of rights, responsibilities, and identity',
              response:
                "That's a sophisticated take. 🧠 You're seeing the full picture — citizenship is legal, social, AND personal. It's who you are AND what you do. Respect."
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Tener estatus legal y derechos',
              response:
                'Cierto, esa es la definición legal. Pero aquí está la cosa — la ciudadanía también es una mentalidad. Puedes ser legalmente ciudadano pero no estar realmente comprometido. Profundicemos...'
            },
            {
              letter: 'B',
              text: 'Estar comprometido y hacer la diferencia',
              response:
                'ESTO. 🎯 Lo entiendes. La ciudadanía no es solo un estatus — es acción. Cada vez que hablas, ayudas o defiendes algo, estás siendo ciudadano. Con o sin papeles.'
            },
            {
              letter: 'C',
              text: 'Una mezcla de derechos, responsabilidades e identidad',
              response:
                'Esa es una perspectiva sofisticada. 🧠 Estás viendo el panorama completo — la ciudadanía es legal, social Y personal. Es quién eres Y lo que haces. Respeto.'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: 'Avoir un statut légal et des droits',
              response:
                "Vrai, c'est la définition légale. Mais voilà — la citoyenneté est aussi un état d'esprit. Tu peux être légalement citoyen mais pas vraiment engagé. Creusons plus..."
            },
            {
              letter: 'B',
              text: 'Être engagé et faire une différence',
              response:
                "ÇA. 🎯 Tu comprends. La citoyenneté n'est pas juste un statut — c'est l'action. Chaque fois que tu t'exprimes, aides ou défends quelque chose, tu es citoyen. Papiers ou pas."
            },
            {
              letter: 'C',
              text: 'Un mélange de droits, responsabilités et identité',
              response:
                "C'est une vision sophistiquée. 🧠 Tu vois l'image complète — la citoyenneté est légale, sociale ET personnelle. C'est qui tu es ET ce que tu fais. Respect."
            }
          ]
        },
        '18-35': {
          en: [
            {
              letter: 'A',
              text: 'Primarily about voting and civic duties',
              response:
                'Voting matters, absolutely. But citizenship extends to your workplace, your apartment building, your online communities. Every space you occupy. Think bigger...'
            },
            {
              letter: 'B',
              text: 'Daily actions that strengthen community bonds',
              response:
                "Exactly. 🎯 Citizenship isn't just election day — it's every day. It's how you treat your barista, your neighbor, your colleague. The small moments add up to big change."
            },
            {
              letter: 'C',
              text: 'A balance of rights, responsibilities, and social capital',
              response:
                "That's a nuanced view. 💡 You're seeing citizenship as an ecosystem — what you take, what you give, and the invisible bonds that hold it together. That's leadership thinking."
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Principalmente sobre votar y deberes cívicos',
              response:
                'Votar importa, absolutamente. Pero la ciudadanía se extiende a tu lugar de trabajo, tu edificio de apartamentos, tus comunidades en línea. Cada espacio que ocupas. Piensa más grande...'
            },
            {
              letter: 'B',
              text: 'Acciones diarias que fortalecen los lazos comunitarios',
              response:
                'Exactamente. 🎯 La ciudadanía no es solo el día de elecciones — es cada día. Es cómo tratas a tu barista, tu vecino, tu colega. Los pequeños momentos suman grandes cambios.'
            },
            {
              letter: 'C',
              text: 'Un equilibrio de derechos, responsabilidades y capital social',
              response:
                'Esa es una visión matizada. 💡 Estás viendo la ciudadanía como un ecosistema — lo que tomas, lo que das, y los lazos invisibles que lo mantienen unido. Eso es pensamiento de liderazgo.'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: 'Principalement voter et devoirs civiques',
              response:
                "Voter compte, absolument. Mais la citoyenneté s'étend à ton lieu de travail, ton immeuble, tes communautés en ligne. Chaque espace que tu occupes. Pense plus grand..."
            },
            {
              letter: 'B',
              text: 'Des actions quotidiennes qui renforcent les liens',
              response:
                "Exactement. 🎯 La citoyenneté n'est pas que le jour des élections — c'est chaque jour. C'est comment tu traites ton barista, ton voisin, ton collègue. Les petits moments créent de grands changements."
            },
            {
              letter: 'C',
              text: 'Un équilibre de droits, responsabilités et capital social',
              response:
                "C'est une vision nuancée. 💡 Tu vois la citoyenneté comme un écosystème — ce que tu prends, ce que tu donnes, et les liens invisibles qui tiennent le tout. C'est une pensée de leader."
            }
          ]
        },
        '36-60': {
          en: [
            {
              letter: 'A',
              text: 'Modeling good behavior for the next generation',
              response:
                "Powerful perspective. The example you set for your children, your employees, your neighbors — that's citizenship in action. They're watching and learning."
            },
            {
              letter: 'B',
              text: 'Building institutions and systems that outlast us',
              response:
                "That's thinking in generations. 🌳 True citizenship is planting trees whose shade you'll never sit in. The schools you support, the policies you advocate for — that's your legacy."
            },
            {
              letter: 'C',
              text: 'Both personal example AND systemic contribution',
              response:
                "You're operating on multiple levels. 🎯 Personal influence through daily actions AND systemic impact through institutions. That's the full expression of citizenship. Well played."
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Modelar buen comportamiento para la próxima generación',
              response:
                'Perspectiva poderosa. El ejemplo que das a tus hijos, tus empleados, tus vecinos — eso es ciudadanía en acción. Están mirando y aprendiendo.'
            },
            {
              letter: 'B',
              text: 'Construir instituciones y sistemas que nos sobrevivan',
              response:
                'Eso es pensar en generaciones. 🌳 La verdadera ciudadanía es plantar árboles cuya sombra nunca disfrutarás. Las escuelas que apoyas, las políticas que defiendes — ese es tu legado.'
            },
            {
              letter: 'C',
              text: 'Tanto ejemplo personal COMO contribución sistémica',
              response:
                'Estás operando en múltiples niveles. 🎯 Influencia personal a través de acciones diarias Y impacto sistémico a través de instituciones. Esa es la expresión completa de la ciudadanía. Bien jugado.'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: "Montrer l'exemple pour la prochaine génération",
              response:
                "Perspective puissante. L'exemple que tu donnes à tes enfants, tes employés, tes voisins — c'est la citoyenneté en action. Ils regardent et apprennent."
            },
            {
              letter: 'B',
              text: 'Construire des institutions qui nous survivront',
              response:
                "C'est penser en générations. 🌳 La vraie citoyenneté, c'est planter des arbres dont tu ne profiteras jamais de l'ombre. Les écoles que tu soutiens, les politiques que tu défends — c'est ton héritage."
            },
            {
              letter: 'C',
              text: "L'exemple personnel ET la contribution systémique",
              response:
                "Tu opères sur plusieurs niveaux. 🎯 Influence personnelle par les actions quotidiennes ET impact systémique par les institutions. C'est l'expression complète de la citoyenneté. Bien joué."
            }
          ]
        },
        '61+': {
          en: [
            {
              letter: 'A',
              text: 'Passing down values and stories',
              response:
                "The oral tradition. 📖 Your stories carry lessons that no textbook can teach. When you share your experiences, you're giving the next generation a compass."
            },
            {
              letter: 'B',
              text: 'Staying engaged and relevant in changing times',
              response:
                "Adaptability is wisdom. 🌊 You've seen so much change, yet you're here, still learning, still growing. That commitment to engagement IS citizenship."
            },
            {
              letter: 'C',
              text: 'Being a bridge between past wisdom and future hope',
              response:
                "You're a living bridge. 🌉 You hold the lessons of history AND the hope for what's next. That perspective is irreplaceable. The world needs your voice."
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Transmitir valores e historias',
              response:
                'La tradición oral. 📖 Tus historias llevan lecciones que ningún libro de texto puede enseñar. Cuando compartes tus experiencias, le das a la próxima generación una brújula.'
            },
            {
              letter: 'B',
              text: 'Mantenerse comprometido y relevante en tiempos cambiantes',
              response:
                'La adaptabilidad es sabiduría. 🌊 Has visto tanto cambio, y sin embargo estás aquí, todavía aprendiendo, todavía creciendo. Ese compromiso con el compromiso ES ciudadanía.'
            },
            {
              letter: 'C',
              text: 'Ser un puente entre la sabiduría del pasado y la esperanza del futuro',
              response:
                'Eres un puente viviente. 🌉 Tienes las lecciones de la historia Y la esperanza de lo que viene. Esa perspectiva es irremplazable. El mundo necesita tu voz.'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: 'Transmettre des valeurs et des histoires',
              response:
                "La tradition orale. 📖 Tes histoires portent des leçons qu'aucun manuel ne peut enseigner. Quand tu partages tes expériences, tu donnes une boussole à la prochaine génération."
            },
            {
              letter: 'B',
              text: 'Rester engagé et pertinent dans des temps changeants',
              response:
                "L'adaptabilité est sagesse. 🌊 Tu as vu tant de changements, et pourtant tu es là, toujours en train d'apprendre, de grandir. Cet engagement EST la citoyenneté."
            },
            {
              letter: 'C',
              text: "Être un pont entre la sagesse passée et l'espoir futur",
              response:
                "Tu es un pont vivant. 🌉 Tu détiens les leçons de l'histoire ET l'espoir de ce qui vient. Cette perspective est irremplaçable. Le monde a besoin de ta voix."
            }
          ]
        }
      }
    },

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 3: Q2 - Local vs Global
    // ═══════════════════════════════════════════════════════════════════
    {
      order: 3,
      type: 'question',
      name: 'Q2',
      expression: 'explaining',

      text: {
        '2-5': {
          en: 'You can be nice to your teddy bear, your friend, AND everyone in the world! Where do you want to start? 🌍',
          es: '¡Puedes ser amable con tu osito de peluche, tu amigo Y con todos en el mundo! ¿Dónde quieres empezar? 🌍',
          fr: 'Tu peux être gentil avec ton nounours, ton ami ET tout le monde dans le monde ! Par où veux-tu commencer ? 🌍'
        },
        '6-12': {
          en: "Here's a puzzle: Should you focus on helping your neighborhood OR helping the whole world? 🌎",
          es: 'Aquí hay un rompecabezas: ¿Deberías enfocarte en ayudar a tu vecindario O ayudar al mundo entero? 🌎',
          fr: 'Voici une énigme : Devrais-tu te concentrer sur aider ton quartier OU aider le monde entier ? 🌎'
        },
        '13-17': {
          en: 'Social media connects us globally, but your school is local. Where does citizenship matter more? 🌐',
          es: 'Las redes sociales nos conectan globalmente, pero tu escuela es local. ¿Dónde importa más la ciudadanía? 🌐',
          fr: 'Les réseaux sociaux nous connectent mondialement, mais ton école est locale. Où la citoyenneté compte-t-elle plus ? 🌐'
        },
        '18-35': {
          en: 'You work in a global economy but live in a local community. Where do you direct your citizenship energy? 🏙️',
          es: 'Trabajas en una economía global pero vives en una comunidad local. ¿Dónde diriges tu energía ciudadana? 🏙️',
          fr: 'Tu travailles dans une économie mondiale mais vis dans une communauté locale. Où diriges-tu ton énergie citoyenne ? 🏙️'
        },
        '36-60': {
          en: 'Your children will inherit both local communities and global challenges. How do you prepare them? 👨‍👩‍👧‍👦',
          es: 'Tus hijos heredarán tanto comunidades locales como desafíos globales. ¿Cómo los preparas? 👨‍👩‍👧‍👦',
          fr: 'Tes enfants hériteront des communautés locales et des défis mondiaux. Comment les prépares-tu ? 👨‍👩‍👧‍👦'
        },
        '61+': {
          en: "You've witnessed both world events and neighborhood changes. What matters more for lasting impact? 🌍",
          es: 'Has presenciado tanto eventos mundiales como cambios en el vecindario. ¿Qué importa más para un impacto duradero? 🌍',
          fr: "Tu as été témoin d'événements mondiaux et de changements de quartier. Qu'est-ce qui compte plus pour un impact durable ? 🌍"
        }
      },

      hint: {
        '2-5': {
          en: 'All kindness counts!',
          es: '¡Toda amabilidad cuenta!',
          fr: 'Toute gentillesse compte !'
        },
        '6-12': { en: 'Maybe both?', es: '¿Quizás ambos?', fr: 'Peut-être les deux ?' },
        '13-17': {
          en: 'Think about ripple effects...',
          es: 'Piensa en los efectos dominó...',
          fr: "Pense aux effets d'onde..."
        },
        '18-35': {
          en: 'Consider your spheres of influence...',
          es: 'Considera tus esferas de influencia...',
          fr: "Considère tes sphères d'influence..."
        },
        '36-60': {
          en: 'What examples are you setting?',
          es: '¿Qué ejemplos estás dando?',
          fr: 'Quels exemples donnes-tu ?'
        },
        '61+': {
          en: 'What has your experience taught you?',
          es: '¿Qué te ha enseñado tu experiencia?',
          fr: "Que t'a appris ton expérience ?"
        }
      },

      choices: {
        '2-5': {
          en: [
            {
              letter: 'A',
              text: 'Start with my teddy bear!',
              response:
                'Aww! 🧸 Being kind to your teddy is practice for being kind to everyone! Start small, love grows!'
            },
            {
              letter: 'B',
              text: 'Start with my friends and family!',
              response:
                "Perfect! 💕 When you're kind at home and with friends, that kindness spreads out like ripples in a pond!"
            },
            {
              letter: 'C',
              text: 'Be nice to EVERYONE!',
              response:
                'What a big heart you have! 🌈 Yes! Every living thing deserves kindness — starting with you!'
            }
          ],
          es: [
            {
              letter: 'A',
              text: '¡Empezar con mi osito!',
              response:
                '¡Aww! 🧸 ¡Ser amable con tu osito es práctica para ser amable con todos! ¡Empieza pequeño, el amor crece!'
            },
            {
              letter: 'B',
              text: '¡Empezar con mis amigos y familia!',
              response:
                '¡Perfecto! 💕 ¡Cuando eres amable en casa y con amigos, esa amabilidad se extiende como ondas en un estanque!'
            },
            {
              letter: 'C',
              text: '¡Ser amable con TODOS!',
              response:
                '¡Qué corazón tan grande tienes! 🌈 ¡Sí! Todo ser viviente merece amabilidad — ¡empezando por ti!'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: 'Commencer avec mon nounours !',
              response:
                "Aww ! 🧸 Être gentil avec ton nounours, c'est s'entraîner à être gentil avec tout le monde ! Commence petit, l'amour grandit !"
            },
            {
              letter: 'B',
              text: 'Commencer avec mes amis et ma famille !',
              response:
                'Parfait ! 💕 Quand tu es gentil à la maison et avec tes amis, cette gentillesse se répand comme des ondes dans un étang !'
            },
            {
              letter: 'C',
              text: 'Être gentil avec TOUT LE MONDE !',
              response:
                'Quel grand cœur tu as ! 🌈 Oui ! Tout être vivant mérite de la gentillesse — en commençant par toi !'
            }
          ]
        },
        '6-12': {
          en: [
            {
              letter: 'A',
              text: 'Neighborhood first! Change starts nearby',
              response:
                'Smart thinking! 🏘️ Your neighborhood is like your training ground. Every local hero becomes a global hero!'
            },
            {
              letter: 'B',
              text: "The whole world! We're all connected",
              response:
                "Big picture thinker! 🌍 You're right that we're all connected. But guess what? Your global impact starts with local action!"
            },
            {
              letter: 'C',
              text: 'BOTH! Local actions have global effects',
              response:
                "You cracked the code! 🔓 When you help someone next door, you're helping the whole world one person at a time. That's the secret!"
            }
          ],
          es: [
            {
              letter: 'A',
              text: '¡Primero el vecindario! El cambio empieza cerca',
              response:
                '¡Pensamiento inteligente! 🏘️ Tu vecindario es como tu campo de entrenamiento. ¡Todo héroe local se convierte en héroe global!'
            },
            {
              letter: 'B',
              text: '¡El mundo entero! Todos estamos conectados',
              response:
                '¡Pensador de panorama general! 🌍 Tienes razón en que todos estamos conectados. ¡Pero adivina qué! ¡Tu impacto global comienza con acción local!'
            },
            {
              letter: 'C',
              text: '¡AMBOS! Las acciones locales tienen efectos globales',
              response:
                '¡Descifraste el código! 🔓 Cuando ayudas a alguien al lado, estás ayudando al mundo entero una persona a la vez. ¡Ese es el secreto!'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: "Le quartier d'abord ! Le changement commence près",
              response:
                "Réflexion intelligente ! 🏘️ Ton quartier est comme ton terrain d'entraînement. Chaque héros local devient un héros mondial !"
            },
            {
              letter: 'B',
              text: 'Le monde entier ! On est tous connectés',
              response:
                "Penseur de grande envergure ! 🌍 Tu as raison qu'on est tous connectés. Mais devine quoi ? Ton impact mondial commence par l'action locale !"
            },
            {
              letter: 'C',
              text: 'LES DEUX ! Les actions locales ont des effets mondiaux',
              response:
                "Tu as trouvé le code ! 🔓 Quand tu aides quelqu'un à côté, tu aides le monde entier une personne à la fois. C'est le secret !"
            }
          ]
        },
        '13-17': {
          en: [
            {
              letter: 'A',
              text: 'Local — real change happens face-to-face',
              response:
                "There's truth here. Digital activism is easy; showing up is hard. The relationships you build locally are your foundation."
            },
            {
              letter: 'B',
              text: "Global — we can't ignore worldwide issues",
              response:
                "You're right that global problems need attention. But here's the thing: your voice online is more powerful when you're also active offline."
            },
            {
              letter: 'C',
              text: 'Both — local action, global awareness',
              response:
                "That's the move. 🎯 Think globally, act locally. Your TikTok reach + your neighborhood presence = maximum impact."
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Local — el cambio real pasa cara a cara',
              response:
                'Hay verdad aquí. El activismo digital es fácil; presentarse es difícil. Las relaciones que construyes localmente son tu base.'
            },
            {
              letter: 'B',
              text: 'Global — no podemos ignorar problemas mundiales',
              response:
                'Tienes razón en que los problemas globales necesitan atención. Pero aquí está la cosa: tu voz en línea es más poderosa cuando también estás activo fuera de línea.'
            },
            {
              letter: 'C',
              text: 'Ambos — acción local, conciencia global',
              response:
                'Ese es el movimiento. 🎯 Piensa globalmente, actúa localmente. Tu alcance en TikTok + tu presencia en el vecindario = impacto máximo.'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: 'Local — le vrai changement se fait en face à face',
              response:
                "Il y a du vrai ici. L'activisme numérique est facile ; se présenter est difficile. Les relations que tu construis localement sont ta base."
            },
            {
              letter: 'B',
              text: 'Mondial — on ne peut pas ignorer les problèmes mondiaux',
              response:
                "Tu as raison que les problèmes mondiaux ont besoin d'attention. Mais voilà : ta voix en ligne est plus puissante quand tu es aussi actif hors ligne."
            },
            {
              letter: 'C',
              text: 'Les deux — action locale, conscience mondiale',
              response:
                "C'est le bon mouvement. 🎯 Pense mondialement, agis localement. Ta portée TikTok + ta présence de quartier = impact maximum."
            }
          ]
        },
        '18-35': {
          en: [
            {
              letter: 'A',
              text: 'Local — my immediate community needs me',
              response:
                'Your apartment building, your office, your coffee shop — these are your first communities. Strengthen those, and you build outward.'
            },
            {
              letter: 'B',
              text: 'Global — my work has international impact',
              response:
                'If your work crosses borders, so does your responsibility. But remember: your global perspective is richer when rooted locally.'
            },
            {
              letter: 'C',
              text: 'Strategic balance based on my skills',
              response:
                "Now we're talking strategy. 🎯 Use your professional skills globally AND your human skills locally. That's how you maximize both."
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Local — mi comunidad inmediata me necesita',
              response:
                'Tu edificio de apartamentos, tu oficina, tu cafetería — estas son tus primeras comunidades. Fortalécelas, y construyes hacia afuera.'
            },
            {
              letter: 'B',
              text: 'Global — mi trabajo tiene impacto internacional',
              response:
                'Si tu trabajo cruza fronteras, también tu responsabilidad. Pero recuerda: tu perspectiva global es más rica cuando está arraigada localmente.'
            },
            {
              letter: 'C',
              text: 'Balance estratégico basado en mis habilidades',
              response:
                'Ahora estamos hablando de estrategia. 🎯 Usa tus habilidades profesionales globalmente Y tus habilidades humanas localmente. Así maximizas ambos.'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: 'Local — ma communauté immédiate a besoin de moi',
              response:
                "Ton immeuble, ton bureau, ton café — ce sont tes premières communautés. Renforce-les, et tu construis vers l'extérieur."
            },
            {
              letter: 'B',
              text: 'Mondial — mon travail a un impact international',
              response:
                'Si ton travail traverse les frontières, ta responsabilité aussi. Mais rappelle-toi : ta perspective mondiale est plus riche quand elle est enracinée localement.'
            },
            {
              letter: 'C',
              text: 'Équilibre stratégique basé sur mes compétences',
              response:
                "Là on parle stratégie. 🎯 Utilise tes compétences professionnelles mondialement ET tes compétences humaines localement. C'est ainsi que tu maximises les deux."
            }
          ]
        },
        '36-60': {
          en: [
            {
              letter: 'A',
              text: 'Focus locally — build strong community roots',
              response:
                'Roots before branches. 🌳 Your children learn citizenship by watching you at PTA meetings, neighborhood cleanups, local elections.'
            },
            {
              letter: 'B',
              text: 'Think globally — prepare them for a connected world',
              response:
                "The world they'll inherit is global. Teaching them to think beyond borders prepares them for challenges we can't yet imagine."
            },
            {
              letter: 'C',
              text: 'Model both — show them connection at every scale',
              response:
                "You're showing them the full picture. 🌐 Local roots give them stability; global awareness gives them perspective. Both together = complete citizens."
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Enfocarse localmente — construir raíces comunitarias fuertes',
              response:
                'Raíces antes que ramas. 🌳 Tus hijos aprenden ciudadanía viéndote en reuniones de padres, limpiezas del vecindario, elecciones locales.'
            },
            {
              letter: 'B',
              text: 'Pensar globalmente — prepararlos para un mundo conectado',
              response:
                'El mundo que heredarán es global. Enseñarles a pensar más allá de las fronteras los prepara para desafíos que aún no podemos imaginar.'
            },
            {
              letter: 'C',
              text: 'Modelar ambos — mostrarles conexión en cada escala',
              response:
                'Les estás mostrando el panorama completo. 🌐 Las raíces locales les dan estabilidad; la conciencia global les da perspectiva. Ambos juntos = ciudadanos completos.'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: 'Se concentrer localement — construire des racines communautaires fortes',
              response:
                'Les racines avant les branches. 🌳 Tes enfants apprennent la citoyenneté en te regardant aux réunions de parents, nettoyages de quartier, élections locales.'
            },
            {
              letter: 'B',
              text: 'Penser mondialement — les préparer pour un monde connecté',
              response:
                "Le monde qu'ils hériteront est mondial. Leur apprendre à penser au-delà des frontières les prépare pour des défis qu'on ne peut pas encore imaginer."
            },
            {
              letter: 'C',
              text: 'Modeler les deux — leur montrer la connexion à chaque échelle',
              response:
                "Tu leur montres l'image complète. 🌐 Les racines locales leur donnent la stabilité ; la conscience mondiale leur donne la perspective. Les deux ensemble = citoyens complets."
            }
          ]
        },
        '61+': {
          en: [
            {
              letter: 'A',
              text: 'Local — lasting change comes from community',
              response:
                'Your experience confirms this. The neighbors who helped, the local institutions that held — those are what endured through all the changes.'
            },
            {
              letter: 'B',
              text: "Global — we're more connected than ever",
              response:
                "You've seen the world shrink. Television, internet, global travel — each connected us more. That connection brings responsibility."
            },
            {
              letter: 'C',
              text: 'The local IS global now',
              response:
                "Wisdom speaking. 🌟 You've watched local become global. Your neighborhood story is now everyone's story. That perspective is rare and valuable."
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Local — el cambio duradero viene de la comunidad',
              response:
                'Tu experiencia confirma esto. Los vecinos que ayudaron, las instituciones locales que resistieron — eso es lo que perduró a través de todos los cambios.'
            },
            {
              letter: 'B',
              text: 'Global — estamos más conectados que nunca',
              response:
                'Has visto el mundo encogerse. Televisión, internet, viajes globales — cada uno nos conectó más. Esa conexión trae responsabilidad.'
            },
            {
              letter: 'C',
              text: 'Lo local ES global ahora',
              response:
                'Sabiduría hablando. 🌟 Has visto lo local volverse global. Tu historia del vecindario es ahora la historia de todos. Esa perspectiva es rara y valiosa.'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: 'Local — le changement durable vient de la communauté',
              response:
                "Ton expérience le confirme. Les voisins qui ont aidé, les institutions locales qui ont tenu — c'est ce qui a duré à travers tous les changements."
            },
            {
              letter: 'B',
              text: 'Mondial — nous sommes plus connectés que jamais',
              response:
                'Tu as vu le monde rétrécir. Télévision, internet, voyages mondiaux — chacun nous a plus connectés. Cette connexion apporte des responsabilités.'
            },
            {
              letter: 'C',
              text: 'Le local EST mondial maintenant',
              response:
                "La sagesse parle. 🌟 Tu as vu le local devenir mondial. L'histoire de ton quartier est maintenant l'histoire de tous. Cette perspective est rare et précieuse."
            }
          ]
        }
      }
    },

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 4: Q3 - What can YOU do?
    // ═══════════════════════════════════════════════════════════════════
    {
      order: 4,
      type: 'question',
      name: 'Q3',
      expression: 'listening',

      text: {
        '2-5': {
          en: "You're amazing! What's ONE nice thing you can do today for someone else? 💝",
          es: '¡Eres increíble! ¿Cuál es UNA cosa buena que puedes hacer hoy por otra persona? 💝',
          fr: "Tu es incroyable ! Quelle est UNE chose gentille que tu peux faire aujourd'hui pour quelqu'un d'autre ? 💝"
        },
        '6-12': {
          en: "You have citizen superpowers! What's one way you'll use them THIS WEEK? 🦸",
          es: '¡Tienes superpoderes de ciudadano! ¿Cuál es una forma en que los usarás ESTA SEMANA? 🦸',
          fr: 'Tu as des super-pouvoirs de citoyen ! Comment vas-tu les utiliser CETTE SEMAINE ? 🦸'
        },
        '13-17': {
          en: "Real talk: What's one concrete action you'll take to be a better citizen? 📝",
          es: 'Hablemos en serio: ¿Cuál es una acción concreta que tomarás para ser mejor ciudadano? 📝',
          fr: 'Parlons vrai : Quelle est une action concrète que tu vas prendre pour être un meilleur citoyen ? 📝'
        },
        '18-35': {
          en: 'Time to commit: What specific citizenship action fits your schedule and skills? 🎯',
          es: 'Hora de comprometerse: ¿Qué acción específica de ciudadanía encaja con tu horario y habilidades? 🎯',
          fr: "C'est l'heure de s'engager : Quelle action citoyenne spécifique correspond à ton emploi du temps et tes compétences ? 🎯"
        },
        '36-60': {
          en: 'With your experience and resources, what legacy-building action will you take? 🏗️',
          es: 'Con tu experiencia y recursos, ¿qué acción de construcción de legado tomarás? 🏗️',
          fr: "Avec ton expérience et tes ressources, quelle action de construction d'héritage vas-tu entreprendre ? 🏗️"
        },
        '61+': {
          en: 'Your wisdom is needed. How will you share it with those coming after you? 📖',
          es: 'Tu sabiduría es necesaria. ¿Cómo la compartirás con los que vienen después de ti? 📖',
          fr: 'Ta sagesse est nécessaire. Comment vas-tu la partager avec ceux qui viennent après toi ? 📖'
        }
      },

      hint: {
        '2-5': {
          en: 'Even small things count!',
          es: '¡Hasta las cosas pequeñas cuentan!',
          fr: 'Même les petites choses comptent !'
        },
        '6-12': {
          en: "Think about what you're good at...",
          es: 'Piensa en lo que eres bueno...',
          fr: 'Pense à ce dans quoi tu es bon...'
        },
        '13-17': {
          en: 'Be specific, not vague...',
          es: 'Sé específico, no vago...',
          fr: 'Sois spécifique, pas vague...'
        },
        '18-35': {
          en: 'Play to your strengths...',
          es: 'Juega con tus fortalezas...',
          fr: 'Joue sur tes forces...'
        },
        '36-60': {
          en: 'What will last beyond you?',
          es: '¿Qué perdurará más allá de ti?',
          fr: "Qu'est-ce qui durera au-delà de toi ?"
        },
        '61+': {
          en: 'Your stories matter...',
          es: 'Tus historias importan...',
          fr: 'Tes histoires comptent...'
        }
      },

      choices: {
        '2-5': {
          en: [
            {
              letter: 'A',
              text: 'Give someone a hug!',
              response:
                "Hugs are MAGIC! 🤗 They make you AND the other person feel happy. That's being a good citizen of your family!"
            },
            {
              letter: 'B',
              text: 'Share my toys with a friend!',
              response:
                "Sharing is caring! 🧸 When you share, you're telling your friend 'you matter to me.' That's beautiful!"
            },
            {
              letter: 'C',
              text: 'Help clean up without being asked!',
              response:
                "WOW! 🌟 Helping without being asked? That's like having secret superhero powers! Your grown-ups will be SO proud!"
            }
          ],
          es: [
            {
              letter: 'A',
              text: '¡Darle un abrazo a alguien!',
              response:
                '¡Los abrazos son MAGIA! 🤗 Hacen que tú Y la otra persona se sientan felices. ¡Eso es ser un buen ciudadano de tu familia!'
            },
            {
              letter: 'B',
              text: '¡Compartir mis juguetes con un amigo!',
              response:
                "¡Compartir es cuidar! 🧸 Cuando compartes, le dices a tu amigo 'me importas.' ¡Eso es hermoso!"
            },
            {
              letter: 'C',
              text: '¡Ayudar a limpiar sin que me lo pidan!',
              response:
                '¡GUAU! 🌟 ¿Ayudar sin que te lo pidan? ¡Eso es como tener superpoderes secretos! ¡Tus adultos estarán TAN orgullosos!'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: "Faire un câlin à quelqu'un !",
              response:
                "Les câlins sont MAGIQUES ! 🤗 Ils rendent toi ET l'autre personne heureux. C'est être un bon citoyen de ta famille !"
            },
            {
              letter: 'B',
              text: 'Partager mes jouets avec un ami !',
              response:
                "Partager c'est prendre soin ! 🧸 Quand tu partages, tu dis à ton ami 'tu comptes pour moi.' C'est magnifique !"
            },
            {
              letter: 'C',
              text: "Aider à ranger sans qu'on me le demande !",
              response:
                "WOW ! 🌟 Aider sans qu'on te le demande ? C'est comme avoir des super-pouvoirs secrets ! Tes adultes seront TELLEMENT fiers !"
            }
          ]
        },
        '6-12': {
          en: [
            {
              letter: 'A',
              text: 'Stand up for someone being bullied',
              response:
                "HERO MOVE! 🦸 Standing up for others is one of the bravest things you can do. You could change someone's whole day — or life!"
            },
            {
              letter: 'B',
              text: 'Start a project to help my community',
              response:
                "Wow, taking initiative! 🚀 A cleanup day? A food drive? A kindness campaign? You don't need to wait for adults — you can lead!"
            },
            {
              letter: 'C',
              text: 'Learn about an issue and tell others',
              response:
                'Knowledge warrior! 📚 When you learn and share, you multiply your impact. One informed kid can inform hundreds!'
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Defender a alguien que está siendo acosado',
              response:
                '¡MOVIMIENTO DE HÉROE! 🦸 Defender a otros es una de las cosas más valientes que puedes hacer. ¡Podrías cambiar el día — o la vida — de alguien!'
            },
            {
              letter: 'B',
              text: 'Iniciar un proyecto para ayudar a mi comunidad',
              response:
                '¡Guau, tomando la iniciativa! 🚀 ¿Un día de limpieza? ¿Una colecta de alimentos? ¿Una campaña de bondad? No necesitas esperar a los adultos — ¡puedes liderar!'
            },
            {
              letter: 'C',
              text: 'Aprender sobre un tema y contarle a otros',
              response:
                '¡Guerrero del conocimiento! 📚 Cuando aprendes y compartes, multiplicas tu impacto. ¡Un niño informado puede informar a cientos!'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: "Défendre quelqu'un qui est harcelé",
              response:
                "MOUVEMENT DE HÉROS ! 🦸 Défendre les autres est l'une des choses les plus courageuses que tu puisses faire. Tu pourrais changer la journée — ou la vie — de quelqu'un !"
            },
            {
              letter: 'B',
              text: 'Démarrer un projet pour aider ma communauté',
              response:
                "Wow, prendre l'initiative ! 🚀 Une journée de nettoyage ? Une collecte de nourriture ? Une campagne de gentillesse ? Tu n'as pas besoin d'attendre les adultes — tu peux diriger !"
            },
            {
              letter: 'C',
              text: 'Apprendre sur un sujet et en parler aux autres',
              response:
                'Guerrier du savoir ! 📚 Quand tu apprends et partages, tu multiplies ton impact. Un enfant informé peut en informer des centaines !'
            }
          ]
        },
        '13-17': {
          en: [
            {
              letter: 'A',
              text: 'Volunteer for a cause I care about',
              response:
                'Action over words! 👏 Find an org doing work you believe in. Your time and energy are valuable — invest them where they matter to YOU.'
            },
            {
              letter: 'B',
              text: 'Use my social media to spread awareness',
              response:
                'Your platform is power. 📱 But make it count — research before sharing, add your perspective, and follow up with action.'
            },
            {
              letter: 'C',
              text: 'Organize something at school that makes a difference',
              response:
                "Leadership energy! 🎓 A club, a campaign, a conversation — you have more power to change your school than you think. What's stopping you?"
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Ser voluntario para una causa que me importa',
              response:
                '¡Acción sobre palabras! 👏 Encuentra una organización haciendo trabajo en el que crees. Tu tiempo y energía son valiosos — inviértelos donde te importen A TI.'
            },
            {
              letter: 'B',
              text: 'Usar mis redes sociales para crear conciencia',
              response:
                'Tu plataforma es poder. 📱 Pero hazlo contar — investiga antes de compartir, añade tu perspectiva, y sigue con acción.'
            },
            {
              letter: 'C',
              text: 'Organizar algo en la escuela que marque diferencia',
              response:
                '¡Energía de liderazgo! 🎓 Un club, una campaña, una conversación — tienes más poder para cambiar tu escuela de lo que piensas. ¿Qué te detiene?'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: 'Faire du bénévolat pour une cause qui me tient à cœur',
              response:
                "L'action plutôt que les mots ! 👏 Trouve une org qui fait un travail auquel tu crois. Ton temps et ton énergie sont précieux — investis-les là où ça compte pour TOI."
            },
            {
              letter: 'B',
              text: 'Utiliser mes réseaux sociaux pour sensibiliser',
              response:
                "Ta plateforme est un pouvoir. 📱 Mais fais que ça compte — recherche avant de partager, ajoute ta perspective, et suis avec de l'action."
            },
            {
              letter: 'C',
              text: "Organiser quelque chose à l'école qui fait la différence",
              response:
                "Énergie de leader ! 🎓 Un club, une campagne, une conversation — tu as plus de pouvoir pour changer ton école que tu ne le penses. Qu'est-ce qui t'arrête ?"
            }
          ]
        },
        '18-35': {
          en: [
            {
              letter: 'A',
              text: 'Donate money or skills to organizations I trust',
              response:
                "Strategic giving. 💡 Whether it's cash or skills, your contribution multiplies when directed to effective organizations. Smart citizenship."
            },
            {
              letter: 'B',
              text: 'Get more involved in local politics and decisions',
              response:
                "That's where the action is! 🏛️ City council meetings, local boards, community planning — your voice shapes your neighborhood."
            },
            {
              letter: 'C',
              text: 'Mentor someone earlier in their journey',
              response:
                'Paying it forward. 🌱 Your experience is valuable. One mentor relationship can change a trajectory. Who needs what you know?'
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Donar dinero o habilidades a organizaciones en las que confío',
              response:
                'Donación estratégica. 💡 Ya sea dinero o habilidades, tu contribución se multiplica cuando se dirige a organizaciones efectivas. Ciudadanía inteligente.'
            },
            {
              letter: 'B',
              text: 'Involucrarme más en la política y decisiones locales',
              response:
                '¡Ahí es donde está la acción! 🏛️ Reuniones del consejo municipal, juntas locales, planificación comunitaria — tu voz moldea tu vecindario.'
            },
            {
              letter: 'C',
              text: 'Mentorear a alguien más temprano en su camino',
              response:
                'Devolver el favor. 🌱 Tu experiencia es valiosa. Una relación de mentoría puede cambiar una trayectoria. ¿Quién necesita lo que tú sabes?'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: "Donner de l'argent ou des compétences à des organisations de confiance",
              response:
                "Don stratégique. 💡 Que ce soit de l'argent ou des compétences, ta contribution se multiplie quand elle est dirigée vers des organisations efficaces. Citoyenneté intelligente."
            },
            {
              letter: 'B',
              text: "M'impliquer plus dans la politique et les décisions locales",
              response:
                "C'est là que l'action est ! 🏛️ Réunions du conseil municipal, conseils locaux, planification communautaire — ta voix façonne ton quartier."
            },
            {
              letter: 'C',
              text: "Mentorer quelqu'un plus tôt dans son parcours",
              response:
                'Rendre la pareille. 🌱 Ton expérience est précieuse. Une relation de mentorat peut changer une trajectoire. Qui a besoin de ce que tu sais ?'
            }
          ]
        },
        '36-60': {
          en: [
            {
              letter: 'A',
              text: 'Create or support programs that outlast me',
              response:
                "Legacy thinking. 🏛️ Scholarships, foundations, community programs — what you build now can serve generations you'll never meet."
            },
            {
              letter: 'B',
              text: 'Run for office or support those who share my values',
              response:
                'Stepping into leadership. 🎯 Whether you run or support, political engagement at this stage of life carries extra weight. Use it.'
            },
            {
              letter: 'C',
              text: "Document and share what I've learned",
              response:
                'Wisdom preservation. 📚 Your lessons — both successes and failures — are textbooks for others. Write, teach, share. It matters.'
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Crear o apoyar programas que me sobrevivan',
              response:
                'Pensamiento de legado. 🏛️ Becas, fundaciones, programas comunitarios — lo que construyes ahora puede servir a generaciones que nunca conocerás.'
            },
            {
              letter: 'B',
              text: 'Postularme a un cargo o apoyar a quienes comparten mis valores',
              response:
                'Entrando en liderazgo. 🎯 Ya sea que te postules o apoyes, el compromiso político en esta etapa de la vida tiene peso extra. Úsalo.'
            },
            {
              letter: 'C',
              text: 'Documentar y compartir lo que he aprendido',
              response:
                'Preservación de sabiduría. 📚 Tus lecciones — tanto éxitos como fracasos — son libros de texto para otros. Escribe, enseña, comparte. Importa.'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: 'Créer ou soutenir des programmes qui me survivront',
              response:
                "Pensée d'héritage. 🏛️ Bourses, fondations, programmes communautaires — ce que tu construis maintenant peut servir des générations que tu ne rencontreras jamais."
            },
            {
              letter: 'B',
              text: 'Me présenter ou soutenir ceux qui partagent mes valeurs',
              response:
                "Entrer dans le leadership. 🎯 Que tu te présentes ou que tu soutiennes, l'engagement politique à ce stade de la vie a un poids supplémentaire. Utilise-le."
            },
            {
              letter: 'C',
              text: "Documenter et partager ce que j'ai appris",
              response:
                'Préservation de la sagesse. 📚 Tes leçons — succès et échecs — sont des manuels pour les autres. Écris, enseigne, partage. Ça compte.'
            }
          ]
        },
        '61+': {
          en: [
            {
              letter: 'A',
              text: 'Spend time with young people and share my stories',
              response:
                'The gift of time. ⏰ Your stories carry lessons no textbook can teach. Every conversation plants seeds you may never see bloom.'
            },
            {
              letter: 'B',
              text: 'Stay active in causes that matter to me',
              response:
                'Continued engagement. 💪 Age is not retirement from citizenship. Your experience makes your participation even more valuable.'
            },
            {
              letter: 'C',
              text: 'Help bridge divides between generations',
              response:
                "The connector role. 🌉 You've lived through changes others can only read about. Your ability to translate between eras is irreplaceable."
            }
          ],
          es: [
            {
              letter: 'A',
              text: 'Pasar tiempo con jóvenes y compartir mis historias',
              response:
                'El regalo del tiempo. ⏰ Tus historias llevan lecciones que ningún libro de texto puede enseñar. Cada conversación planta semillas que quizás nunca veas florecer.'
            },
            {
              letter: 'B',
              text: 'Mantenerme activo en causas que me importan',
              response:
                'Compromiso continuo. 💪 La edad no es jubilación de la ciudadanía. Tu experiencia hace tu participación aún más valiosa.'
            },
            {
              letter: 'C',
              text: 'Ayudar a unir las divisiones entre generaciones',
              response:
                'El rol de conector. 🌉 Has vivido cambios que otros solo pueden leer. Tu habilidad para traducir entre eras es irremplazable.'
            }
          ],
          fr: [
            {
              letter: 'A',
              text: 'Passer du temps avec les jeunes et partager mes histoires',
              response:
                "Le cadeau du temps. ⏰ Tes histoires portent des leçons qu'aucun manuel ne peut enseigner. Chaque conversation plante des graines que tu ne verras peut-être jamais fleurir."
            },
            {
              letter: 'B',
              text: 'Rester actif dans les causes qui me tiennent à cœur',
              response:
                "Engagement continu. 💪 L'âge n'est pas la retraite de la citoyenneté. Ton expérience rend ta participation encore plus précieuse."
            },
            {
              letter: 'C',
              text: 'Aider à combler les fossés entre les générations',
              response:
                "Le rôle de connecteur. 🌉 Tu as vécu des changements que d'autres ne peuvent que lire. Ta capacité à traduire entre les époques est irremplaçable."
            }
          ]
        }
      }
    },

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 5: WISDOM
    // ═══════════════════════════════════════════════════════════════════
    {
      order: 5,
      type: 'wisdom',
      name: 'Wisdom',
      expression: 'wisdom',

      text: {
        '2-5': {
          en: "Here's a secret to remember: 🌟 Every time you're kind, every time you share, every time you help — you're making the world a better place! You're already a wonderful citizen of your family, your class, and your world. Be proud! Now give yourself a big hug — you earned it! 🤗",
          es: 'Aquí hay un secreto para recordar: 🌟 ¡Cada vez que eres amable, cada vez que compartes, cada vez que ayudas — estás haciendo del mundo un lugar mejor! Ya eres un ciudadano maravilloso de tu familia, tu clase y tu mundo. ¡Siéntete orgulloso! ¡Ahora date un gran abrazo — te lo ganaste! 🤗',
          fr: "Voici un secret à retenir : 🌟 Chaque fois que tu es gentil, chaque fois que tu partages, chaque fois que tu aides — tu rends le monde meilleur ! Tu es déjà un citoyen merveilleux de ta famille, ta classe et ton monde. Sois fier ! Maintenant fais-toi un gros câlin — tu l'as mérité ! 🤗"
        },
        '6-12': {
          en: "Here's the wisdom I want you to carry: 🦸 You don't need to wait until you're older to be a great citizen. Every day you have chances to be a hero — in your classroom, on your street, in your family. Start small, dream big, and remember: the world needs your superpowers right NOW. Go be amazing!",
          es: 'Aquí está la sabiduría que quiero que lleves: 🦸 No necesitas esperar hasta ser mayor para ser un gran ciudadano. Cada día tienes oportunidades de ser un héroe — en tu salón, en tu calle, en tu familia. Empieza pequeño, sueña en grande, y recuerda: ¡el mundo necesita tus superpoderes AHORA MISMO! ¡Ve a ser increíble!',
          fr: "Voici la sagesse que je veux que tu emportes : 🦸 Tu n'as pas besoin d'attendre d'être plus grand pour être un grand citoyen. Chaque jour tu as des chances d'être un héros — dans ta classe, dans ta rue, dans ta famille. Commence petit, rêve grand, et rappelle-toi : le monde a besoin de tes super-pouvoirs MAINTENANT. Va être incroyable !"
        },
        '13-17': {
          en: "Real talk one more time: 🎯 You're not 'too young' to matter. You're not 'just a kid.' The world needs your energy, your ideas, your questions. Citizenship isn't something you grow into — it's something you DO, starting now. Your move. Make it count.",
          es: "Hablemos en serio una vez más: 🎯 No eres 'muy joven' para importar. No eres 'solo un niño.' El mundo necesita tu energía, tus ideas, tus preguntas. La ciudadanía no es algo en lo que creces — es algo que HACES, empezando ahora. Tu turno. Hazlo contar.",
          fr: "Parlons vrai une dernière fois : 🎯 Tu n'es pas 'trop jeune' pour compter. Tu n'es pas 'juste un ado.' Le monde a besoin de ton énergie, tes idées, tes questions. La citoyenneté n'est pas quelque chose dans quoi tu grandis — c'est quelque chose que tu FAIS, à partir de maintenant. À toi de jouer. Fais que ça compte."
        },
        '18-35': {
          en: "Here's what I want you to take with you: 🌱 You're at the age where your actions compound. The habits you build now, the communities you nurture, the stands you take — they all add up. Citizenship isn't separate from your career, your relationships, your life. It IS your life. Make it intentional.",
          es: 'Esto es lo que quiero que te lleves: 🌱 Estás en la edad donde tus acciones se multiplican. Los hábitos que construyes ahora, las comunidades que nutres, las posturas que tomas — todo suma. La ciudadanía no está separada de tu carrera, tus relaciones, tu vida. ES tu vida. Hazla intencional.',
          fr: "Voici ce que je veux que tu emportes : 🌱 Tu es à l'âge où tes actions se cumulent. Les habitudes que tu construis maintenant, les communautés que tu nourris, les positions que tu prends — tout s'additionne. La citoyenneté n'est pas séparée de ta carrière, tes relations, ta vie. C'EST ta vie. Rends-la intentionnelle."
        },
        '36-60': {
          en: "A final thought: 🏗️ You're in the building years. What you construct now — in your family, your company, your community — becomes the infrastructure for those who follow. Not everyone gets to build. You do. Use it well. The scaffolding you leave behind holds others up.",
          es: 'Un pensamiento final: 🏗️ Estás en los años de construcción. Lo que construyes ahora — en tu familia, tu empresa, tu comunidad — se convierte en la infraestructura para los que siguen. No todos pueden construir. Tú puedes. Úsalo bien. El andamiaje que dejas sostiene a otros.',
          fr: "Une dernière pensée : 🏗️ Tu es dans les années de construction. Ce que tu construis maintenant — dans ta famille, ton entreprise, ta communauté — devient l'infrastructure pour ceux qui suivent. Tout le monde ne peut pas construire. Toi si. Utilise-le bien. L'échafaudage que tu laisses soutient les autres."
        },
        '61+': {
          en: "Let me leave you with this: 🕊️ Your life is a textbook that's still being written. The wisdom you hold, the mistakes you've learned from, the victories you've won — they're not just your story. They're a gift for everyone who comes after. Share it. Your citizenship never ends. Thank you for all you've already given.",
          es: 'Déjame dejarte con esto: 🕊️ Tu vida es un libro de texto que todavía se está escribiendo. La sabiduría que tienes, los errores de los que has aprendido, las victorias que has ganado — no son solo tu historia. Son un regalo para todos los que vienen después. Compártelo. Tu ciudadanía nunca termina. Gracias por todo lo que ya has dado.',
          fr: "Laisse-moi te laisser avec ceci : 🕊️ Ta vie est un manuel qui s'écrit encore. La sagesse que tu détiens, les erreurs dont tu as appris, les victoires que tu as remportées — ce n'est pas juste ton histoire. C'est un cadeau pour tous ceux qui viennent après. Partage-le. Ta citoyenneté ne finit jamais. Merci pour tout ce que tu as déjà donné."
        }
      },

      hint: null,
      choices: null
    }
  ]
};

// Export for use in learn.html
if (typeof window !== 'undefined') {
  window.GOLDEN_LESSON_333 = GOLDEN_LESSON_333;
}









