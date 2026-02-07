import type { Metadata } from 'next'
import Link from 'next/link'
import { ArrowRight, BookOpen, Globe, Users, Sparkles, Code, Scale } from 'lucide-react'

export const metadata: Metadata = {
  title: 'About the Curriculum — The Daily Lesson',
  description: 'Learn about The Daily Lesson 365-day curriculum: the philosophy, structure, age personalization, language support, and how to cite our educational resources.',
  openGraph: {
    title: 'About the Curriculum — The Daily Lesson',
    description: 'A new lesson every day, 365 days a year. Personalized for ages 2-102 in 47+ languages.',
    url: 'https://www.thedailylesson.com/about/curriculum',
  },
}

export default function AboutCurriculumPage() {
  return (
    <main className="min-h-screen bg-zinc-950 text-white">
      {/* Header */}
      <header className="border-b border-white/5">
        <div className="max-w-4xl mx-auto px-6 py-8">
          <Link 
            href="/" 
            className="text-zinc-500 hover:text-white text-sm transition-colors"
          >
            &larr; Back to The Daily Lesson
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <p className="text-zinc-600 text-xs tracking-[0.3em] uppercase mb-4">About the Curriculum</p>
        <h1 
          className="text-white mb-6"
          style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: 'clamp(2rem, 5vw, 3rem)', fontWeight: 400, lineHeight: 1.2 }}
        >
          365 Days of Learning
        </h1>
        <p className="text-zinc-400 text-lg leading-relaxed max-w-2xl">
          The Daily Lesson is a complete curriculum delivered one lesson at a time—every day, for everyone, forever. 
          Universal knowledge for every age, language, and learning style.
        </p>
      </section>

      {/* What is The Daily Lesson */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <h2 
          className="text-white mb-8"
          style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: '1.5rem', fontWeight: 400 }}
        >
          What is The Daily Lesson?
        </h2>
        <div className="space-y-6 text-zinc-400 leading-relaxed">
          <p>
            The Daily Lesson is an AI-powered educational platform that delivers one lesson every day, 365 days a year. 
            Each lesson explores a universal truth—something true regardless of where you live, what you believe, or how old you are.
          </p>
          <p>
            Your AI teacher, <strong className="text-white">Curious Kelly</strong>, adapts each lesson to your age, language, and learning style. 
            A 5-year-old learning about tides gets a different experience than a 65-year-old learning the same concept—but both learn the same universal truth.
          </p>
          <p>
            We believe that education should be accessible to every human on Earth. That's why The Daily Lesson is designed 
            to work offline, in 47+ languages, across any device, from a $10 e-ink reader to a high-end tablet.
          </p>
        </div>
      </section>

      {/* The 365-Day Structure */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <h2 
          className="text-white mb-8"
          style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: '1.5rem', fontWeight: 400 }}
        >
          The 365-Day Structure
        </h2>
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {[
            { month: 'January', theme: 'The Physical World', examples: 'Gravity, light, seasons, tides' },
            { month: 'February', theme: 'Living Systems', examples: 'Cells, ecosystems, evolution, health' },
            { month: 'March', theme: 'Human History', examples: 'Civilizations, inventions, movements' },
            { month: 'April', theme: 'Mathematics & Logic', examples: 'Numbers, patterns, probability' },
            { month: 'May', theme: 'Language & Communication', examples: 'Stories, symbols, translation' },
            { month: 'June', theme: 'Arts & Creativity', examples: 'Music, visual arts, design' },
            { month: 'July', theme: 'Technology & Innovation', examples: 'Tools, computing, engineering' },
            { month: 'August', theme: 'Society & Economics', examples: 'Trade, governance, cooperation' },
            { month: 'September', theme: 'Mind & Psychology', examples: 'Memory, emotions, learning' },
            { month: 'October', theme: 'Earth & Environment', examples: 'Climate, geology, conservation' },
            { month: 'November', theme: 'Space & Cosmos', examples: 'Stars, planets, exploration' },
            { month: 'December', theme: 'Philosophy & Ethics', examples: 'Truth, justice, meaning' },
          ].map((item) => (
            <div 
              key={item.month}
              className="p-4 rounded-lg"
              style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
            >
              <p className="text-white font-medium mb-1">{item.month}</p>
              <p className="text-zinc-400 text-sm">{item.theme}</p>
              <p className="text-zinc-600 text-xs mt-2">{item.examples}</p>
            </div>
          ))}
        </div>
        <p className="text-zinc-500 text-sm">
          Each lesson follows a 5-phase structure: <span className="text-zinc-400">Hook</span> (capture attention), 
          <span className="text-zinc-400"> Story</span> (narrative context), <span className="text-zinc-400">Wonder</span> (the core insight), 
          <span className="text-zinc-400">Action</span> (interactive element), and <span className="text-zinc-400">Wisdom</span> (lasting takeaway).
        </p>
      </section>

      {/* Age Personalization - from Master Curriculum Thesis */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <div className="flex items-start gap-4 mb-8">
          <Users className="w-6 h-6 text-zinc-600 flex-shrink-0 mt-1" />
          <div>
            <h2 
              className="text-white mb-4"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: '1.5rem', fontWeight: 400 }}
            >
              Age Personalization System
            </h2>
            <p className="text-zinc-400 leading-relaxed">
              Every lesson adapts to five developmental stages, each with different vocabulary, complexity, duration, and cognitive approach:
            </p>
          </div>
        </div>
        <div className="grid md:grid-cols-5 gap-4">
          {[
            { age: 'Child', range: '2-7', style: 'Concrete, sensory, story-based, simple vocabulary', duration: '3-5 min' },
            { age: 'Youth', range: '8-14', style: 'Concrete with emerging abstract, relatable examples', duration: '8-12 min' },
            { age: 'Adult', range: '15-44', style: 'Abstract reasoning, practical application', duration: '15-20 min' },
            { age: 'Mature', range: '45-64', style: 'Life experience integration, legacy thinking', duration: '15-20 min' },
            { age: 'Elder', range: '65+', style: 'Reflection emphasis, meaning-making', duration: '12-15 min' },
          ].map((item) => (
            <div 
              key={item.age}
              className="p-4 rounded-lg text-center"
              style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
            >
              <p className="text-white font-medium">{item.age}</p>
              <p className="text-zinc-500 text-xs mb-2">{item.range} years</p>
              <p className="text-zinc-600 text-xs mb-2">{item.style}</p>
              <p className="text-zinc-700 text-[10px] font-mono">{item.duration}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 12 Archetypes - from Master Curriculum Thesis */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <h2 
          className="text-white mb-4"
          style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: '1.5rem', fontWeight: 400 }}
        >
          The 12 Archetypes
        </h2>
        <p className="text-zinc-400 leading-relaxed mb-8">
          Kelly adapts her teaching style to match how each learner sees the world, using 12 Jungian archetypes:
        </p>
        <div className="grid md:grid-cols-3 gap-4">
          {[
            { name: 'Explorer', motivation: 'Discovery, adventure', framing: 'Lessons as journeys' },
            { name: 'Sage', motivation: 'Wisdom, understanding', framing: 'Lessons as insights' },
            { name: 'Hero', motivation: 'Achievement, mastery', framing: 'Lessons as challenges' },
            { name: 'Caregiver', motivation: 'Helping others', framing: 'Lessons as tools to support' },
            { name: 'Creator', motivation: 'Making new things', framing: 'Lessons as building blocks' },
            { name: 'Rebel', motivation: 'Independence, change', framing: 'Lessons as tools for freedom' },
            { name: 'Lover', motivation: 'Connection, intimacy', framing: 'Lessons as ways to bond' },
            { name: 'Jester', motivation: 'Joy, humor', framing: 'Lessons as playful discoveries' },
            { name: 'Everyman', motivation: 'Belonging, connection', framing: 'Lessons as shared experience' },
            { name: 'Ruler', motivation: 'Control, order', framing: 'Lessons as leadership tools' },
            { name: 'Magician', motivation: 'Transformation', framing: 'Lessons as powers to wield' },
            { name: 'Innocent', motivation: 'Safety, happiness', framing: 'Lessons as paths to peace' },
          ].map((item) => (
            <div 
              key={item.name}
              className="p-4 rounded-lg"
              style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
            >
              <p className="text-white font-medium text-sm">{item.name}</p>
              <p className="text-zinc-500 text-xs mt-1">{item.motivation}</p>
              <p className="text-zinc-600 text-xs mt-2 italic">{item.framing}</p>
            </div>
          ))}
        </div>
        <div 
          className="mt-8 p-4 rounded-lg text-center"
          style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.08)' }}
        >
          <p className="text-zinc-400 text-sm">
            <span className="text-white font-mono">365</span> lessons × <span className="text-white font-mono">5</span> age buckets × <span className="text-white font-mono">12</span> archetypes × <span className="text-white font-mono">47</span> languages = 
          </p>
          <p className="text-white text-2xl font-mono mt-2">4,116,600</p>
          <p className="text-zinc-500 text-xs mt-1">unique lesson variants per year</p>
        </div>
      </section>

      {/* 5-Phase Lesson Structure - from Master Curriculum Thesis */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <h2 
          className="text-white mb-4"
          style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: '1.5rem', fontWeight: 400 }}
        >
          The 5-Phase Lesson Structure
        </h2>
        <p className="text-zinc-400 leading-relaxed mb-8">
          Every lesson follows a 100-second narrative arc designed for maximum engagement and retention:
        </p>
        <div className="space-y-4">
          {[
            { phase: 'HOOK', duration: '15 sec', purpose: 'Grab attention, pose True/False question', expression: 'Excited' },
            { phase: 'STORY', duration: '30 sec', purpose: 'Narrative illustration with emotional connection', expression: 'Talking' },
            { phase: 'WONDER', duration: '20 sec', purpose: 'Deeper questions, connections to learner\'s life', expression: 'Curious' },
            { phase: 'ACTION', duration: '20 sec', purpose: '60-second hands-on activity prompt', expression: 'Thinking' },
            { phase: 'WISDOM', duration: '15 sec', purpose: 'Memorable one-sentence takeaway', expression: 'Talking' },
          ].map((item, i) => (
            <div 
              key={item.phase}
              className="flex items-center gap-6 p-4 rounded-lg"
              style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
            >
              <div className="w-8 h-8 rounded-full bg-zinc-900 flex items-center justify-center text-zinc-500 text-sm font-mono">
                {i + 1}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-1">
                  <p className="text-white font-medium text-sm">{item.phase}</p>
                  <span className="text-zinc-600 text-xs font-mono">{item.duration}</span>
                </div>
                <p className="text-zinc-500 text-xs">{item.purpose}</p>
              </div>
              <div className="text-zinc-600 text-xs">Kelly: {item.expression}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Language Support */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <div className="flex items-start gap-4 mb-8">
          <Globe className="w-6 h-6 text-zinc-600 flex-shrink-0 mt-1" />
          <div>
            <h2 
              className="text-white mb-4"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: '1.5rem', fontWeight: 400 }}
            >
              Language Support
            </h2>
            <p className="text-zinc-400 leading-relaxed">
              The Daily Lesson is available in 47+ languages, with more being added continuously. 
              Each translation is culturally adapted, not just word-for-word converted.
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {['English', 'Spanish', 'French', 'German', 'Portuguese', 'Chinese', 'Japanese', 'Korean', 'Arabic', 'Hindi', 'Russian', 'Italian', 'Dutch', 'Polish', 'Turkish', 'Vietnamese', 'Thai', 'Indonesian', 'Swahili', 'Hebrew', '+27 more'].map((lang) => (
            <span 
              key={lang}
              className="px-3 py-1 rounded-full text-xs"
              style={{ background: 'rgba(255,255,255,0.05)', color: 'rgba(255,255,255,0.6)' }}
            >
              {lang}
            </span>
          ))}
        </div>
      </section>

      {/* Kelly's Voice - from Master Curriculum Thesis */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <h2 
          className="text-white mb-4"
          style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: '1.5rem', fontWeight: 400 }}
        >
          Kelly's Voice
        </h2>
        <p className="text-zinc-400 leading-relaxed mb-8">
          Kelly has a distinct personality: warm but not saccharine, direct, curious, humble, and never condescending.
        </p>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <p className="text-zinc-500 text-xs uppercase tracking-wider mb-4">Words Kelly Uses</p>
            <ul className="space-y-2">
              {[
                '"I wonder..."',
                '"What if..."',
                '"Have you ever noticed..."',
                '"Here\'s what I find fascinating..."',
                '"You might be surprised..."',
              ].map((phrase) => (
                <li key={phrase} className="text-zinc-400 text-sm italic">{phrase}</li>
              ))}
            </ul>
          </div>
          <div>
            <p className="text-zinc-500 text-xs uppercase tracking-wider mb-4">Words Kelly Avoids</p>
            <ul className="space-y-2">
              {[
                { word: '"Actually..."', reason: 'condescending' },
                { word: '"Basically..."', reason: 'implies simplification' },
                { word: '"Obviously..."', reason: 'makes learner feel dumb' },
                { word: '"You should..."', reason: 'prescriptive' },
                { word: 'Jargon', reason: 'without explanation' },
              ].map((item) => (
                <li key={item.word} className="text-zinc-500 text-sm">
                  <span className="text-zinc-400">{item.word}</span> <span className="text-zinc-600">— {item.reason}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="mt-8 p-4 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
          <p className="text-zinc-500 text-xs uppercase tracking-wider mb-3">Phase Closings</p>
          <div className="grid grid-cols-5 gap-4 text-center">
            {[
              { phase: 'HOOK', closing: '"Let\'s find out together."' },
              { phase: 'STORY', closing: '"Pretty interesting, right?"' },
              { phase: 'WONDER', closing: '"What do you think?"' },
              { phase: 'ACTION', closing: '"You\'ve got this."' },
              { phase: 'WISDOM', closing: '"I\'m Kelly. See you tomorrow."' },
            ].map((item) => (
              <div key={item.phase}>
                <p className="text-zinc-600 text-[10px] uppercase tracking-wider">{item.phase}</p>
                <p className="text-zinc-400 text-xs italic mt-1">{item.closing}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* What We Do NOT Teach - from Master Curriculum Thesis */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <h2 
          className="text-white mb-4"
          style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: '1.5rem', fontWeight: 400 }}
        >
          What We Do NOT Teach
        </h2>
        <p className="text-zinc-400 leading-relaxed mb-8">
          Our curriculum is universal by design. We deliberately exclude topics that require local context or are subject to debate:
        </p>
        <div className="grid md:grid-cols-2 gap-4">
          {[
            'Political positions or partisan content',
            'Religious doctrine or sectarian beliefs',
            'Culturally-specific customs requiring local context',
            'Current events or breaking news',
            'Technical skills requiring proprietary tools',
            'Content that varies by jurisdiction',
            'Medical or financial advice',
          ].map((item) => (
            <div 
              key={item}
              className="flex items-center gap-3 p-3 rounded-lg"
              style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
            >
              <span className="text-zinc-600 text-xs">×</span>
              <span className="text-zinc-400 text-sm">{item}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Philosophy */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <div className="flex items-start gap-4 mb-8">
          <Sparkles className="w-6 h-6 text-zinc-600 flex-shrink-0 mt-1" />
          <div>
            <h2 
              className="text-white mb-4"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: '1.5rem', fontWeight: 400 }}
            >
              Design Principles
            </h2>
            <div className="space-y-4 text-zinc-400 leading-relaxed">
              <p><strong className="text-white">Universal Relevance.</strong> Every topic must matter to every human being regardless of age, culture, language, or circumstance.</p>
              <p><strong className="text-white">Timeless Content.</strong> Lessons address human experiences that have existed for millennia and will continue for millennia more.</p>
              <p><strong className="text-white">Developmental Appropriateness.</strong> The same topic is presented differently based on the learner's developmental stage.</p>
              <p><strong className="text-white">Practical Application.</strong> Every lesson connects to daily life with actionable takeaways.</p>
              <p><strong className="text-white">Narrative Structure.</strong> Humans learn through stories. Every lesson follows a narrative arc.</p>
            </div>
          </div>
        </div>
      </section>

      {/* How to Cite */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <div className="flex items-start gap-4 mb-8">
          <BookOpen className="w-6 h-6 text-zinc-600 flex-shrink-0 mt-1" />
          <div>
            <h2 
              className="text-white mb-4"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: '1.5rem', fontWeight: 400 }}
            >
              How to Cite The Daily Lesson
            </h2>
            <p className="text-zinc-400 leading-relaxed mb-6">
              We encourage citation and reference of The Daily Lesson curriculum under the Creative Commons BY-NC-SA 4.0 license.
            </p>
          </div>
        </div>
        <div 
          className="p-6 rounded-lg font-mono text-sm"
          style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
        >
          <p className="text-zinc-400 mb-4">Academic citation format:</p>
          <code className="text-zinc-300 block leading-relaxed">
            Lesson of the Day, PBC. (2026). The Daily Lesson™ Curriculum.<br />
            Retrieved from https://www.thedailylesson.com/curriculum
          </code>
        </div>
        <p className="text-zinc-600 text-sm mt-4">
          For specific lessons, include the day number: <code className="text-zinc-500">/lesson/32</code> for Day 32.
        </p>
      </section>

      {/* API Documentation */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <div className="flex items-start gap-4 mb-8">
          <Code className="w-6 h-6 text-zinc-600 flex-shrink-0 mt-1" />
          <div>
            <h2 
              className="text-white mb-4"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: '1.5rem', fontWeight: 400 }}
            >
              API Documentation
            </h2>
            <p className="text-zinc-400 leading-relaxed mb-6">
              We provide public APIs for AI systems, researchers, and developers to access our curriculum data.
            </p>
          </div>
        </div>
        <div className="space-y-4">
          {[
            { 
              endpoint: '/api/curriculum/topics', 
              description: 'Complete 365-day curriculum with all lesson metadata',
              method: 'GET'
            },
            { 
              endpoint: '/api/lesson/today', 
              description: 'Today\'s lesson with full content',
              method: 'GET'
            },
            { 
              endpoint: '/api/lesson/[day]', 
              description: 'Specific lesson by day number (1-365)',
              method: 'GET'
            },
          ].map((api) => (
            <div 
              key={api.endpoint}
              className="p-4 rounded-lg flex items-center justify-between"
              style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
            >
              <div>
                <code className="text-white text-sm">{api.endpoint}</code>
                <p className="text-zinc-500 text-xs mt-1">{api.description}</p>
              </div>
              <span className="text-xs text-zinc-600 px-2 py-1 rounded bg-zinc-900">{api.method}</span>
            </div>
          ))}
        </div>
        <p className="text-zinc-600 text-sm mt-6">
          Response format: JSON with Schema.org structured data. No authentication required for public endpoints.
        </p>
      </section>

      {/* License */}
      <section className="max-w-4xl mx-auto px-6 py-16 border-b border-white/5">
        <div className="flex items-start gap-4 mb-8">
          <Scale className="w-6 h-6 text-zinc-600 flex-shrink-0 mt-1" />
          <div>
            <h2 
              className="text-white mb-4"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: '1.5rem', fontWeight: 400 }}
            >
              License
            </h2>
            <p className="text-zinc-400 leading-relaxed">
              The Daily Lesson curriculum is licensed under <strong className="text-white">Creative Commons BY-NC-SA 4.0</strong>. 
              You may share and adapt for non-commercial purposes with attribution.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="max-w-4xl mx-auto px-6 py-16">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div>
            <p className="text-zinc-500 text-sm">The Daily Lesson™ is a trademark of Lesson of the Day, PBC.</p>
            <p className="text-zinc-600 text-xs mt-1">Curious Kelly™ and Kelly™ are trademarks of Lesson of the Day, PBC.</p>
            <p className="text-zinc-700 text-xs mt-1">© 2026 Lesson of the Day, PBC. All rights reserved.</p>
          </div>
          <div className="flex gap-4">
            <Link 
              href="/curriculum"
              className="flex items-center gap-2 text-sm text-zinc-400 hover:text-white transition-colors"
            >
              View Curriculum <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </footer>
    </main>
  )
}
