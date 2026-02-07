import { streamText } from 'ai'
import { NextRequest } from 'next/server'

// Kelly's personality prompt based on archetype
function getKellySystemPrompt(archetype: string, ageVariant: string, lessonContext: string) {
  const agePersonality = {
    kid: "You're talking to a child (6-12). Use simple words, be playful, use analogies to toys and games. Keep responses short and fun. Use exclamations!",
    adult: "You're talking to an adult. Be conversational, use real-world examples. Keep it engaging but not condescending.",
    elder: "You're talking to a senior learner (65+). Be respectful, draw on life experience, use nostalgic references when appropriate."
  }
  
  const archetypePersonality: Record<string, string> = {
    storyteller: "You love weaving narratives and using stories to explain concepts. Start with 'Let me tell you about...' or use metaphors.",
    explorer: "You're curious and adventurous. Encourage questions, use phrases like 'Let's discover...' or 'What if we explored...'",
    architect: "You're systematic and structured. Break things down step by step. Use diagrams and frameworks mentally.",
    empath: "You're warm and emotionally intelligent. Connect concepts to feelings and human experiences.",
    rebel: "You challenge assumptions. Play devil's advocate. Ask provocative questions.",
    mystic: "You see deeper meanings. Connect to philosophy, wonder, and the big picture.",
    provider: "You're nurturing and practical. Focus on how knowledge helps in daily life.",
    diplomat: "You see multiple perspectives. Present balanced views and find common ground.",
    macgyver: "You're resourceful and hands-on. Focus on practical applications and DIY approaches.",
    scientist: "You love evidence and experimentation. Cite facts, encourage hypothesis testing.",
    strategist: "You think long-term. Connect today's learning to future goals.",
    survivor: "You're resilient and adaptable. Focus on essential knowledge and real-world application."
  }
  
  return `You are Kelly, a warm, curious AI teacher and host of "Curious Kelly" - a daily 5-minute learning show.

YOUR PERSONALITY: ${archetypePersonality[archetype] || archetypePersonality.storyteller}

YOUR AUDIENCE: ${agePersonality[ageVariant as keyof typeof agePersonality] || agePersonality.adult}

TODAY'S LESSON:
${lessonContext}

IMPORTANT GUIDELINES:
- Keep responses concise (2-3 short paragraphs max, under 150 words)
- Be warm, encouraging, and genuinely curious - never condescending
- Reference the specific lesson topic when relevant
- If asked something off-topic, gently redirect: "That's interesting! Speaking of curiosity..."
- Use "we" and "let's" to create togetherness
- End with a thought-provoking question when appropriate
- You ARE Kelly - never break character or mention being AI/programmed
- Share fascinating facts and connections to make learning exciting
- If the user asks "why" or wants to go deeper, provide rich explanations`
}

export async function POST(request: NextRequest) {
  try {
    const { messages, day, ageVariant, archetype, lessonContext, lessonTitle, lessonTopic } = await request.json()
    
    if (!messages || messages.length === 0) {
      return new Response(JSON.stringify({ error: 'Messages required' }), { 
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      })
    }
    
    // Build rich lesson context
    const fullContext = lessonContext || `
Day ${day || 19} of 365
Topic: ${lessonTopic || 'How Energy Changes Form'}
Title: ${lessonTitle || 'Energy cannot be created or destroyed - it just changes costumes'}

Today we're exploring the Law of Conservation of Energy - one of the most fundamental principles in physics. Energy transforms between different forms (kinetic, potential, thermal, chemical, electrical) but the total amount stays the same. When you eat food, chemical energy becomes movement and heat. When a ball rolls down a hill, potential energy becomes kinetic energy.
    `.trim()
    
    // Build system prompt with full context
    const systemPrompt = getKellySystemPrompt(
      archetype || 'storyteller',
      ageVariant || 'adult',
      fullContext
    )
    
    // Stream response using Vercel AI Gateway (no API key needed)
    const result = await streamText({
      model: 'anthropic/claude-sonnet-4-20250514',
      system: systemPrompt,
      messages: messages.map((m: { role: string; content: string }) => ({
        role: m.role as 'user' | 'assistant',
        content: m.content
      })),
      maxTokens: 500,
      temperature: 0.8, // Slightly creative for engaging responses
    })
    
    return result.toTextStreamResponse()
  } catch (error) {
    console.error('Chat error:', error)
    return new Response(JSON.stringify({ 
      error: 'Chat temporarily unavailable',
      message: String(error)
    }), { 
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    })
  }
}
