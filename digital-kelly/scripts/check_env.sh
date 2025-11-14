#!/bin/bash

echo "🔍 Kelly OS - Environment Check"
echo "================================"
echo ""

if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env from .env.example"
        echo "⚠️  Please edit .env and fill in ELEVENLABS_API_KEY"
        exit 1
    else
        echo "❌ .env.example not found"
        exit 1
    fi
fi

source .env

if [ -z "$ELEVENLABS_API_KEY" ] || [ "$ELEVENLABS_API_KEY" = "__PUT_KEY_IN_YOUR_LOCAL_.env__" ]; then
    echo "❌ ELEVENLABS_API_KEY not set in .env"
    echo "   Edit .env and add your ElevenLabs API key"
    exit 1
fi

if [ ${#ELEVENLABS_API_KEY} -lt 20 ]; then
    echo "⚠️  ELEVENLABS_API_KEY looks suspiciously short"
    echo "   Verify it's correct"
    exit 1
fi

echo "✅ .env file exists"
echo "✅ ELEVENLABS_API_KEY is set (length: ${#ELEVENLABS_API_KEY})"
echo ""
echo "✅ Environment check passed!"



















