#!/usr/bin/env python3
"""
Content Distribution Analysis for Kelly25 Training Data
Analyze emotional and conversational distribution for optimal training
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np
from collections import Counter
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_content_distribution():
    """Analyze content distribution for training optimization"""
    
    print("📊 Analyzing Kelly25 Content Distribution")
    print("=" * 50)
    
    # Paths
    data_dir = Path("kelly25_training_data")
    metadata_file = data_dir / "metadata.csv"
    
    # Load metadata
    metadata = pd.read_csv(metadata_file, sep='|', comment='#', names=['id', 'normalized_text', 'raw_text'])
    logger.info(f"Loaded {len(metadata)} metadata entries")
    
    # Analyze text characteristics
    text_analysis = {
        "total_samples": len(metadata),
        "character_stats": {
            "mean": float(metadata['raw_text'].str.len().mean()),
            "std": float(metadata['raw_text'].str.len().std()),
            "min": int(metadata['raw_text'].str.len().min()),
            "max": int(metadata['raw_text'].str.len().max())
        },
        "word_stats": {
            "mean": float(metadata['raw_text'].str.split().str.len().mean()),
            "std": float(metadata['raw_text'].str.split().str.len().std()),
            "min": int(metadata['raw_text'].str.split().str.len().min()),
            "max": int(metadata['raw_text'].str.split().str.len().max())
        }
    }
    
    # Analyze emotional content
    emotional_keywords = {
        "joy": ["happy", "excited", "thrilled", "delighted", "amazing", "wonderful", "fantastic", "brilliant"],
        "calm": ["calm", "peaceful", "relaxed", "gentle", "quiet", "serene", "tranquil", "soothing"],
        "encouragement": ["encourage", "support", "believe", "confident", "proud", "success", "achieve", "motivate"],
        "curious": ["curious", "wonder", "explore", "discover", "question", "investigate", "learn", "understand"],
        "confident": ["confident", "sure", "certain", "definitely", "absolutely", "certainly", "assured", "positive"],
        "empathetic": ["understand", "empathy", "compassion", "care", "support", "listen", "feel", "emotion"],
        "playful": ["fun", "playful", "enjoy", "laugh", "smile", "game", "adventure", "exciting"],
        "thoughtful": ["think", "consider", "reflect", "analyze", "contemplate", "ponder", "examine", "study"],
        "warm": ["warm", "affectionate", "loving", "caring", "kind", "gentle", "tender", "fond"],
        "excited": ["excited", "thrilled", "energetic", "enthusiastic", "eager", "pumped", "stoked", "buzzing"],
        "proud": ["proud", "accomplishment", "achievement", "success", "victory", "triumph", "win", "excellence"],
        "gentle": ["gentle", "soft", "tender", "mild", "calm", "peaceful", "quiet", "subtle"]
    }
    
    # Count emotional content
    emotional_counts = {}
    for emotion, keywords in emotional_keywords.items():
        count = 0
        for text in metadata['raw_text']:
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    count += 1
                    break
        emotional_counts[emotion] = count
    
    # Analyze conversation types
    conversation_keywords = {
        "greeting": ["hello", "hi", "welcome", "good morning", "good afternoon", "meet", "introduce"],
        "qa": ["question", "answer", "ask", "wonder", "curious", "explain", "tell me", "what"],
        "explanation": ["explain", "understand", "learn", "teach", "show", "demonstrate", "illustrate", "clarify"],
        "problem_solving": ["problem", "solve", "solution", "challenge", "figure out", "work through", "approach"],
        "encouragement": ["encourage", "support", "believe", "confident", "proud", "success", "achieve", "motivate"],
        "reflection": ["reflect", "think", "consider", "review", "summary", "conclude", "wrap up", "recap"],
        "storytelling": ["story", "tale", "narrative", "once upon", "imagine", "picture", "scenario", "example"],
        "clarification": ["clarify", "explain", "detail", "elaborate", "expand", "further", "more", "additional"],
        "validation": ["correct", "right", "good", "excellent", "perfect", "accurate", "valid", "true"],
        "transition": ["now", "next", "moving on", "transition", "shift", "change", "turn", "switch"],
        "personal": ["personal", "relationship", "connection", "bond", "trust", "care", "value", "important"],
        "closing": ["goodbye", "farewell", "end", "finish", "complete", "wrap up", "conclude", "until next"]
    }
    
    # Count conversation types
    conversation_counts = {}
    for conv_type, keywords in conversation_keywords.items():
        count = 0
        for text in metadata['raw_text']:
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    count += 1
                    break
        conversation_counts[conv_type] = count
    
    # Analyze sentence patterns
    sentence_patterns = {
        "questions": len(metadata[metadata['raw_text'].str.contains(r'\?', na=False)]),
        "exclamations": len(metadata[metadata['raw_text'].str.contains(r'!', na=False)]),
        "long_sentences": len(metadata[metadata['raw_text'].str.len() > 200]),
        "short_sentences": len(metadata[metadata['raw_text'].str.len() < 100]),
        "complex_sentences": len(metadata[metadata['raw_text'].str.contains(r'[,;]', na=False)])
    }
    
    # Create comprehensive analysis
    analysis = {
        "text_analysis": text_analysis,
        "emotional_distribution": emotional_counts,
        "conversation_distribution": conversation_counts,
        "sentence_patterns": sentence_patterns,
        "recommendations": {}
    }
    
    # Calculate percentages
    total_samples = len(metadata)
    analysis["emotional_percentages"] = {
        emotion: (count / total_samples) * 100 
        for emotion, count in emotional_counts.items()
    }
    
    analysis["conversation_percentages"] = {
        conv_type: (count / total_samples) * 100 
        for conv_type, count in conversation_counts.items()
    }
    
    # Generate recommendations
    recommendations = []
    
    # Check emotional balance
    emotional_percentages = analysis["emotional_percentages"]
    if max(emotional_percentages.values()) > 20:
        recommendations.append("Consider rebalancing emotional content - some emotions are over-represented")
    
    if min(emotional_percentages.values()) < 2:
        recommendations.append("Some emotions are under-represented - consider adding more samples")
    
    # Check conversation balance
    conversation_percentages = analysis["conversation_percentages"]
    if max(conversation_percentages.values()) > 25:
        recommendations.append("Consider rebalancing conversation types - some types are over-represented")
    
    # Check text length distribution
    if text_analysis["character_stats"]["std"] > 100:
        recommendations.append("High text length variation - consider normalizing text lengths")
    
    # Check sentence patterns
    if sentence_patterns["questions"] < total_samples * 0.1:
        recommendations.append("Low question ratio - consider adding more question-answer content")
    
    if sentence_patterns["exclamations"] < total_samples * 0.05:
        recommendations.append("Low exclamation ratio - consider adding more expressive content")
    
    analysis["recommendations"] = recommendations
    
    # Save analysis
    analysis_file = data_dir / "content_distribution_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print("\n📈 Content Distribution Summary:")
    print(f"📝 Total Samples: {total_samples:,}")
    print(f"📊 Text Length: {text_analysis['character_stats']['mean']:.1f} ± {text_analysis['character_stats']['std']:.1f} chars")
    print(f"📊 Word Count: {text_analysis['word_stats']['mean']:.1f} ± {text_analysis['word_stats']['std']:.1f} words")
    
    print("\n🎭 Emotional Distribution:")
    for emotion, percentage in sorted(analysis["emotional_percentages"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion.title()}: {percentage:.1f}% ({emotional_counts[emotion]} samples)")
    
    print("\n💬 Conversation Type Distribution:")
    for conv_type, percentage in sorted(analysis["conversation_percentages"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {conv_type.replace('_', ' ').title()}: {percentage:.1f}% ({conversation_counts[conv_type]} samples)")
    
    print("\n📋 Sentence Patterns:")
    for pattern, count in sentence_patterns.items():
        percentage = (count / total_samples) * 100
        print(f"  {pattern.replace('_', ' ').title()}: {percentage:.1f}% ({count} samples)")
    
    if recommendations:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("\n✅ No major issues detected - dataset is well-balanced!")
    
    print(f"\n💾 Analysis saved to: {analysis_file}")
    
    return analysis

if __name__ == "__main__":
    analyze_content_distribution()
