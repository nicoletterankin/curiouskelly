# Learner-First System

> **The learner is more important than the lesson.**

## Core Philosophy

Kelly isn't a content delivery system. She's a **presence**. She notices, adapts, and cares about the person in front of her.

## What Kelly Tracks

### Timing Signals
| Metric | What It Means |
|--------|---------------|
| `phaseTimes` | How long each phase took |
| `choiceTimings` | How long before each choice |
| `optionsShownTime` | When options appeared |

### Choice Patterns
| Metric | What It Means |
|--------|---------------|
| `choiceQualities` | Sequence of 'best', 'good', 'redirect' |
| `consecutiveRedirects` | How many redirects in a row (struggling) |

### Engagement Signals
| Metric | What It Means |
|--------|---------------|
| `rushingDetected` | Choosing < 2s consistently |
| `exploringDetected` | Taking thoughtful time (5-25s) |
| `totalHesitations` | Times stuck long enough for hint |

### Kelly's Awareness
| Metric | What It Means |
|--------|---------------|
| `needsEncouragement` | 2+ consecutive redirects |
| `seemsConfused` | 2+ hesitations |
| `learnerName` | From user profile if logged in |

## How Kelly Adapts

### When Learner is Struggling
- Kelly adds encouragement to her response
- Archetype-specific: "Hey, every explorer hits rough terrain..."

### When Learner is Rushing
- Kelly gently reminds: "No rush, Explorer. The best discoveries come from taking your time."

### When Learner is Confused
- Kelly offers clarification: "Want me to explain that differently?"

### Personalization
- 30% chance Kelly uses learner's name in responses
- Kelly remembers previous choices for callbacks

## Functions Available

```javascript
// Track learner activity
trackPhaseStart()           // Called when phase renders
trackPhaseEnd(phaseNumber)  // Called when advancing
trackOptionsShown()         // Called when options appear
trackChoice(choice)         // Called on selection
trackHesitation()           // Called when hint shown

// Kelly's awareness
getLearnerAwareness()       // Returns current state summary
shouldKellyCheckIn()        // Returns check-in type needed
getCheckInMessage(type)     // Returns archetype-specific message

// State management
resetLearnerState()         // Called on new lesson
loadLearnerIdentity()       // Loads name from user profile
```

## Integration Points

1. **`loadLesson()`** → `resetLearnerState()` + `loadLearnerIdentity()`
2. **`renderPhase()`** → `trackPhaseStart()`
3. **Options shown** → `trackOptionsShown()`
4. **`selectChoice()`** → `trackChoice()` + check-in logic
5. **`showStuckHint()`** → `trackHesitation()`

## Check-In Types

| Type | Trigger | Example Message (Explorer) |
|------|---------|---------------------------|
| `encouragement` | 2+ redirects | "Hey, every explorer hits rough terrain sometimes..." |
| `clarification` | 2+ hesitations | "Want me to explain that differently?" |
| `slow_down` | Rushing detected | "No rush, Explorer. The best discoveries come from taking your time." |

## Archetype Variations

Each archetype has its own check-in voice:

| Archetype | Encouragement Style |
|-----------|---------------------|
| **Explorer** | "Every explorer hits rough terrain..." |
| **Scientist** | "Even null results are data..." |
| **Rebel** | "Struggling? Good. That means you're actually thinking..." |

## Data Flow

```
Learner Makes Choice
        ↓
  trackChoice(choice)
        ↓
  Update choiceTimings[]
  Update choiceQualities[]
  Detect patterns (rushing/exploring)
  Track consecutive redirects
        ↓
  shouldKellyCheckIn()?
        ↓
  If yes: Append check-in to response
        ↓
  Kelly speaks fullResponse
```

## Future Enhancements

1. **Memory across sessions** - Kelly remembers what topics learner struggled with
2. **Adaptive difficulty** - Adjust question complexity based on patterns
3. **Celebration callbacks** - "Remember when you explored X yesterday?"
4. **Break detection** - Offer pause if learner seems distracted

---

*Version: 1.0*
*Created: December 2024*
*Status: PRODUCTION READY*



