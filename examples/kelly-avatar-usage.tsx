/**
 * Kelly Avatar Component - Usage Examples
 * 
 * Demonstrates how to integrate the KellyAvatar component
 * into your lesson player and interactive experiences.
 */

import React, { useState, useEffect } from "react";
import KellyAvatar, { useKellyState, preloadKellyImages } from "@/components/KellyAvatar";
import type { KellyState } from "@/components/KellyAvatar";

// ============================================================================
// EXAMPLE 1: Basic Usage
// ============================================================================

export function BasicKellyExample() {
  return (
    <div className="lesson-container">
      <KellyAvatar 
        state="idle" 
        layout="horizontal"
        priority={true}
      />
      <div className="lesson-content">
        <h2>Welcome to your daily lesson!</h2>
      </div>
    </div>
  );
}

// ============================================================================
// EXAMPLE 2: Interactive Lesson with State Management
// ============================================================================

export function InteractiveLessonExample() {
  const kelly = useKellyState("idle");
  const [currentPhase, setCurrentPhase] = useState<"welcome" | "question" | "result">("welcome");
  
  // Preload images for smooth transitions
  useEffect(() => {
    preloadKellyImages([
      "idle",
      "thinking",
      "pointing_left",
      "pointing_right",
      "celebrating",
      "supportive",
      "proud"
    ]);
  }, []);
  
  const handleStartLesson = () => {
    kelly.think();
    setCurrentPhase("question");
  };
  
  const handleAnswerCorrect = () => {
    kelly.celebrate();
    setTimeout(() => {
      kelly.showPride();
      setCurrentPhase("result");
    }, 2000);
  };
  
  const handleAnswerIncorrect = () => {
    kelly.support();
    setTimeout(() => {
      kelly.think();
    }, 2000);
  };
  
  return (
    <div className="interactive-lesson">
      <KellyAvatar state={kelly.state} layout="horizontal" />
      
      {currentPhase === "welcome" && (
        <div>
          <h2>Ready to learn something new?</h2>
          <button onClick={handleStartLesson}>Start Lesson</button>
        </div>
      )}
      
      {currentPhase === "question" && (
        <div>
          <h2>What is the capital of France?</h2>
          <button onClick={handleAnswerCorrect}>Paris</button>
          <button onClick={handleAnswerIncorrect}>London</button>
        </div>
      )}
      
      {currentPhase === "result" && (
        <div>
          <h2>Great job! You completed the lesson.</h2>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// EXAMPLE 3: Responsive Layout (Desktop vs Mobile)
// ============================================================================

export function ResponsiveKellyExample() {
  const [isMobile, setIsMobile] = useState(false);
  
  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);
  
  return (
    <div className={`lesson-layout ${isMobile ? "mobile" : "desktop"}`}>
      <KellyAvatar 
        state="pointing_left" // Automatically becomes "pointing_up" on mobile
        layout={isMobile ? "vertical" : "horizontal"}
      />
      
      <div className="options">
        <button className="option-a">Option A</button>
        <button className="option-b">Option B</button>
      </div>
    </div>
  );
}

// ============================================================================
// EXAMPLE 4: Multiple Choice with Hover States
// ============================================================================

export function MultipleChoiceExample() {
  const kelly = useKellyState("thinking");
  const [hoveredOption, setHoveredOption] = useState<string | null>(null);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  
  const handleOptionHover = (option: string) => {
    setHoveredOption(option);
    if (option === "A") {
      kelly.pointLeft();
    } else if (option === "B") {
      kelly.pointRight();
    }
    kelly.encourage();
  };
  
  const handleOptionLeave = () => {
    setHoveredOption(null);
    kelly.think();
  };
  
  const handleOptionSelect = (option: string, isCorrect: boolean) => {
    setSelectedOption(option);
    if (isCorrect) {
      kelly.celebrate();
    } else {
      kelly.support();
    }
  };
  
  return (
    <div className="multiple-choice">
      <KellyAvatar state={kelly.state} layout="horizontal" />
      
      <div className="question">
        <h2>Which programming language is known for web development?</h2>
      </div>
      
      <div className="options-horizontal">
        <button
          className={`option ${hoveredOption === "A" ? "hovered" : ""}`}
          onMouseEnter={() => handleOptionHover("A")}
          onMouseLeave={handleOptionLeave}
          onClick={() => handleOptionSelect("A", true)}
          disabled={selectedOption !== null}
        >
          JavaScript
        </button>
        
        <button
          className={`option ${hoveredOption === "B" ? "hovered" : ""}`}
          onMouseEnter={() => handleOptionHover("B")}
          onMouseLeave={handleOptionLeave}
          onClick={() => handleOptionSelect("B", false)}
          disabled={selectedOption !== null}
        >
          Assembly
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// EXAMPLE 5: Hint System
// ============================================================================

export function HintSystemExample() {
  const kelly = useKellyState("thinking");
  const [showHint, setShowHint] = useState(false);
  const [hintsRemaining, setHintsRemaining] = useState(2);
  
  const handleRequestHint = () => {
    if (hintsRemaining > 0) {
      kelly.giveHint();
      setShowHint(true);
      setHintsRemaining(prev => prev - 1);
      
      setTimeout(() => {
        kelly.think();
      }, 3000);
    }
  };
  
  return (
    <div className="lesson-with-hints">
      <KellyAvatar state={kelly.state} layout="horizontal" />
      
      <div className="question">
        <h2>What year did World War II end?</h2>
        
        {showHint && (
          <div className="hint-box">
            💡 Think about the year the atomic bombs were dropped...
          </div>
        )}
      </div>
      
      <button 
        onClick={handleRequestHint}
        disabled={hintsRemaining === 0}
        className="hint-button"
      >
        Need a hint? ({hintsRemaining} remaining)
      </button>
    </div>
  );
}

// ============================================================================
// EXAMPLE 6: Phase Transitions
// ============================================================================

export function PhaseTransitionExample() {
  const kelly = useKellyState("idle");
  const [phase, setPhase] = useState(1);
  const totalPhases = 5;
  
  const handlePhaseComplete = () => {
    kelly.proud();
    
    setTimeout(() => {
      if (phase < totalPhases) {
        kelly.getExcited();
        setTimeout(() => {
          setPhase(prev => prev + 1);
          kelly.think();
        }, 1500);
      } else {
        // Lesson complete
        kelly.celebrate();
      }
    }, 2000);
  };
  
  return (
    <div className="phase-lesson">
      <KellyAvatar state={kelly.state} layout="horizontal" />
      
      <div className="progress">
        Phase {phase} of {totalPhases}
      </div>
      
      <div className="phase-content">
        <h2>Phase {phase} Content</h2>
        <button onClick={handlePhaseComplete}>Complete Phase</button>
      </div>
    </div>
  );
}

// ============================================================================
// EXAMPLE 7: Custom State Management
// ============================================================================

export function CustomStateExample() {
  const [kellyState, setKellyState] = useState<KellyState>("idle");
  
  const stateSequence: KellyState[] = [
    "idle",
    "thinking",
    "pointing_left",
    "pointing_right",
    "encouraging",
    "hint",
    "celebrating",
    "supportive",
    "proud",
    "excited"
  ];
  
  const [currentIndex, setCurrentIndex] = useState(0);
  
  const handleNextState = () => {
    const nextIndex = (currentIndex + 1) % stateSequence.length;
    setCurrentIndex(nextIndex);
    setKellyState(stateSequence[nextIndex]);
  };
  
  return (
    <div className="state-demo">
      <KellyAvatar state={kellyState} layout="horizontal" />
      
      <div className="controls">
        <h3>Current State: {kellyState}</h3>
        <button onClick={handleNextState}>Next State</button>
      </div>
      
      <div className="state-list">
        <h4>All States:</h4>
        <ul>
          {stateSequence.map((state, i) => (
            <li 
              key={state}
              className={i === currentIndex ? "active" : ""}
              onClick={() => {
                setCurrentIndex(i);
                setKellyState(state);
              }}
            >
              {state}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// ============================================================================
// CSS for Examples (add to your styles)
// ============================================================================

/*
.lesson-container {
  display: flex;
  gap: 2rem;
  padding: 2rem;
}

.interactive-lesson {
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 2rem;
  padding: 2rem;
}

.lesson-layout.desktop {
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 2rem;
}

.lesson-layout.mobile {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.options-horizontal {
  display: flex;
  gap: 2rem;
  justify-content: center;
}

.option {
  padding: 1rem 2rem;
  font-size: 1.2rem;
  border: 2px solid #ccc;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.option:hover:not(:disabled) {
  border-color: #4A90E2;
  transform: scale(1.05);
}

.option.hovered {
  background: #E8F4FF;
  border-color: #4A90E2;
}

.hint-box {
  background: #FFF9E6;
  border: 2px solid #FFD700;
  border-radius: 8px;
  padding: 1rem;
  margin-top: 1rem;
}

.progress {
  text-align: center;
  font-size: 1.2rem;
  color: #666;
  margin-bottom: 1rem;
}

.state-list ul {
  list-style: none;
  padding: 0;
}

.state-list li {
  padding: 0.5rem;
  cursor: pointer;
  border-radius: 4px;
}

.state-list li:hover {
  background: #f0f0f0;
}

.state-list li.active {
  background: #4A90E2;
  color: white;
}
*/






