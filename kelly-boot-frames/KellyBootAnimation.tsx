/**
 * 🚀 Kelly Boot Animation Component
 * 
 * A polished loading/boot animation that transitions Kelly from
 * "thinking" to "greeting" the user using the actual Kelly LoRA images.
 * 
 * Usage:
 *   import KellyBootAnimation from './KellyBootAnimation';
 *   <KellyBootAnimation onComplete={() => setReady(true)} />
 */

import { useState, useEffect, useCallback } from 'react';

// Frame data with actual images
const FRAMES = [
  {
    id: 'thinking',
    image: '/kelly-boot-frames/kelly-boot-thinking.png',
    title: 'Thinking...',
    subtitle: 'Kelly is contemplating',
    loadingText: 'Waking up...',
  },
  {
    id: 'transition',
    image: '/kelly-boot-frames/kelly-boot-transition.png',
    title: 'Noticing you',
    subtitle: 'Turning toward camera',
    loadingText: 'Getting curious...',
  },
  {
    id: 'greeting',
    image: '/kelly-boot-frames/kelly-boot-greeting.png',
    title: 'Hello there!',
    subtitle: 'Making eye contact',
    loadingText: 'Almost ready...',
  },
  {
    id: 'smile',
    image: '/kelly-boot-frames/kelly-boot-smile.png',
    title: 'Welcome! 😊',
    subtitle: 'Ready to learn together',
    loadingText: 'Ready!',
  },
];

interface KellyBootAnimationProps {
  /** Callback when boot animation completes */
  onComplete?: () => void;
  /** Duration for each frame in ms (default: 800) */
  frameDuration?: number;
  /** Delay before transitioning to app after final frame (default: 1200) */
  holdDuration?: number;
  /** Enable debug mode to show frame info */
  debug?: boolean;
}

export default function KellyBootAnimation({
  onComplete,
  frameDuration = 800,
  holdDuration = 1200,
  debug = false,
}: KellyBootAnimationProps) {
  const [frame, setFrame] = useState(0);
  const [phase, setPhase] = useState<'boot' | 'fadeout' | 'complete'>('boot');
  const [isBlinking, setIsBlinking] = useState(false);

  // Boot sequence
  useEffect(() => {
    if (phase !== 'boot') return;

    const timers: NodeJS.Timeout[] = [];

    // Frame transitions
    FRAMES.forEach((_, index) => {
      if (index > 0) {
        timers.push(setTimeout(() => setFrame(index), frameDuration * index));
      }
    });

    // Start blinking after reaching final frame
    const blinkStartTime = frameDuration * (FRAMES.length - 1) + 500;
    let blinkInterval: NodeJS.Timeout;

    timers.push(
      setTimeout(() => {
        const blink = () => {
          setIsBlinking(true);
          setTimeout(() => setIsBlinking(false), 150);
        };
        blink();
        blinkInterval = setInterval(blink, 3000);
      }, blinkStartTime)
    );

    // Fade out and complete
    const fadeOutTime = frameDuration * (FRAMES.length - 1) + holdDuration;
    timers.push(
      setTimeout(() => {
        if (blinkInterval) clearInterval(blinkInterval);
        setPhase('fadeout');
        setTimeout(() => {
          setPhase('complete');
          onComplete?.();
        }, 600);
      }, fadeOutTime)
    );

    return () => {
      timers.forEach(clearTimeout);
      if (blinkInterval) clearInterval(blinkInterval);
    };
  }, [phase, frameDuration, holdDuration, onComplete]);

  // Restart function for demo
  const restart = useCallback(() => {
    setFrame(0);
    setPhase('boot');
    setIsBlinking(false);
  }, []);

  const currentFrame = FRAMES[frame];

  return (
    <div style={styles.container}>
      {/* Boot Screen */}
      <div
        style={{
          ...styles.bootScreen,
          opacity: phase === 'fadeout' ? 0 : 1,
          pointerEvents: phase === 'complete' ? 'none' : 'auto',
        }}
      >
        <div style={styles.bootContainer}>
          {/* Progress Dots */}
          <div style={styles.progressDots}>
            {FRAMES.map((_, i) => (
              <div
                key={i}
                style={{
                  ...styles.progressDot,
                  ...(i === frame ? styles.progressDotActive : {}),
                  ...(i < frame ? styles.progressDotComplete : {}),
                }}
              />
            ))}
          </div>

          {/* Kelly Frame Container */}
          <div style={styles.frameContainer}>
            {/* Glow ring on final frame */}
            <div
              style={{
                ...styles.glowRing,
                opacity: frame === FRAMES.length - 1 ? 1 : 0,
                animation: frame === FRAMES.length - 1 ? 'pulseRing 2s ease-in-out infinite' : 'none',
              }}
            />

            {/* Frame images */}
            {FRAMES.map((f, i) => (
              <div
                key={f.id}
                style={{
                  ...styles.frame,
                  backgroundImage: `url(${f.image})`,
                  opacity: i === frame ? 1 : 0,
                  animation: i === frame && i === FRAMES.length - 1 ? 'breathe 4s ease-in-out infinite' : 'none',
                }}
              />
            ))}

            {/* Blink overlay - simulates eye blink */}
            <div
              style={{
                ...styles.blinkOverlay,
                opacity: isBlinking ? 1 : 0,
              }}
            />

            {/* Sparkles */}
            {frame === FRAMES.length - 1 && (
              <>
                <Sparkle top="20%" left="15%" delay={0} />
                <Sparkle top="30%" right="20%" delay={0.5} />
                <Sparkle top="60%" left="25%" delay={1} />
                <Sparkle top="45%" right="15%" delay={1.5} />
              </>
            )}
          </div>

          {/* Frame Label */}
          <div style={styles.frameLabel}>
            <h3 style={styles.frameLabelTitle}>{currentFrame.title}</h3>
            <p style={styles.frameLabelSubtitle}>Frame {frame + 1} of {FRAMES.length}</p>
          </div>

          {/* Loading Indicator */}
          <div style={styles.loadingIndicator}>
            {frame < FRAMES.length - 1 && (
              <div style={styles.loadingDots}>
                <LoadingDot delay={0} />
                <LoadingDot delay={0.2} />
                <LoadingDot delay={0.4} />
              </div>
            )}
            <div
              style={{
                ...styles.loadingText,
                color: frame === FRAMES.length - 1 ? '#d97757' : '#737373',
                fontWeight: frame === FRAMES.length - 1 ? 500 : 400,
              }}
            >
              {currentFrame.loadingText}
            </div>
          </div>
        </div>
      </div>

      {/* Restart Button (for demo) */}
      <button style={styles.restartBtn} onClick={restart}>
        ↺ Restart
      </button>

      {/* Debug Panel */}
      {debug && (
        <div style={styles.debugPanel}>
          <span style={styles.debugLabel}>Frame:</span> {frame} |{' '}
          <span style={styles.debugLabel}>Phase:</span> {phase}
        </div>
      )}

      {/* Keyframe animations */}
      <style>{`
        @keyframes breathe {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.008) translateY(-2px); }
        }
        @keyframes pulseRing {
          0%, 100% { transform: scale(1); opacity: 0.5; }
          50% { transform: scale(1.02); opacity: 1; }
        }
        @keyframes sparkle {
          0%, 100% { opacity: 0; transform: scale(0); }
          50% { opacity: 1; transform: scale(1); }
        }
        @keyframes loadingBounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40% { transform: translateY(-8px); opacity: 1; }
        }
      `}</style>
    </div>
  );
}

// Sparkle component
function Sparkle({ top, left, right, delay }: { top: string; left?: string; right?: string; delay: number }) {
  return (
    <div
      style={{
        position: 'absolute',
        top,
        left,
        right,
        width: 4,
        height: 4,
        background: 'white',
        borderRadius: '50%',
        animation: `sparkle 2s ease-in-out ${delay}s infinite`,
      }}
    />
  );
}

// Loading dot component
function LoadingDot({ delay }: { delay: number }) {
  return (
    <div
      style={{
        width: 6,
        height: 6,
        borderRadius: '50%',
        background: '#737373',
        animation: `loadingBounce 1.4s ease-in-out ${delay}s infinite`,
      }}
    />
  );
}

// Styles
const styles: Record<string, React.CSSProperties> = {
  container: {
    position: 'relative',
    width: '100%',
    height: '100vh',
    background: 'white',
    overflow: 'hidden',
  },
  bootScreen: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(180deg, #fafafa 0%, #f0f0f0 100%)',
    zIndex: 50,
    transition: 'opacity 600ms ease-out',
  },
  bootContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    position: 'relative',
  },
  progressDots: {
    display: 'flex',
    gap: 12,
    marginBottom: 32,
  },
  progressDot: {
    width: 10,
    height: 10,
    borderRadius: '50%',
    background: '#e5e5e5',
    transition: 'all 400ms cubic-bezier(0.34, 1.56, 0.64, 1)',
  },
  progressDotActive: {
    background: '#d97757',
    transform: 'scale(1.3)',
    boxShadow: '0 0 20px rgba(217, 119, 87, 0.4)',
  },
  progressDotComplete: {
    background: '#7BA7C2',
  },
  frameContainer: {
    position: 'relative',
    width: 320,
    height: 400,
    borderRadius: 32,
    overflow: 'hidden',
    boxShadow: '0 25px 80px rgba(0, 0, 0, 0.12), 0 10px 30px rgba(0, 0, 0, 0.08), inset 0 0 0 1px rgba(255, 255, 255, 0.5)',
    background: 'white',
  },
  glowRing: {
    position: 'absolute',
    inset: -20,
    borderRadius: 52,
    border: '2px solid rgba(217, 119, 87, 0.3)',
    transition: 'opacity 600ms',
    pointerEvents: 'none',
  },
  frame: {
    position: 'absolute',
    inset: 0,
    backgroundSize: 'cover',
    backgroundPosition: 'center top',
    backgroundRepeat: 'no-repeat',
    transition: 'opacity 500ms ease-out',
  },
  blinkOverlay: {
    position: 'absolute',
    inset: 0,
    background: 'linear-gradient(180deg, transparent 0%, transparent 15%, rgba(255, 255, 255, 0.9) 18%, rgba(255, 255, 255, 0.9) 25%, transparent 28%, transparent 100%)',
    transition: 'opacity 100ms',
    pointerEvents: 'none',
  },
  frameLabel: {
    marginTop: 28,
    textAlign: 'center',
  },
  frameLabelTitle: {
    fontSize: 18,
    color: '#262626',
    fontWeight: 600,
    marginBottom: 4,
  },
  frameLabelSubtitle: {
    fontSize: 13,
    color: '#737373',
    margin: 0,
  },
  loadingIndicator: {
    marginTop: 40,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 12,
    minHeight: 40,
  },
  loadingDots: {
    display: 'flex',
    gap: 6,
  },
  loadingText: {
    fontSize: 12,
    fontFamily: "'SF Mono', 'Fira Code', monospace",
    letterSpacing: 0.5,
    transition: 'color 300ms',
  },
  restartBtn: {
    position: 'absolute',
    bottom: 24,
    right: 24,
    zIndex: 100,
    padding: '12px 20px',
    background: 'rgba(38, 38, 38, 0.9)',
    color: 'white',
    fontSize: 14,
    fontWeight: 500,
    borderRadius: 12,
    border: '1px solid rgba(255, 255, 255, 0.1)',
    cursor: 'pointer',
  },
  debugPanel: {
    position: 'absolute',
    bottom: 24,
    left: 24,
    zIndex: 100,
    background: 'rgba(0, 0, 0, 0.8)',
    padding: '16px 20px',
    borderRadius: 12,
    fontFamily: "'SF Mono', 'Fira Code', monospace",
    fontSize: 12,
    color: '#a3a3a3',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  debugLabel: {
    color: '#d97757',
    marginRight: 8,
  },
};
