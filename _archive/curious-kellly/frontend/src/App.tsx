import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styled from 'styled-components';
import { 
  Play, Pause, Settings, Globe, Clock, MapPin, Sparkles, User, X 
} from 'lucide-react';

// -- Styled Components --

const Container = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: #000;
  overflow: hidden;
  font-family: 'Inter', sans-serif;
`;

const KellyLayer = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  background: linear-gradient(to bottom, #1a1a1a, #000); 
  display: flex;
  align-items: center;
  justify-content: center;
  color: #333;
  font-size: 2rem;
`;

const ControlRail = styled(motion.div)`
  position: absolute;
  top: 0;
  right: 0;
  height: 100%;
  width: 80px;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(20px);
  z-index: 10;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 2rem;
  gap: 1.5rem;
  border-left: 1px solid rgba(255, 255, 255, 0.1);
`;

const IconBtn = styled(motion.button)<{ $active?: boolean }>`
  background: ${(props) => (props.$active ? 'rgba(255,255,255,0.2)' : 'transparent')};
  border: none;
  color: white;
  width: 48px;
  height: 48px;
  border-radius: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  
  &:hover {
    background: rgba(255, 255, 255, 0.1);
  }

  /* Tooltip on hover could go here */
`;

const Panel = styled(motion.div)`
  position: absolute;
  top: 0;
  right: 80px; /* Next to rail */
  height: 100%;
  width: 320px;
  background: rgba(0, 0, 0, 0.9);
  backdrop-filter: blur(30px);
  z-index: 9;
  padding: 2rem;
  color: white;
  display: flex;
  flex-direction: column;
  gap: 2rem;
  border-left: 1px solid rgba(255, 255, 255, 0.1);
`;

const PanelHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  
  h2 {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0;
  }
`;

const Section = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
`;

const Label = styled.label`
  font-size: 0.875rem;
  color: #aaa;
  display: flex;
  justify-content: space-between;
`;

const SliderContainer = styled.div`
  position: relative;
  height: 40px; /* Space for labels */
`;

const Slider = styled.input`
  width: 100%;
  -webkit-appearance: none;
  height: 4px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 2px;
  outline: none;
  margin-top: 8px;
  
  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background: #fff;
    border-radius: 50%;
    cursor: pointer;
    transition: transform 0.1s;
  }
  
  &:active::-webkit-slider-thumb {
    transform: scale(1.2);
  }
`;

const ValuePreview = styled.span`
  font-size: 0.75rem;
  color: #00f2ff;
  font-weight: 600;
`;

// -- Data: Vibe Mappings --

const VIBE_MAP = {
  0: "The Survivor (Protective)",
  25: "The MacGyver (Practical)",
  50: "Balanced",
  75: "The Explorer (Curious)",
  100: "The Mystic (Deep)"
};

// -- Main Component --

export const App = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [activePanel, setActivePanel] = useState<string | null>(null);
  
  // State
  const [age, setAge] = useState(42); // Birth Year calculator would go here
  const [vibe, setVibe] = useState(50); 
  
  const togglePanel = (panel: string) => {
    if (activePanel === panel) setActivePanel(null);
    else setActivePanel(panel);
  };

  const getVibeLabel = (val: number) => {
    // Simple bucketing for demo
    if (val < 20) return "Survivor";
    if (val < 40) return "MacGyver";
    if (val < 60) return "Balanced";
    if (val < 80) return "Explorer";
    return "Mystic";
  };

  return (
    <Container>
      <KellyLayer>
        <h1>Kelly (No UI Mode)</h1>
      </KellyLayer>

      {/* -- THE PANELS -- */}
      <AnimatePresence>
        
        {/* 1. IDENTITY PANEL (Age/Clock) */}
        {activePanel === 'identity' && (
          <Panel
            key="identity"
            initial={{ x: 50, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 50, opacity: 0 }}
          >
            <PanelHeader>
              <h2>Identity</h2>
              <X size={24} cursor="pointer" onClick={() => setActivePanel(null)} />
            </PanelHeader>
            
            <Section>
              <Label>Your Age</Label>
              <input 
                type="number" 
                value={age} 
                onChange={(e) => setAge(Number(e.target.value))}
                style={{ 
                  background: 'rgba(255,255,255,0.1)', 
                  border: 'none', 
                  padding: '10px', 
                  color: 'white', 
                  fontSize: '1.2rem',
                  borderRadius: '8px'
                }} 
              />
              <Label>Born: {2025 - age}</Label>
            </Section>
          </Panel>
        )}

        {/* 2. VIBE PANEL (User Icon) */}
        {activePanel === 'vibe' && (
          <Panel
            key="vibe"
            initial={{ x: 50, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 50, opacity: 0 }}
          >
            <PanelHeader>
              <h2>The Vibe</h2>
              <X size={24} cursor="pointer" onClick={() => setActivePanel(null)} />
            </PanelHeader>
            
            <Section>
              <Label>
                <span>Perspective</span>
                <ValuePreview>{getVibeLabel(vibe)}</ValuePreview>
              </Label>
              <SliderContainer>
                <Slider 
                  type="range" 
                  min="0" 
                  max="100" 
                  value={vibe} 
                  onChange={(e) => setVibe(Number(e.target.value))} 
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#666', marginTop: '5px' }}>
                  <span>Gritty</span>
                  <span>Dreamy</span>
                </div>
              </SliderContainer>
              <p style={{ fontSize: '0.8rem', color: '#ccc', lineHeight: '1.4' }}>
                {vibe < 50 
                  ? "Focus on practical utility, survival, and concrete tools." 
                  : "Focus on wonder, exploration, and abstract connections."}
              </p>
            </Section>
          </Panel>
        )}

      </AnimatePresence>

      {/* -- THE CONTROL RAIL -- */}
      <ControlRail>
        {/* PLAY/PAUSE */}
        <IconBtn onClick={() => setIsPlaying(!isPlaying)}>
          {isPlaying ? <Pause size={24} fill="white" /> : <Play size={24} fill="white" />}
        </IconBtn>

        <div style={{ height: '1px', width: '40px', background: 'rgba(255,255,255,0.1)', margin: '10px 0' }} />

        {/* 1. LANGUAGE */}
        <IconBtn $active={activePanel === 'lang'} onClick={() => togglePanel('lang')}>
          <Globe size={24} />
        </IconBtn>

        {/* 2. IDENTITY (Age/Time) */}
        <IconBtn $active={activePanel === 'identity'} onClick={() => togglePanel('identity')}>
          <Clock size={24} />
        </IconBtn>

        {/* 3. LOCATION (Hemisphere) */}
        <IconBtn $active={activePanel === 'location'} onClick={() => togglePanel('location')}>
          <MapPin size={24} />
        </IconBtn>

        {/* 4. TONE (Sparkles) */}
        <IconBtn $active={activePanel === 'tone'} onClick={() => togglePanel('tone')}>
          <Sparkles size={24} />
        </IconBtn>

        {/* 5. VIBE (Archetype) */}
        <IconBtn $active={activePanel === 'vibe'} onClick={() => togglePanel('vibe')}>
          <User size={24} />
        </IconBtn>

      </ControlRail>
    </Container>
  );
};
