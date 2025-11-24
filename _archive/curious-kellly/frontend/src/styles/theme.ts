/* 
  Kelly - The "No UI" Education Platform
  Design System & Theme Definitions
*/

export const theme = {
  colors: {
    background: "#0a0a0a", // Deep space black
    text: "#ffffff",
    accent: "#00f2ff", // Cyan glow
    glass: "rgba(255, 255, 255, 0.1)",
    glassHover: "rgba(255, 255, 255, 0.2)",
  },
  typography: {
    fontFamily: "'Inter', sans-serif",
    sizes: {
      h1: "2.5rem",
      body: "1rem",
      micro: "0.75rem",
    },
  },
  layout: {
    sidebarWidth: "80px", // Collapsed width
    sidebarExpanded: "320px",
  },
  animation: {
    transition: "all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)",
  },
};

/* 
  Component Architecture:
  - <KellyCanvas />: The full-screen WebGL/Video layer
  - <ImmersiveOverlay />: The UI layer that sits on top
  - <ControlRail />: The right-side vertical menu (Play/Pause/Settings)
  - <VibeSliders />: The specific toggles for Archetypes
*/






