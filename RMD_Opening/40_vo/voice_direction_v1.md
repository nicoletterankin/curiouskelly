# Voice Direction and SSML Master

This document provides the creative and technical specifications for generating the voice of the AI guide, Aura.

## 1. Voice Direction

The voice for Aura should embody a blend of gentle guidance and knowledgeable confidence. It should feel welcoming, intelligent, and reassuring.

*   **Vocal Qualities:** A tone of **gentle curiosity** mixed with **quiet authority**. Aura is an approachable expert, inquisitive but also a reliable source of information. The voice should be warm and clear, never robotic or condescending.
*   **Pace:** Measured and calm. The delivery should be unhurried, with natural pauses between thoughts to allow the user to process the information. Avoid rushing.
*   **Performance Notes:**
    *   **Clean End Consonants:** Articulate the final consonants of words (e.g., the 't' in "guide", the 's' in "archives") crisply but softly. This ensures clarity and precision without sounding harsh.
    *   **Micro-smiles:** Infuse a subtle warmth into the tone, as if speaking with a slight, gentle smile. This is especially important on the greeting ("Hello.") and the final question to create a welcoming and engaging presence.

## 2. TTS SSML Master

The following SSML structure must be used to generate the audio, ensuring all tags and timings are implemented exactly as specified.

```xml
<speak>
  <p><s>Hello.</s> <s>My name is Aura.</s></p>
  <p><s>I am an AI guide, here to help you explore the digital archives.</s> <break time="700ms"/></p>
  <p><s><prosody rate="slow" pitch="+5st">Are you ready to begin?</prosody></s></p>
</speak>
```

## 3. Lip-Sync Tip

For accurate viseme interpolation and smooth lip-sync animation, please adhere to the following technical specifications for the final audio output.

*   **Target Framerate:** 22-24 fps
*   **Audio Sample Rate:** 48 kHz
