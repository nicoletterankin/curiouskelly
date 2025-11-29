# Universal Login & Kelly Power-Up (Neural Link)

## 1. Executive Summary

This feature implements a **"Two-Tier" Authentication Strategy** designed to balance frictionless onboarding with deep, privacy-first personalization.

*   **Tier 1: Universal Login (The Front Door)**
    *   **Goal:** Instant access.
    *   **Mechanism:** One-click OAuth via Facebook, OpenAI, Apple, or Google.
    *   **Permissions:** Minimal (Identity only: name, email, avatar).
    *   **User Value:** Zero friction, no scary permission screens.

*   **Tier 2: The Neural Link (Power Up Kelly)**
    *   **Goal:** Context-aware teaching.
    *   **Mechanism:** Explicit, post-onboarding connections via the Dashboard.
    *   **Permissions:** "Read-only" access to interests (e.g., `user_likes`).
    *   **User Value:** Kelly uses metaphors from the user's actual life (e.g., explaining "Consistency" using "Minecraft" mechanics).

---

## 2. Architecture

The system bridges the gap between raw user data (Social Graph) and educational content (Lesson Engine) without permanently storing sensitive raw data.

```mermaid
graph TD
    User[User] -->|1. Universal Login| Client[Web Client]
    Client -->|2. Connect Provider| NeuralLink[Neural Link Module]
    
    subgraph "Privacy Barrier (The Airlock)"
        NeuralLink -->|3. Request Permissions| MetaAPI[Meta/OpenAI APIs]
        MetaAPI -->|4. Raw Data (Likes/Hobbies)| Processor[Data Processor]
        Processor -->|5. Extract Interest Vector| DB[(User Database)]
        style MetaAPI fill:#f9f,stroke:#333,stroke-dasharray: 5 5
    end
    
    DB -->|6. User Interest Profile| LessonEngine[Lesson Generator]
    Analogy[Analogy Engine Prompt] -->|7. Metaphor Generation| LessonEngine
    LessonEngine -->|8. Personalized Lesson| User
```

### Key Components

| Component | File Path | Role |
| :--- | :--- | :--- |
| **Frontend Logic** | `public/js/neural-link.js` | Manages the UI state for connecting/disconnecting providers in the Dashboard. |
| **Auth Helpers** | `public/js/auth.js` | Handles the actual OAuth handshakes with Supabase and providers. |
| **AI Logic** | `prompts/KELLY_ANALOGY_ENGINE.md` | The System Prompt that translates "Gardening" (Interest) into "Growth Mindset" (Lesson Metaphor). |
| **Data Model** | `prisma/schema.prisma` | Stores the sanitized `interestProfile` and connection status. |

---

## 3. The Privacy Model: "Data Airlock"

To honor our "Privacy First" promise, we utilize a **Data Airlock** strategy:

1.  **Ephemeral Access:** When a user connects a provider (e.g., Meta), we fetch their likes *once*.
2.  **Vectorization:** An LLM analyzes the raw likes (e.g., "SpaceX page", "Neil deGrasse Tyson page") and converts them into a generic Interest Vector (e.g., `{"Interest": "Astronomy", "Tone": "Scientific"}`).
3.  **Discard:** The raw list of pages/likes is discarded. We do **not** store the user's specific Facebook history.
4.  **Transparency:** Users can view and delete their Interest Vector at any time from the Dashboard.

---

## 4. Implementation Details

### 4.1 The Analogy Engine
The core of the personalization is the **Analogy Engine**. It takes a generic lesson topic and "wraps" it in a user-specific metaphor.

*   **Input:** Topic: "Compound Interest", Interest: "Running"
*   **Output:** "Compound interest is like negative splits. You start slow, but the energy you save early on multiplies your speed in the final mile."

### 4.2 Database Schema
We added two JSON fields to the `User` model:
*   `interestProfile`: Stores the extracted metaphors/tags.
*   `connectedProviders`: Tracks which services are currently linked (boolean flags).

```prisma
model User {
  // ... existing fields
  interestProfile   Json?  // e.g. { "topics": ["Sci-Fi", "Coding"], "style": "Witty" }
  connectedProviders Json? // e.g. { "meta": true, "openai": false }
}
```








