# Deployment Checklist: Universal Login & Power-Up

Use this checklist to deploy the new Authentication and Neural Link features to production.

## 1. Supabase Configuration

### 1.1 Enable Facebook (Meta) Login
1.  Go to **Authentication > Providers** in your Supabase Dashboard.
2.  Select **Facebook**.
3.  Enter your **Facebook Client ID** and **Client Secret** (from developers.facebook.com).
4.  Set the **Callback URL (for OAuth)** to: `https://<PROJECT_REF>.supabase.co/auth/v1/callback`.
5.  **CRITICAL:** Toggle "Enable" to **ON**.

### 1.2 Enable OpenAI Login (OIDC)
*Note: OpenAI uses the standard OpenID Connect (OIDC) provider.*
1.  Go to **Authentication > Providers**.
2.  Select **OpenID Connect** (if available) or configure a **Custom Provider**.
3.  **Issuer URL:** `https://auth0.openai.com` (Verify current OpenAI OIDC issuer endpoint).
4.  **Client ID / Secret:** Enter credentials from your OpenAI Platform dashboard.
5.  **Scopes:** Ensure `openid`, `profile`, `email` are requested.

### 1.3 URL Redirects
1.  Go to **Authentication > URL Configuration**.
2.  Add the following to **Redirect URLs**:
    *   `https://curiouskelly.com/dashboard.html`
    *   `http://localhost:3000/dashboard.html` (for local testing)

---

## 2. Database Migration

The `User` table schema has changed. You must push these changes to the production database.

```bash
# 1. Validate schema locally
npx prisma validate

# 2. Push changes to production (Warning: Check for breaking changes first)
npx prisma db push

# 3. Verify columns exist
# Connect to DB and run:
# SELECT column_name FROM information_schema.columns WHERE table_name = 'User';
# Check for: 'interestProfile', 'connectedProviders'
```

---

## 3. Meta App Review Guide (Privacy Power-Up)

To get access to `user_likes`, you must submit for App Review. Use this text:

**Permission:** `user_likes`

**How we use this data:**
> "Curious Kelly is an AI-driven educational platform. We use the `user_likes` permission to programmatically personalize educational analogies. For example, if a user likes 'gardening' pages, our AI Lesson Generator will explain complex topics (like 'Consistency') using gardening metaphors (e.g., 'watering habits'). We do not store the raw like data permanently; we convert it into a generic 'Interest Vector' and discard the raw identifiers immediately."

**Screencast Script:**
1.  **Start:** Show the User Dashboard.
2.  **Action:** Click "Connect Facebook" in the "Power Up Kelly" section.
3.  **Flow:** Show the Facebook Login popup.
4.  **Result:** Show the UI changing to "Connected".
5.  **Payoff:** Navigate to a Lesson page and show a generated lesson that mentions a topic related to the user's likes (mocked if necessary).

---

## 4. Testing Protocol

### 4.1 Universal Login Smoke Test
- [ ] Click "Continue with Facebook" on Landing Page.
- [ ] Verify redirect to Facebook.
- [ ] Verify return to `dashboard.html`.
- [ ] Verify `user-name` is populated correctly.

### 4.2 Neural Link State
- [ ] On Dashboard, click "Connect" for Meta.
- [ ] Verify button changes to "Connecting...".
- [ ] Verify button changes to "Disconnect" (Simulated success in `neural-link.js` until API is live).

### 4.3 Analogy Engine (Manual Test)
- [ ] Copy the content of `prompts/KELLY_ANALOGY_ENGINE.md`.
- [ ] Paste into ChatGPT/Claude.
- [ ] Input: `Topic: "Rust Memory Safety", Interest: "Cooking", Tone: "Fun"`.
- [ ] **Pass Criteria:** Output is a valid JSON with a cooking metaphor about memory safety (e.g., "Don't double-dip the spoon!").
























