/**
 * Launch Readiness Eval
 *
 * Live checks against a base URL (defaults to https://curiouskelly.com).
 * Covers landing pages + key APIs with edge cases.
 *
 * Run: pnpm eval
 */

type Json = Record<string, unknown>;

function getBaseUrl(): string {
  const env = process.env.BASE_URL?.trim();
  return env && env.length > 0 ? env.replace(/\/+$/, "") : "https://curiouskelly.com";
}

async function getText(url: string): Promise<{ status: number; text: string }> {
  const res = await fetch(url, { redirect: "follow" });
  const text = await res.text();
  return { status: res.status, text };
}

async function postJson(url: string, body: Json): Promise<{ status: number; json: Json }> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    redirect: "follow",
  });
  const json = (await res.json().catch(() => ({}))) as Json;
  return { status: res.status, json };
}

function assert(cond: unknown, msg: string) {
  if (!cond) throw new Error(msg);
}

export async function runEvals(): Promise<void> {
  const base = getBaseUrl();
  console.log(`\nLaunch readiness (BASE_URL=${base})\n`);

  // Landing pages must load
  const pages: Array<{ path: string; mustInclude?: string }> = [
    { path: "/", mustInclude: "Curious Kelly" },
    { path: "/learn.html", mustInclude: "Learn" },
    { path: "/pricing.html", mustInclude: "Pricing" },
    { path: "/gifts.html", mustInclude: "Gift" },
    { path: "/redeem.html", mustInclude: "Redeem" },
  ];

  for (const p of pages) {
    const url = `${base}${p.path}`;
    const { status, text } = await getText(url);
    assert(status >= 200 && status < 400, `Page ${p.path} returned ${status}`);
    if (p.mustInclude) {
      assert(text.includes(p.mustInclude), `Page ${p.path} missing expected text: ${p.mustInclude}`);
    }
    console.log(`✅ page ${p.path} (${status})`);
  }

  // APIs: events should never hard-fail
  {
    const { status, json } = await postJson(`${base}/api/events`, { event_type: "launch.eval", payload: { source: "eval" } });
    assert(status === 200, `/api/events expected 200, got ${status}`);
    assert(json.success === true, `/api/events expected success:true, got ${JSON.stringify(json)}`);
    console.log(`✅ api /api/events (${status})`);
  }

  // APIs: lesson purchase edge cases (Stripe may be configured or not)
  {
    const { status: okStatus } = await getText(`${base}/api/lesson-purchase?day=1`);
    assert(okStatus === 200 || okStatus === 400, `/api/lesson-purchase GET unexpected status ${okStatus}`);
    console.log(`✅ api /api/lesson-purchase GET day=1 (${okStatus})`);

    const { status: badStatus } = await getText(`${base}/api/lesson-purchase?day=0`);
    assert(badStatus === 400, `/api/lesson-purchase GET day=0 should be 400, got ${badStatus}`);
    console.log(`✅ api /api/lesson-purchase GET day=0 (${badStatus})`);

    const postInvalid = await postJson(`${base}/api/lesson-purchase`, { day_number: 0 });
    assert(postInvalid.status === 400, `/api/lesson-purchase POST invalid should be 400, got ${postInvalid.status}`);
    console.log(`✅ api /api/lesson-purchase POST invalid (${postInvalid.status})`);

    const postValid = await postJson(`${base}/api/lesson-purchase`, { day_number: 1 });
    assert([200, 503].includes(postValid.status), `/api/lesson-purchase POST expected 200 or 503, got ${postValid.status}`);
    console.log(`✅ api /api/lesson-purchase POST valid (${postValid.status})`);
  }

  // APIs: gift redeem edge cases
  {
    const invalidCode = await postJson(`${base}/api/gift-redeem`, { code: "ABC", email: "test@example.com" });
    assert(invalidCode.status === 400, `/api/gift-redeem invalid code should be 400, got ${invalidCode.status}`);
    console.log(`✅ api /api/gift-redeem invalid code (${invalidCode.status})`);

    const invalidEmail = await postJson(`${base}/api/gift-redeem`, { code: "ABCDEFGHJKLM", email: "not-an-email" });
    assert(invalidEmail.status === 400, `/api/gift-redeem invalid email should be 400, got ${invalidEmail.status}`);
    console.log(`✅ api /api/gift-redeem invalid email (${invalidEmail.status})`);
  }

  console.log(`\n✅ Launch readiness checks complete\n`);
}

