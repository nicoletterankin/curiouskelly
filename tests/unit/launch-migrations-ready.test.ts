import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";

describe("Launch DB migrations", () => {
  it("contains required migrations for launch tracking + purchases + gifts", () => {
    const migrationsDir = path.resolve(process.cwd(), "..", "supabase", "migrations");
    const names = new Set(fs.readdirSync(migrationsDir));

    // Foundation for launch readiness (audit trail + purchases + community + live + downloads + gift codes)
    const required = [
      "025_user_events_audit_trail.sql",
      "026_lesson_purchases.sql",
      "027_users_lifetime_tracking.sql",
      "028_community_features.sql",
      "029_live_classes.sql",
      "030_lesson_downloads.sql",
      "031_gift_codes.sql",
    ];

    const missing = required.filter((f) => !names.has(f));
    expect(missing, `Missing migrations: ${missing.join(", ")}`).toEqual([]);
  });
});

