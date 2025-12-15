import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";

function listHtmlFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...listHtmlFiles(p));
      continue;
    }
    if (entry.isFile() && entry.name.toLowerCase().endsWith(".html")) {
      out.push(p);
    }
  }
  return out;
}

describe("Launch copy law", () => {
  it("does not use the word 'free' in marketing pages", () => {
    const publicDir = path.resolve(process.cwd(), "..", "public");
    const files = listHtmlFiles(publicDir);

    // Allow-list: legal docs can use legal language.
    const allow = [
      path.join(publicDir, "terms.html"),
      path.join(publicDir, "privacy.html"),
      path.join(publicDir, "legal"),
    ];

    const offenders: Array<{ file: string; matches: number }> = [];
    const re = /\bfree\b/gi;

    for (const file of files) {
      // skip allow-listed paths
      if (allow.some((a) => file === a || file.startsWith(a + path.sep))) continue;

      const contents = fs.readFileSync(file, "utf8");
      const matches = contents.match(re)?.length ?? 0;
      if (matches > 0) {
        offenders.push({ file: path.relative(publicDir, file), matches });
      }
    }

    expect(offenders, `Found forbidden word in: ${JSON.stringify(offenders, null, 2)}`).toEqual([]);
  });
});

