import { describe, expect, it } from "vitest";
import { SKILL_REGISTRY } from "./skillDownload";

describe("Arsenal catalog", () => {
  it("names the licence of every model, at the point of choice", () => {
    for (const s of SKILL_REGISTRY) expect(s.license, s.id).toBeTruthy();
  });

  it("pins every file to a revision and a SHA-256", () => {
    for (const s of SKILL_REGISTRY) {
      expect(s.files.length, s.id).toBeGreaterThan(0);
      for (const f of s.files) {
        const where = `${s.id}/${f.filename}`;
        expect(f.sha256, where).toMatch(/^[0-9a-f]{64}$/);
        // A moving branch can change what users get without notice.
        expect(f.url, where).not.toMatch(/\/(resolve\/main|master)\//);
      }
    }
  });
});
