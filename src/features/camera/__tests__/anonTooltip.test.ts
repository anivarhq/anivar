import { describe, expect, it } from "vitest";
import { depthModelSkill } from "../../agent/skillDownload";
import { anonymizeTooltip } from "../anonTooltip";

describe("anonymizeTooltip", () => {
  it("uses the depth catalog size and never a hard-coded one", () => {
    const { sizeLabel } = depthModelSkill();
    expect(anonymizeTooltip(false)).toContain(sizeLabel);
    expect(anonymizeTooltip(false)).not.toMatch(/95\s*MB/);
  });

  it("still describes the ON state", () => {
    expect(anonymizeTooltip(true)).toContain("Depth Anonymization ON");
  });
});
