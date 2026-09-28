import { describe, it, expect, vi } from "vitest";

vi.mock("@tauri-apps/plugin-opener", () => ({ openUrl: vi.fn(() => Promise.resolve()) }));
import { externalHref } from "./openExternal";

describe("externalHref", () => {
  it("sends https links that open a new window to the browser", () => {
    expect(externalHref({ href: "https://tailscale.com/download", target: "_blank" }))
      .toBe("https://tailscale.com/download");
  });

  it("leaves links that stay in the app alone", () => {
    expect(externalHref({ href: "https://example.com", target: "" })).toBeNull();
    expect(externalHref({ href: "http://localhost:8882/clip/1", target: "_blank" })).toBeNull();
    expect(externalHref({ href: "javascript:alert(1)", target: "_blank" })).toBeNull();
    expect(externalHref(null)).toBeNull();
  });
});
