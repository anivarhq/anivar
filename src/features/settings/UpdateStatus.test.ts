import { describe, expect, it } from "vitest";
import { releaseHighlights } from "./UpdateStatus";

// The shape of a real manifest: release-page boilerplate, then the changelog.
const NOTES = `## Install

Download the installer for your platform below.

- **Windows** shows "Windows protected your PC"

## Changes

The first release you can install from inside the app.

### Added

- **Home Assistant and MQTT.** Point Anivar at an MQTT broker, with:
  - a motion sensor
- **Turn remote access off, and see what you've shared.** Settings now has it.

### Fixed

- **Exports that start while the camera was off begin where footage
  resumes**, instead of skipping into the recording.
- A plain bullet with \`code\`
`;

describe("releaseHighlights", () => {
  it("keeps only the changelog's top-level bullets, by section", () => {
    expect(releaseHighlights(NOTES)).toEqual([
      { heading: "New", items: ["Home Assistant and MQTT", "Turn remote access off, and see what you've shared"] },
      { heading: "Fixed", items: ["Exports that start while the camera was off begin where footage resumes", "A plain bullet with code"] },
    ]);
  });

  it("returns nothing for notes without bullets", () => {
    expect(releaseHighlights("")).toEqual([]);
    expect(releaseHighlights("Bug fixes.")).toEqual([]);
  });
});
