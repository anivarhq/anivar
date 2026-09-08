/** Single source of truth for the user's preferred playback volume (0..1).
 *  Every player in the app (clip overlay, audio-event modal) reads the same
 *  key, so setting the level once sets it everywhere — and it survives
 *  restarts. Zero is never saved: mute is a state, not a level. */
const VOL_KEY = "sc.playerVolume";

export function loadSavedVolume(): number {
  const n = Number(localStorage.getItem(VOL_KEY) ?? "");
  return Number.isFinite(n) && n > 0 && n <= 1 ? n : 0.8;
}

export function saveVolume(v: number) {
  if (Number.isFinite(v) && v > 0 && v <= 1) localStorage.setItem(VOL_KEY, String(v));
}

/** Muted state is global too: unmute a player once and every future player
 *  opens with sound at the saved level. Default (no key yet) = muted, so the
 *  out-of-box experience never blasts audio on first autoplay. */
const MUTE_KEY = "sc.playerMuted";

export function loadSavedMuted(): boolean {
  return localStorage.getItem(MUTE_KEY) !== "0";
}

export function saveMuted(m: boolean) {
  localStorage.setItem(MUTE_KEY, m ? "1" : "0");
}
