import { openUrl } from "@tauri-apps/plugin-opener";

/** Open a web link in the user's default browser.
 *
 *  Inside the app window, `window.open` and `<a target="_blank">` do nothing:
 *  the webview has no new-window handler, so WebView2 marks every such request
 *  handled and drops it (wry `add_NewWindowRequested` → `SetHandled(true)`).
 *  That is why "Download update", the Tailscale link and the assistant's links
 *  never opened. The opener plugin hands the URL to the OS instead; its
 *  capability allows https only. */
export function openExternal(url: string): void {
  openUrl(url).catch((e) => console.warn("could not open link", url, e));
}

/** The URL to hand to the browser for a click on this link, or null to let
 *  the webview handle it (in-app links, anchors, anything not https). */
export function externalHref(a: { href: string; target: string } | null): string | null {
  if (!a || a.target !== "_blank") return null;
  return /^https:\/\//i.test(a.href) ? a.href : null;
}

/** One document-level listener, so every `<a target="_blank">` in the app —
 *  including ones rendered later, like links in assistant replies — opens in
 *  the browser without each component having to remember to. */
export function routeExternalLinks(): () => void {
  const onClick = (e: MouseEvent) => {
    if (e.defaultPrevented || e.button !== 0) return;
    const a = (e.target as Element | null)?.closest?.("a[href]") as HTMLAnchorElement | null;
    const url = externalHref(a);
    if (!url) return;
    e.preventDefault();
    openExternal(url);
  };
  document.addEventListener("click", onClick);
  return () => document.removeEventListener("click", onClick);
}
