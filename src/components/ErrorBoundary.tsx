import React from "react";

/**
 * App-wide error boundary. Without one, a single runtime error in ANY component
 * unmounts the WHOLE React tree — a blank window while the backend keeps recording
 * (camera light on, no visible feed, "nothing works"). This catches the error,
 * keeps the app usable, and SHOWS what failed (with a reload) instead of a white
 * screen — so a UI bug degrades one render, not the entire appliance.
 */
interface State { error: Error | null; info: string | null }

export class ErrorBoundary extends React.Component<{ children: React.ReactNode }, State> {
  state: State = { error: null, info: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Surface to the console for devtools + keep the stack for the fallback UI.
    console.error("[Anivar] UI crash caught by ErrorBoundary:", error, info);
    this.setState({ info: info.componentStack ?? null });
  }

  render() {
    const { error, info } = this.state;
    if (!error) return this.props.children;
    return (
      <div style={{
        position: "fixed", inset: 0, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center", gap: 16, padding: 32,
        background: "#0b0f0d", color: "#e6e6e6", fontFamily: "system-ui, sans-serif",
        textAlign: "center", zIndex: 99999,
      }}>
        <div style={{ fontSize: 20, fontWeight: 800 }}>The interface hit an error</div>
        <div style={{ fontSize: 13, color: "#9aa0a6", maxWidth: 560, lineHeight: 1.6 }}>
          Recording is still running in the background — this is only the on-screen UI.
          Reload to recover.
        </div>
        <pre style={{
          maxWidth: "80vw", maxHeight: "40vh", overflow: "auto", textAlign: "left",
          background: "rgb(var(--ink) / 0.04)", border: "1px solid rgb(var(--ink) / 0.1)",
          borderRadius: 10, padding: "12px 14px", fontSize: 11.5, color: "var(--status-alert)",
        }}>
          {String(error?.stack || error?.message || error)}
          {info ? `\n\nComponent stack:${info}` : ""}
        </pre>
        <button
          onClick={() => { this.setState({ error: null, info: null }); window.location.reload(); }}
          style={{
            padding: "9px 22px", borderRadius: 999, border: "none", cursor: "pointer",
            fontSize: 13, fontWeight: 700, background: "var(--status-ok)", color: "var(--on-accent)",
          }}>
          Reload interface
        </button>
      </div>
    );
  }
}
