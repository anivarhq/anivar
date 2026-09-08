import { useState } from "react";
import { Shield, Lock, KeyRound, LifeBuoy } from "lucide-react";
import { api } from "../../api";

const REMEMBER_KEY = "sc-remember-token";

type Mode = "password" | "otp" | "recovery";

/** Full-screen lock screen shown when `login_required` and the app is locked.
 *  Password → (optional Telegram 2FA) → unlock. "Forgot password" delivers a
 *  recovery code to Telegram, then lets the user set a new password. */
export function LoginGate({ rememberEnabled, onUnlocked }: {
  rememberEnabled: boolean;
  onUnlocked: () => void;
}) {
  const [mode, setMode] = useState<Mode>("password");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [newPw, setNewPw] = useState("");
  const [remember, setRemember] = useState(false);
  const [challenge, setChallenge] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const finishUnlock = (token: string | null) => {
    if (token) localStorage.setItem(REMEMBER_KEY, token);
    onUnlocked();
  };

  const submitPassword = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await api.login(password, remember);
      if (r.status === "Unlocked") finishUnlock(r.remember_token);
      else { setChallenge(r.challenge); setMode("otp"); setInfo("A login code was sent to your Telegram."); }
    } catch (e) { setErr(String(e).replace(/^.*Error:\s*/, "")); }
    finally { setBusy(false); }
  };

  const submitOtp = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await api.loginVerifyOtp(challenge, code, remember);
      if (r.status === "Unlocked") finishUnlock(r.remember_token);
    } catch (e) { setErr(String(e).replace(/^.*Error:\s*/, "")); }
    finally { setBusy(false); }
  };

  const startRecovery = async () => {
    setBusy(true); setErr(null); setInfo(null);
    try {
      const ch = await api.requestRecovery();
      setChallenge(ch); setMode("recovery"); setCode(""); setNewPw("");
      setInfo("A recovery code was sent to your Telegram. Enter it and choose a new password.");
    } catch (e) { setErr(String(e).replace(/^.*Error:\s*/, "")); }
    finally { setBusy(false); }
  };

  const submitRecovery = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await api.recoveryReset(challenge, code, newPw, remember);
      if (r.status === "Unlocked") finishUnlock(r.remember_token);
    } catch (e) { setErr(String(e).replace(/^.*Error:\s*/, "")); }
    finally { setBusy(false); }
  };

  const input: React.CSSProperties = {
    width: "100%", padding: "11px 13px", borderRadius: 12, fontSize: 14,
    border: "1px solid var(--border-strong)", background: "rgb(var(--ink) / 0.05)",
    color: "var(--text-primary)", outline: "none",
  };
  const primary: React.CSSProperties = {
    width: "100%", padding: "11px 0", borderRadius: 12, fontSize: 14, fontWeight: 700,
    cursor: busy ? "default" : "pointer", opacity: busy ? 0.6 : 1,
    border: "none", background: "var(--accent-fill)", color: "var(--on-accent)",
    display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 7,
  };

  return (
    <div style={{
      position: "fixed", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
      background: "var(--bg, var(--bg-base))", padding: 24, zIndex: 5000,
    }}>
      <div className="glass" style={{ width: 380, maxWidth: "100%", padding: 26 }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8, marginBottom: 18 }}>
          <div style={{ width: 52, height: 52, borderRadius: 16, display: "flex", alignItems: "center",
            justifyContent: "center", background: "var(--accent-glow)", color: "var(--accent)" }}>
            <Shield size={26} />
          </div>
          <div style={{ fontWeight: 800, fontSize: 18 }}>Anivar</div>
          <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
            {mode === "password" && "Enter your password to unlock"}
            {mode === "otp" && "Enter the code from your Telegram"}
            {mode === "recovery" && "Reset your password"}
          </div>
        </div>

        {info && (
          <div style={{ fontSize: 11.5, color: "var(--accent)", background: "var(--accent-glow)",
            borderRadius: 10, padding: "8px 11px", marginBottom: 12, lineHeight: 1.45 }}>{info}</div>
        )}
        {err && (
          <div style={{ fontSize: 11.5, color: "var(--accent-red)", background: "rgba(255,69,58,0.1)",
            borderRadius: 10, padding: "8px 11px", marginBottom: 12, lineHeight: 1.45 }}>{err}</div>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
          {mode === "password" && (
            <>
              <input type="password" autoFocus value={password} placeholder="Password" style={input}
                onChange={e => setPassword(e.target.value)}
                onKeyDown={e => e.key === "Enter" && !busy && submitPassword()} />
              {rememberEnabled && <RememberRow remember={remember} setRemember={setRemember} />}
              <button style={primary} disabled={busy} onClick={submitPassword}><Lock size={15} /> Unlock</button>
              <button onClick={startRecovery} disabled={busy} style={linkBtn}>
                <LifeBuoy size={12} /> Forgot password? Recover via Telegram
              </button>
            </>
          )}

          {mode === "otp" && (
            <>
              <input inputMode="numeric" autoFocus value={code} placeholder="6-digit code"
                style={{ ...input, letterSpacing: 4, textAlign: "center", fontSize: 18 }}
                onChange={e => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                onKeyDown={e => e.key === "Enter" && !busy && submitOtp()} />
              {rememberEnabled && <RememberRow remember={remember} setRemember={setRemember} />}
              <button style={primary} disabled={busy} onClick={submitOtp}><KeyRound size={15} /> Verify</button>
              <button onClick={() => { setMode("password"); setErr(null); setInfo(null); }} style={linkBtn}>Back</button>
            </>
          )}

          {mode === "recovery" && (
            <>
              <input inputMode="numeric" autoFocus value={code} placeholder="Recovery code from Telegram"
                style={{ ...input, letterSpacing: 2, textAlign: "center" }}
                onChange={e => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))} />
              <input type="password" value={newPw} placeholder="New password (min 12 chars)" style={input}
                onChange={e => setNewPw(e.target.value)}
                onKeyDown={e => e.key === "Enter" && !busy && submitRecovery()} />
              {rememberEnabled && <RememberRow remember={remember} setRemember={setRemember} />}
              <button style={primary} disabled={busy} onClick={submitRecovery}><KeyRound size={15} /> Reset & unlock</button>
              <button onClick={() => { setMode("password"); setErr(null); setInfo(null); }} style={linkBtn}>Back</button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

const linkBtn: React.CSSProperties = {
  background: "transparent", border: "none", color: "var(--text-secondary)", fontSize: 11.5,
  cursor: "pointer", padding: "4px 0", display: "inline-flex", alignItems: "center",
  justifyContent: "center", gap: 5,
};

function RememberRow({ remember, setRemember }: { remember: boolean; setRemember: (v: boolean) => void }) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "var(--text-secondary)", cursor: "pointer" }}>
      <input type="checkbox" checked={remember} onChange={e => setRemember(e.target.checked)} />
      Remember this device
    </label>
  );
}
