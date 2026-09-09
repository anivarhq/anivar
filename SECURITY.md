# Security Policy

Anivar is a security camera system that processes video, biometric data and
network credentials on a user's own machine. Bugs here have real consequences,
and reports are genuinely welcome.

## Reporting a vulnerability

**Do not open a public issue for a security bug.**

Use GitHub's private reporting instead:
[**Report a vulnerability**](https://github.com/anivarhq/anivar/security/advisories/new)
— it is private to the maintainers, and it lets us prepare a fix before the
details are public.

What helps, roughly in order of usefulness:

- What an attacker gets — read footage, run code, escape the LAN, recover a
  biometric descriptor
- The steps to reproduce, and the version (Help → About, or `tauri.conf.json`)
- Whether it needs local access, an authenticated session, or nothing at all
- A patch, if you have one — very welcome, never expected

### What to expect

| | |
|---|---|
| First reply | Within 7 days |
| Assessment | Within 14 days, with a severity and a rough timeline |
| Fix | Critical issues first; a release, then the advisory |
| Credit | Named in the advisory unless you'd rather not be |

This is a small project without a paid security team. There is **no bug bounty**
— what there is instead is an honest reply, a real fix, and credit.

## Scope

**In scope** — the application and everything it ships:

- Remote code execution, path traversal, SQL injection
- Authentication or authorisation bypass (login gate, share links, remote access,
  the local media server)
- **Anything that leaks raw frames from a camera marked depth-anonymised** — that
  setting is a privacy guarantee, and a leak defeats it
- **Anything that recovers or exfiltrates biometric data** (face descriptors,
  body-appearance vectors) — see [PRIVACY.md](PRIVACY.md)
- Recovery of encrypted secrets (API keys, bot tokens) from disk
- SSRF via the camera proxy or ONVIF discovery
- Command injection through camera URLs, filenames, or agent input
- **Prompt injection that makes the agent act rather than answer** — the agent
  ingests untrusted input (Telegram messages, VLM output, camera names) and is
  monitor-only by design; anything that gets it to control the app, exfiltrate
  data, or reach a tool it should not is in scope
- Tampering with the update channel or a bundled model download

**Out of scope:**

- Attacks needing physical access to an unlocked machine. Local admin is game
  over on any desktop application
- **An unencrypted database on an unencrypted disk.** Known and documented —
  SQLite is plaintext apart from secrets; full-disk encryption is the answer, and
  PRIVACY.md says so. A *new* way to read it remotely is very much in scope
- Vulnerabilities in a camera's own firmware. Report those to its vendor
- Missing hardening with no exploit path (a header, a compiler flag). Fine as a
  normal issue
- Denial of service by exhausting the machine's own resources
- Anything requiring a user to paste attacker-supplied code into their own
  console
- Automated scanner output with no demonstrated impact

## Supported versions

| Version | Supported |
|---|---|
| latest release | ✅ |
| `main` | ✅ |
| anything older | ❌ — update first |

Fixes land on `main` and go out in the next release. While the project is at
`0.x`, only the newest release is patched: there is no back-porting and there
are no long-term support branches. Installed builds update themselves, so
"update first" is usually a restart.

## What the design already assumes

Useful context if you are looking for something:

- **The agent is monitor-only by design.** It reads, summarises and shares; it
  has no app-control tools. That is a hard constraint, not a current limitation,
  because it consumes untrusted text
- **Every user-derived value in a query is a bound parameter**, never string
  interpolation. A place where that is not true is a bug and worth reporting
- **The local media server binds loopback unless you opt in.** `lan_access`
  defaults to OFF, so a fresh install answers only on `127.0.0.1`. Turning it on
  (Settings → Sharing & tunnel) listens on all interfaces so other devices on
  your network can reach it. Remote viewing does not depend on it — Tailscale
  Funnel proxies to loopback. Either way requests carry a token, compared in
  constant time and rate-limited per IP, and paths are validated. The camera
  proxy is restricted to private ranges with DNS-rebinding blocked
- **Footage wipes deliberately exclude face data**, so a retention purge cannot
  silently destroy enrolments
- **Bundled binaries are SHA-pinned** (ffmpeg, go2rtc, runtime libraries).
  Downloads verify before install and install atomically
- Secrets are AES-GCM encrypted at rest; the login gate is Argon2id with
  constant-time comparison
- CI runs `cargo audit` and `npm audit` against production dependencies on every
  push, and fails on high severity

## Third-party dependencies

Advisories in upstream crates or packages are best reported upstream first. If an
advisory is unfixed upstream and Anivar's use of it is exploitable, report it
here too — how a dependency is *used* is our problem.

Dependabot and the CI audit job cover routine version bumps; those do not need a
private report.
