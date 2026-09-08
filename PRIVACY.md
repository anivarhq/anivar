# Privacy

Anivar runs on your own machine. There is no Anivar server, no account, and
no telemetry — nobody at this project can see your cameras, your footage, or your
data. What follows is what the software stores locally, what leaves the machine
and only when you configure it to, and how to delete any of it.

This document exists because Anivar processes **biometric data**. That is not
a detail to leave undocumented.

---

## The short version

| | |
|---|---|
| Where data lives | One folder on your machine. Nothing is uploaded by default |
| Telemetry / analytics | **None.** No usage reporting, no crash reporting, no phone-home |
| Account required | None |
| Biometrics | Face descriptors and body-appearance vectors, stored locally — **special-category data** |
| Leaves the machine | Only via features you turn on: cloud AI provider, Telegram, remote access, share links |
| Deleting it | Per person, per event, per camera, or wholesale — see [Deleting data](#deleting-data) |

---

## Biometric data — read this part

If you enable face recognition, Anivar computes and stores a **face
descriptor**: a numeric vector derived from the geometry of a face, used to tell
one person from another. If you enable person Re-ID, it also stores
**body-appearance vectors** (clothing colour and shape) to follow the same person
between cameras.

Under the **UK GDPR / EU GDPR (Article 9)**, biometric data processed *for the
purpose of uniquely identifying a person* is special-category data — the
strictest tier, alongside health and biometric identifiers. Equivalent rules
apply under **Illinois BIPA**, **Texas CUBI**, the **Colorado / Washington** health
and biometric statutes, and **CCPA/CPRA**, which classes biometric data as
sensitive personal information.

**What that means for you, the operator.** Anivar gives you the technical
controls. It cannot give you a lawful basis. If you record anyone other than
yourself and members of your household:

- **Purely personal / household use is generally exempt** from GDPR — a camera
  covering your own front door, viewed only by you.
- **The exemption does not survive the boundary.** Once cameras capture a shared
  hallway, a pavement, a neighbour's property, or anywhere the public goes, you
  are a data controller with the full set of obligations.
- **Employees and visitors are never household use.** A camera at a business
  needs a lawful basis, a retention policy, signage, and — for facial
  recognition specifically — usually a **Data Protection Impact Assessment**
  before you switch it on.
- **Some jurisdictions require prior written consent** for face recognition
  specifically. Illinois BIPA does, with a private right of action; that is why
  it is the statute most often litigated. Check your own before enrolling
  anybody.

Enrolling a person is an explicit act in Anivar — the software never enrols a
face on its own. That is deliberate: the decision, and the responsibility for it,
is yours.

### What is stored, exactly

| Data | Where | Why | Retention |
|---|---|---|---|
| Face descriptor (vector) | `face_embeddings` | Recognition | Until you delete the person, or the enrolment cap (30/person) rotates it |
| Aligned face crop (JPEG) | `blobs/` on disk, referenced from the DB | So you can see what was matched | With the descriptor; display crops capped at 500/person |
| Person name, role | `known_persons` | Labelling | Until deleted |
| Sightings log | `face_sightings` | "when was X last seen" | With the person |
| Body-appearance vector | `body_embeddings` | Cross-camera tracking | Unnamed bodies expire after **2 days**; named ones with the person |
| Hard negatives | `face_negatives`, `body_negatives` | "this is NOT X" corrections, so a fixed mistake stays fixed | With the person |
| Licence plates | `motion_events.recognized_plate` | ALPR | With the event |
| Trained classifier | `face_classifier_model` | Recognition speed | Rebuilt on any change |

Descriptors are **not reversible into a photograph**, but they are still
biometric identifiers and are treated as such here.

### Data you did not ask for

Faces that appear on camera but are not enrolled are still detected, and
unidentified crops are retained so you can label them later ("who is this?").
Unnamed body tracks expire automatically after two days. To stop unknown-face
retention entirely, turn face recognition off — detection of *people* as objects
continues without any biometric processing.

---

## What leaves your machine

Nothing, until you enable one of these. Each is off by default.

| Feature | What is sent | To whom |
|---|---|---|
| **Cloud AI provider** (OpenAI, Anthropic, Gemini, OpenRouter, …) | Event snapshots and a text prompt, for scene description | That provider, under their terms |
| **On-device AI** (default) | Nothing — the model runs inside the app | — |
| **Telegram** | Alert text, snapshots, and clips you or the agent send | Telegram |
| **Home Assistant / MQTT** | Event metadata | Your broker |
| **Remote access** (Tailscale Funnel) | Your live streams and UI, over an authenticated tunnel | Whoever holds the credential |
| **Share links** | The single clip you shared, until the link expires | Whoever has the link |

Two consequences worth stating plainly:

- **A cloud AI provider sees faces.** Snapshots sent for scene description
  contain whoever was in frame. If you are subject to GDPR and using a US
  provider, that is an international transfer of special-category data. Use the
  on-device model — it is the default for exactly this reason — or make sure your
  provider terms cover it.
- **Depth anonymisation is the mitigation.** Per camera, enable it and
  recordings, streams, snapshots and anything sent outward carry a depth map
  instead of a recognisable image. Raw frames stay in memory for detection and
  are never written or transmitted.

Model downloads (weights, ffmpeg, runtime libraries) fetch from Hugging Face and
GitHub. Those requests reveal your IP address to those hosts, like any download.
No Anivar-specific identifier is attached.

---

## Deleting data

| To delete | Where | What happens |
|---|---|---|
| **Everything about one person** | People → the person → **Remove** → confirm **erase** | Descriptors, crops, body vectors, sightings and negatives destroyed; classifier retrained without them. Irreversible |
| Just the *label* (mislabelled) | Same button → decline the erase prompt | Name removed, crops returned to the training pool for re-tagging |
| One event and its clip | Review / Events → delete | Event, thumbnails, satellites and any exported clip |
| Recorded footage | Settings → retention, or delete a range in Review | Video only — **face data is deliberately excluded from footage wipes**, so a retention purge never destroys an enrolment you meant to keep |
| One camera's history | Remove the camera | Its capture stops immediately |
| All of it | Quit and delete the data folder | Nothing survives outside it |

The erase path is the one to use for a data-subject request. It exists because
unlinking a name does not satisfy Article 17 — the descriptor still identifies
the person.

**It does not delete recorded footage**, and that is intentional: a command that
destroyed hours of unrelated recording because one face appeared somewhere in it
would be its own accident. Delete footage by range in Review, or let retention do
it.

### Where the data folder is

| OS | Path |
|---|---|
| Windows | `%APPDATA%\com.nivar.app` |
| macOS | `~/Library/Application Support/com.nivar.app` |
| Linux | `~/.local/share/com.nivar.app` |

Recordings, the SQLite database, blobs and model weights are all under it.

---

## Security of what is stored

- **Secrets** — API keys, bot tokens, passwords — are encrypted at rest with
  AES-GCM, keyed per install.
- **Login**, if enabled, uses Argon2id with constant-time comparison, and
  optional TOTP 2FA.
- **The database itself is not encrypted.** Face descriptors, event metadata and
  crops sit in a plain SQLite file. Anyone with access to the folder — or to a
  backup of it, or to the disk — can read them. **If the machine holds
  biometric data, use full-disk encryption** (BitLocker, FileVault, LUKS). That
  is the honest current state, not a recommendation to skip.
- Remote access requires authentication; share links expire; the local media
  server binds loopback and validates every path.

---

## Children

Not directed at children, and it collects nothing from them online. A camera in a
home will record the children in it; that footage is yours and never leaves the
machine unless you send it. Enrolling a child's face is a decision with extra
weight in most jurisdictions — several require parental consent for biometric
processing of minors regardless of the household exemption.

---

## Changes, and how to ask

This file is versioned in the repository; its history is the changelog.

Anivar has no privacy inbox, because it has no server and receives nothing
from you. **You are the data controller for your own installation** — a request
from someone you have recorded comes to you, and this document tells you how to
answer it.

Bugs in the controls described here — an erase that leaves data behind, an
anonymised camera that leaks a raw frame — are security issues. Report them per
[SECURITY.md](SECURITY.md).

---

*Not legal advice. Whether your particular installation is lawful depends on
where you are, what your cameras see, and who they see — questions this document
cannot answer for you.*
