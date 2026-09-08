//! Schema migrations + SQLite pool setup + per-table helper queries.

use sqlx::SqlitePool;
use uuid::Uuid;

use crate::{Settings, default_ai_model};


pub(crate) async fn init_db(pool: &SqlitePool) -> anyhow::Result<()> {
    // (Per-connection pragmas — journal mode, synchronous, cache_size,
    // journal_size_limit — are applied by ConnectOptions in boot.rs, which
    // reaches EVERY pooled connection; a one-shot PRAGMA here reached only 1
    // of 32 and is not repeated. busy_timeout comes from sqlx's 5 s default.)
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS motion_events (
            id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
            duration_secs REAL, peak_score REAL NOT NULL DEFAULT 0, clip_path TEXT, thumbnail TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_me_started ON motion_events(started_at DESC);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
    ).execute(pool).await?;
    // Idempotent migrations — silently ignored if column already exists
    for ddl in &[
        "ALTER TABLE motion_events ADD COLUMN detections TEXT",
        "ALTER TABLE motion_events ADD COLUMN ai_summary TEXT",
        "ALTER TABLE motion_events ADD COLUMN behavior_flags TEXT",
        "ALTER TABLE agent_alerts ADD COLUMN feedback TEXT",
        "ALTER TABLE agent_alerts ADD COLUMN behavior_flags TEXT",
        "ALTER TABLE agent_alerts ADD COLUMN escalated INTEGER DEFAULT 0",
        // Re-ID: track whether a person was auto-recognized vs manually enrolled
        "ALTER TABLE known_persons ADD COLUMN is_auto INTEGER NOT NULL DEFAULT 0",
        // Re-ID: cumulative sighting count for confidence ranking
        "ALTER TABLE known_persons ADD COLUMN sighting_count INTEGER NOT NULL DEFAULT 1",
    ] {
        let _ = sqlx::query(ddl).execute(pool).await;
    }
    // Durable Guardian chat log — conversations survive restarts; the backend
    // owns history (the agent references earlier discussion + computes
    // "since we last talked" deltas from the last row's timestamp).
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS chat_log (
            id         TEXT PRIMARY KEY,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chatlog_created ON chat_log(created_at DESC);"
    ).execute(pool).await?;

    // Guardian agent tables
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS agent_alerts (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            threat_type TEXT NOT NULL DEFAULT 'unknown',
            summary TEXT NOT NULL,
            is_false_positive INTEGER NOT NULL DEFAULT 0,
            actions_taken TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_aa_created ON agent_alerts(created_at DESC);
        CREATE TABLE IF NOT EXISTS agent_memory (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );",
    ).execute(pool).await?;
    // Known persons — face recognition
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS known_persons (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'resident',
            embeddings TEXT NOT NULL,
            thumbnail TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_seen_at TEXT
        );",
    ).execute(pool).await?;
    // Face sightings — cross-camera person tracking
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS face_sightings (
            id TEXT PRIMARY KEY,
            person_name TEXT NOT NULL,
            camera_id INTEGER NOT NULL DEFAULT 0,
            event_id TEXT,
            seen_at TEXT NOT NULL DEFAULT (datetime('now')),
            confidence REAL NOT NULL DEFAULT 0.0
        );",
    ).execute(pool).await?;
    // Create index for fast cross-camera queries
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_face_sightings_person ON face_sightings(person_name, seen_at DESC);"
    ).execute(pool).await?;
    // Body Re-ID — HSV histogram descriptors for cross-camera cross-session person tracking
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS body_embeddings (
            id         TEXT PRIMARY KEY,
            person_id  TEXT NOT NULL,
            descriptor BLOB NOT NULL,
            cam_id     INTEGER NOT NULL DEFAULT 0,
            event_id   TEXT,
            seen_at    TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_be_person ON body_embeddings(person_id, seen_at DESC);
        CREATE INDEX IF NOT EXISTS idx_be_seen ON body_embeddings(seen_at DESC);"
    ).execute(pool).await?;
    // Carry a base64 JPEG crop of the person (first sighting) so the People →
    // Tracked view shows the actual person — body Re-ID stores only a histogram
    // descriptor, so without this the cards had no image to render.
    let _ = sqlx::query(
        "ALTER TABLE body_embeddings ADD COLUMN thumbnail_b64 TEXT;"
    ).execute(pool).await; // ignore if column already exists
    // Face↔body fusion (Avigilon/BriefCam-style): when a face is recognised up
    // close, the co-occurring body is auto-labelled with that known person. Rows
    // in the person's `kp_<known_id>` gallery carry the link here so a distant /
    // cross-camera body match resolves to a name. NULL = anonymous "Person N".
    let _ = sqlx::query(
        "ALTER TABLE body_embeddings ADD COLUMN known_person_id TEXT;"
    ).execute(pool).await; // ignore if column already exists
    // Clothing attributes per sighting: compact JSON like {"top":"blue",
    // "bottom":"black"} from the HSV band classifier. Words only (no imagery);
    // majority-voted per track into the UI's "outfit" line so anonymous tracks
    // are human-recognisable. NULL on rows stored before this column existed.
    let _ = sqlx::query(
        "ALTER TABLE body_embeddings ADD COLUMN attrs TEXT;"
    ).execute(pool).await; // ignore if column already exists
    // Native face recognition (standard) — embeddings produced by FaceNet (128-d) or ArcFace (512-d).
    // `person_id` is NULL for sightings that didn't match any known person (kept for later auto-enrolment).
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS face_embeddings (
            id         TEXT PRIMARY KEY,
            person_id  TEXT,
            descriptor BLOB NOT NULL,
            dim        INTEGER NOT NULL,
            quality    REAL NOT NULL DEFAULT 1.0,
            cam_id     INTEGER NOT NULL DEFAULT 0,
            event_id   TEXT,
            seen_at    TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_fe_person ON face_embeddings(person_id, seen_at DESC);
        CREATE INDEX IF NOT EXISTS idx_fe_event  ON face_embeddings(event_id);"
    ).execute(pool).await?;
    // Carry the aligned 112×112 face crop as a base64 JPEG so the Persons UI
    // can render "what the agent saw" when training on unknown sightings.
    // Mature NVRs ship the same — without a thumbnail there's nothing to tag.
    let _ = sqlx::query(
        "ALTER TABLE face_embeddings ADD COLUMN thumbnail_b64 TEXT;"
    ).execute(pool).await; // ignore if column already exists
    // Full source frame (downscaled JPEG) the face was captured in, so the Train UI
    // can expand a face crop into the whole scene — context for "who is this?".
    let _ = sqlx::query(
        "ALTER TABLE face_embeddings ADD COLUMN context_b64 TEXT;"
    ).execute(pool).await; // ignore if column already exists
    // Desktop login "remember this device" tokens — stored HASHED (SHA-256) with an
    // expiry. Presenting a matching unexpired token on launch resumes the unlocked
    // state without re-entering the password. Logout / disable deletes them.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS auth_remember (
            token_hash TEXT PRIMARY KEY,
            expires_at TEXT NOT NULL
        );"
    ).execute(pool).await?;
    // Hybrid face matching: the trained linear classifier (logistic regression over
    // the 512-d ArcFace vectors) lives as a single JSON row. Recomputed on every
    // enroll/confirm/delete; absent → recognition falls back to pure cosine-NN.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS face_classifier_model (
            id         INTEGER PRIMARY KEY CHECK (id = 1),
            model_json TEXT NOT NULL,
            trained_at TEXT NOT NULL
        );"
    ).execute(pool).await?;

    // Hard negatives: face descriptors the user EXPLICITLY corrected away from a
    // person ("this is NOT X"). The classifier feeds these into its reject class so
    // a corrected mistake stops repeating — the "corrections become training signal"
    // loop edge-AI systems rely on. person_id = the WRONG person it was confused with.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS face_negatives (
            id          TEXT PRIMARY KEY,
            person_id   TEXT NOT NULL,
            descriptor  BLOB NOT NULL,
            dim         INTEGER NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );"
    ).execute(pool).await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_face_negatives_person ON face_negatives(person_id);")
        .execute(pool).await?;

    // BODY hard-negatives — the body-Re-ID analog of face_negatives. A row means
    // "this body descriptor is NOT known_person_id" and makes a tracking correction
    // DURABLE: matching/clustering suppress a person whose negatives the candidate
    // resembles, so the same appearance mistake can't re-merge next sighting.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS body_negatives (
            id              TEXT PRIMARY KEY,
            known_person_id TEXT NOT NULL,
            descriptor      BLOB NOT NULL,
            dim             INTEGER NOT NULL,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );"
    ).execute(pool).await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_body_negatives_person ON body_negatives(known_person_id);")
        .execute(pool).await?;
    // One-time: drop legacy STRANGER face crops captured before context_b64 existed —
    // those are the tilted, tightly-cropped aligned-warp images with no full frame
    // that the user can't identify. Guarded so it runs once; enrolled people are kept;
    // new captures (which all carry context_b64 + a person crop) survive.
    {
        let purged: Option<(String,)> = sqlx::query_as("SELECT value FROM settings WHERE key='face_legacy_purge_v1'")
            .fetch_optional(pool).await.ok().flatten();
        if purged.is_none() {
            let _ = sqlx::query("DELETE FROM face_embeddings WHERE person_id IS NULL AND context_b64 IS NULL")
                .execute(pool).await;
            let _ = sqlx::query("INSERT OR REPLACE INTO settings(key,value) VALUES('face_legacy_purge_v1','1')")
                .execute(pool).await;
        }
    }
    // One-time: drop cached event clips built with the BROKEN concat-`inpoint` trim
    // (black "no footage" at the front on H.264). Deleting the files + NULLing
    // clip_path forces `ensure_event_clip` to regenerate them with the `-ss` trim on
    // next open, so PAST events come back clean — not just new ones.
    {
        let done: Option<(String,)> = sqlx::query_as("SELECT value FROM settings WHERE key='clip_reencode_ss_v1'")
            .fetch_optional(pool).await.ok().flatten();
        if done.is_none() {
            let paths: Vec<(String,)> = sqlx::query_as(
                "SELECT clip_path FROM motion_events WHERE clip_path IS NOT NULL AND clip_path != ''"
            ).fetch_all(pool).await.unwrap_or_default();
            for (p,) in &paths { let _ = std::fs::remove_file(p); }
            let _ = sqlx::query("UPDATE motion_events SET clip_path=NULL WHERE clip_path IS NOT NULL")
                .execute(pool).await;
            let _ = sqlx::query("INSERT OR REPLACE INTO settings(key,value) VALUES('clip_reencode_ss_v1','1')")
                .execute(pool).await;
        }
    }
    // One-time: drop 0-byte / missing cached clips (junk from the old EAGER close-time
    // generation that ran before the footage existed). NULL their clip_path so they
    // regenerate on next open. `ensure_event_clip` now also self-heals these on access.
    {
        let done: Option<(String,)> = sqlx::query_as("SELECT value FROM settings WHERE key='clip_zero_purge_v1'")
            .fetch_optional(pool).await.ok().flatten();
        if done.is_none() {
            let rows: Vec<(String, String)> = sqlx::query_as(
                "SELECT id, clip_path FROM motion_events WHERE clip_path IS NOT NULL AND clip_path != ''"
            ).fetch_all(pool).await.unwrap_or_default();
            for (id, p) in rows {
                let bad = std::fs::metadata(&p).map(|m| m.len() < 4096).unwrap_or(true);
                if bad {
                    let _ = std::fs::remove_file(&p);
                    let _ = sqlx::query("UPDATE motion_events SET clip_path=NULL WHERE id=?")
                        .bind(&id).execute(pool).await;
                }
            }
            let _ = sqlx::query("INSERT OR REPLACE INTO settings(key,value) VALUES('clip_zero_purge_v1','1')")
                .execute(pool).await;
        }
    }
    // Body Re-ID retention: anonymous appearance fragments (`body_*`, NOT face-anchored)
    // are only useful while the outfit is unchanged (~a day). Without pruning they pile
    // up into hundreds of orphan "Person N" (the same person re-minted every day/outfit).
    // Delete anonymous fragments older than 2 days on every boot — face-linked galleries
    // (`kp_*` / `known_person_id` set) are ALWAYS kept. (An hourly prune in the inference
    // loop, `reid::prune_anon_bodies`, covers long-running sessions.)
    let _ = sqlx::query(
        "DELETE FROM body_embeddings
          WHERE known_person_id IS NULL AND person_id LIKE 'body\\_%' ESCAPE '\\'
            AND seen_at < datetime('now','-2 days')"
    ).execute(pool).await;

    // Normalize orphans left by the OLD delete_person (which removed only the
    // roster row): linked face crops pointing at deleted people were invisible
    // zombies (excluded from Train by person_id being set, from galleries by the
    // missing person); orphaned body links were kept FOREVER (excluded from
    // matching by the JOIN, from pruning by known_person_id being set). Same
    // normalization the fixed delete_person applies, run once per boot — cheap
    // and idempotent (0 rows once clean, and delete_person now cleans as it goes).
    let _ = sqlx::query(
        "UPDATE face_embeddings SET person_id=NULL
          WHERE person_id IS NOT NULL
            AND person_id NOT IN (SELECT id FROM known_persons)"
    ).execute(pool).await;
    let _ = sqlx::query(
        "DELETE FROM face_negatives
          WHERE person_id NOT IN (SELECT id FROM known_persons)"
    ).execute(pool).await;
    let _ = sqlx::query(
        "UPDATE body_embeddings
            SET known_person_id=NULL,
                person_id='body_' || substr(lower(hex(randomblob(4))),1,8)
          WHERE known_person_id IS NOT NULL
            AND known_person_id NOT IN (SELECT id FROM known_persons)"
    ).execute(pool).await;
    let _ = sqlx::query(
        "DELETE FROM face_sightings
          WHERE person_name NOT IN (SELECT name FROM known_persons)"
    ).execute(pool).await;

    // Semantic event search (standard) — CLIP embeddings of each closed
    // event. `kind` = 'image' (thumbnail embedding) or 'text' (ai_summary embedding);
    // both land in the same shared CLIP space so a text query can cosine-match either.
    // `model` tags which embedding model produced the vector (mobileclip_s0 / clip_b32 /
    // jina_clip) — different models live in different spaces, so cosine only ever
    // compares same-model rows. `descriptor` is little-endian f32 (same encoding as
    // face/body embeddings). Populated lazily by agent::clip::embed_event when a
    // search model is active; absent rows simply mean keyword search is used.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS event_embeddings (
            id         TEXT PRIMARY KEY,
            event_id   TEXT NOT NULL,
            kind       TEXT NOT NULL,
            descriptor BLOB NOT NULL,
            dim        INTEGER NOT NULL,
            model      TEXT NOT NULL DEFAULT 'jina_clip',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_ee_event ON event_embeddings(event_id);"
    ).execute(pool).await?;
    // Migration: add `model` to pre-existing tables (ignored if already present),
    // then replace the old (event_id, kind) unique index with one that includes
    // `model` so multiple models' vectors can coexist per event.
    let _ = sqlx::query("ALTER TABLE event_embeddings ADD COLUMN model TEXT NOT NULL DEFAULT 'jina_clip'")
        .execute(pool).await;
    sqlx::query(
        "DROP INDEX IF EXISTS idx_ee_event_kind;
         CREATE UNIQUE INDEX IF NOT EXISTS idx_ee_event_kind_model ON event_embeddings(event_id, kind, model);"
    ).execute(pool).await?;

    // Camera configuration — persist names, sources, and layout for each of the 16 slots
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS camera_configs (
            cam_id  INTEGER PRIMARY KEY,
            name    TEXT NOT NULL DEFAULT '',
            source_type TEXT NOT NULL DEFAULT 'browser',
            source_url  TEXT NOT NULL DEFAULT '',
            device_id   TEXT NOT NULL DEFAULT '',
            enabled     INTEGER NOT NULL DEFAULT 1,
            transport   TEXT NOT NULL DEFAULT 'tcp',
            brand       TEXT NOT NULL DEFAULT ''
        );"
    ).execute(pool).await?;
    // Migrate older DBs that predate the transport / brand columns.
    let _ = sqlx::query("ALTER TABLE camera_configs ADD COLUMN transport TEXT NOT NULL DEFAULT 'tcp'")
        .execute(pool).await;
    let _ = sqlx::query("ALTER TABLE camera_configs ADD COLUMN brand TEXT NOT NULL DEFAULT ''")
        .execute(pool).await;
    // Optional low-res DETECT sub-stream URL (mature NVRs' model: detect on the
    // camera's substream, record the main stream). Empty = detect on main.
    let _ = sqlx::query("ALTER TABLE camera_configs ADD COLUMN detect_url TEXT NOT NULL DEFAULT ''")
        .execute(pool).await;
    // NVR segments table — continuous recording segments with rich metadata
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS nvr_segments (
            id           TEXT PRIMARY KEY,
            cam_id       INTEGER NOT NULL,
            path         TEXT NOT NULL,
            started_at   TEXT NOT NULL,
            ended_at     TEXT,
            size_bytes   INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_nvr_cam     ON nvr_segments(cam_id, started_at DESC);"
    ).execute(pool).await?;

    // Scrub previews: one low-res file per camera per hour, dragged instead of the
    // real recording. See nvr_preview.rs.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS nvr_previews (
            id          TEXT PRIMARY KEY,
            cam_id      INTEGER NOT NULL,
            start_time  TEXT NOT NULL,
            end_time    TEXT NOT NULL,
            path        TEXT NOT NULL,
            size_bytes  INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_prev_cam ON nvr_previews(cam_id, start_time);"
    ).execute(pool).await?;

    // Idempotent migrations — silently no-op if column already exists.
    for ddl in &[
        "ALTER TABLE nvr_segments ADD COLUMN duration_secs REAL",
        "ALTER TABLE nvr_segments ADD COLUMN peak_motion REAL NOT NULL DEFAULT 0",
        "ALTER TABLE nvr_segments ADD COLUMN objects TEXT",
        "ALTER TABLE nvr_segments ADD COLUMN has_alert INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE nvr_segments ADD COLUMN recording_mode TEXT NOT NULL DEFAULT 'all'",
        "ALTER TABLE nvr_segments ADD COLUMN is_rtsp_copy INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE motion_events ADD COLUMN motion_regions TEXT",
        "ALTER TABLE motion_events ADD COLUMN cam_id INTEGER NOT NULL DEFAULT 0",
        // scored-memory scored memory: score = confidence, confirmations = how often reinforced
        "ALTER TABLE agent_memory ADD COLUMN score REAL NOT NULL DEFAULT 1.0",
        "ALTER TABLE agent_memory ADD COLUMN confirmations INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE nvr_segments ADD COLUMN has_audio INTEGER NOT NULL DEFAULT 0",
        // Audio-event loudness (dBFS) — YAMNet-properly enrichment for audio cards.
        "ALTER TABLE motion_events ADD COLUMN loudness_db REAL",
        // Audio classification metadata gets its OWN column: the clip-analysis
        // pipeline owns `attributes` (writes its array wholesale), which was
        // CLOBBERING the audio {classes,high_pitch} object stored there.
        // Separate columns end the fight. + one-time salvage of survivors.
        "ALTER TABLE motion_events ADD COLUMN audio_meta TEXT",
        // Vehicle body color (HSV-voted at clip analysis) — own column.
        "ALTER TABLE motion_events ADD COLUMN vehicle_color TEXT",
        "UPDATE motion_events SET audio_meta = json_extract(attributes,'$.audio') \
         WHERE audio_meta IS NULL AND json_extract(attributes,'$.audio') IS NOT NULL",
        // REPAIR: clip analysis used to overwrite event_category with its
        // YOLO-derived guess, silently turning audio events into 'other' (they
        // vanished from the Audio tab). The event_timeline 'audio' rows are the
        // durable birth record — restore the category from them. Idempotent.
        "UPDATE motion_events SET event_category='audio'          WHERE event_category != 'audio'            AND id IN (SELECT DISTINCT event_id FROM event_timeline WHERE class_type='audio')",
        // Legacy audio-only review segments were stored severity='alert' (audio
        // presence / YAMNet confidence used to escalate). Idempotent normalize:
        // audio-only (no non-audio category, no zones/fall/crossing) => detection.
        "UPDATE review_segments SET severity='detection' \
         WHERE severity='alert' \
           AND json_extract(data,'$.audio') IS NOT NULL \
           AND (SELECT COUNT(*) FROM json_each(json_extract(data,'$.categories')) je WHERE je.value != 'audio') = 0 \
           AND json_array_length(coalesce(json_extract(data,'$.zones'),'[]')) = 0 \
           AND json_extract(data,'$.fall') IS NULL \
           AND json_extract(data,'$.crossing') IS NULL",
        "ALTER TABLE agent_memory ADD COLUMN created_at TEXT",
        "ALTER TABLE agent_memory ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'manual'",
        // ── Identity traceability (unified face+body naming) ──────────────
        // Provenance on every naming decision: WHICH recognizer, top score,
        // runner-up margin. Old rows stay NULL (display falls back gracefully);
        // never rewrite existing face data (hard rule).
        "ALTER TABLE face_sightings ADD COLUMN person_id TEXT",
        "ALTER TABLE face_sightings ADD COLUMN method TEXT",
        "ALTER TABLE face_embeddings ADD COLUMN match_method TEXT",
        "ALTER TABLE face_embeddings ADD COLUMN match_score REAL",
        "ALTER TABLE face_embeddings ADD COLUMN match_margin REAL",
        "ALTER TABLE body_embeddings ADD COLUMN match_method TEXT",
        "ALTER TABLE body_embeddings ADD COLUMN match_score REAL",
    ] {
        let _ = sqlx::query(ddl).execute(pool).await;
    }
    // Indexes on the new columns — created AFTER migrations so columns exist
    for ddl in &[
        "CREATE INDEX IF NOT EXISTS idx_nvr_started ON nvr_segments(started_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_nvr_alert   ON nvr_segments(has_alert, started_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_me_cam       ON motion_events(cam_id, started_at DESC)",
    ] {
        let _ = sqlx::query(ddl).execute(pool).await;
    }
    // DEDUPE nvr_segments + enforce ONE row per file. The live post-processor indexed
    // each segment with a fresh random UUID and no path check, so re-scans piled up
    // 2–3 duplicate rows per file (3000+ rows for ~1300 files) — which bloated the DB,
    // hammered SQLite with INSERT/scan churn, and made the continuous player concat
    // overlapping/duplicate segments. Collapse to the earliest row per path, then a
    // UNIQUE index makes the recorders' `INSERT OR IGNORE` actually dedupe by path.
    for ddl in &[
        "DELETE FROM nvr_segments WHERE rowid NOT IN (SELECT MIN(rowid) FROM nvr_segments GROUP BY path)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_nvr_path_unique ON nvr_segments(path)",
    ] {
        let _ = sqlx::query(ddl).execute(pool).await;
    }

    // assistant-parity features
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS alert_conditions (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            condition   TEXT NOT NULL,      -- plain English: 'person at door after 10pm'
            channels    TEXT NOT NULL DEFAULT 'all',  -- 'all' | 'telegram' | 'discord' | 'silent'
            min_risk    TEXT NOT NULL DEFAULT 'medium',
            enabled     INTEGER NOT NULL DEFAULT 1,
            trigger_count INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL
        )"
    ).execute(pool).await?;

    // Quiet hours, persona, structured memory types — added as settings migrations
    for ddl in &[
        "ALTER TABLE motion_events ADD COLUMN frame_descriptions TEXT", // stroboscopic frame analysis
        "ALTER TABLE motion_events ADD COLUMN top_speed_kmh REAL",      // zone speed estimation (Part 6B)
        // NVR-parity labelling: structured typed attributes + confidence scores.
        "ALTER TABLE motion_events ADD COLUMN attributes TEXT",         // JSON [{type,value,score}] (face/plate/object/audio)
        "ALTER TABLE motion_events ADD COLUMN sub_label_score REAL",    // confidence of the sub_label (face name)
        "ALTER TABLE motion_events ADD COLUMN plate_score REAL",        // confidence of recognized_plate
        "ALTER TABLE motion_events ADD COLUMN false_positive INTEGER",  // promoted from ai_summary.fp for filtering
        // Clothing colours of the dominant person, voted across frames:
        // {"top":"red","bottom":"black"} — either key may be absent, and NULL
        // means UNKNOWN, never "not red". This is what makes "the person in the
        // red jacket" a query instead of a wish: the colours were already being
        // computed per frame (`reid::classify_person_colors`) and thrown into
        // `body_embeddings.attrs`, a column with zero WHERE clauses against it.
        // Its own column rather than folded into `attributes`, because the clip
        // pipeline rewrites that one wholesale (see the audio_meta salvage below).
        "ALTER TABLE motion_events ADD COLUMN outfit TEXT",
        "ALTER TABLE agent_alerts ADD COLUMN condition_id TEXT",        // which user condition triggered
        // standard event categorisation: "person" | "vehicle" | "animal" |
        // "package" | "other" — derived from YOLO detections in
        // `agent/clip.rs::categorise_detections`. Lets the Events UI render a
        // type badge and lets filters scope to "all vehicles in the last week".
        "ALTER TABLE motion_events ADD COLUMN event_category TEXT",
        // Recognised license plate text (when ALPR skill is installed and a
        // vehicle was detected in the event). Populated by `alpr.rs`.
        "ALTER TABLE motion_events ADD COLUMN recognized_plate TEXT",
        // v8: CSV of named zones the event's detections passed through.
        // Built in `agent/clip.rs::analyze_event_clip` from the detection
        // buffer × the camera's parsed zones. Lets ReviewPanel render a
        // "front_porch" / "driveway" chip alongside the category badge and
        // feeds the agent's "## Zones entered" context block.
        "ALTER TABLE motion_events ADD COLUMN zones_entered TEXT",
        // v27 mature NVRs per-label model: the specific dominant COCO label of the
        // event (e.g. "person", "dog", "car") — shown as the real-object chip in
        // the Review UI, distinct from the broad `event_category` bucket.
        "ALTER TABLE motion_events ADD COLUMN dominant_label TEXT",
        // v27: sub-label refinement (mature NVRs attributes) — a known face name, a
        // recognised/known plate, or a delivery brand. Refines the base label.
        "ALTER TABLE motion_events ADD COLUMN sub_label TEXT",
        // v28: wall-clock (RFC3339) when YOLO FIRST confirmed a tracked object in
        // this event. started_at is motion onset; the object usually appears a
        // few seconds later. Used to trim the event clip's empty lead-in so it
        // starts ~pre seconds before the object, not before the raw motion.
        "ALTER TABLE motion_events ADD COLUMN first_object_at TEXT",
        // ── Event lifecycle timeline (mature NVRs `Timeline` table) ──────────────
        // One row per notable moment WITHIN an event: object appeared, entered a
        // zone, became stationary, was recognised (face), plate read (lpr), line
        // crossed, sped, audio detected, fall, gone. Gives a queryable
        // "what happened, when" record per event — rendered as a Review timeline,
        // fed to the agent, and folded into the searchable summary.
        "CREATE TABLE IF NOT EXISTS event_timeline (\
            id INTEGER PRIMARY KEY AUTOINCREMENT,\
            event_id TEXT NOT NULL,\
            cam_id INTEGER NOT NULL DEFAULT 0,\
            ts TEXT NOT NULL,\
            class_type TEXT NOT NULL,\
            label TEXT,\
            value TEXT,\
            score REAL,\
            data TEXT)",
        "CREATE INDEX IF NOT EXISTS idx_event_timeline_event ON event_timeline(event_id)",
    ] {
        let _ = sqlx::query(ddl).execute(pool).await;
    }

    // ── Review segments (`ReviewSegment` parity) ────────────────
    // A "review item" is a non-overlapping per-camera time window that bundles
    // overlapping `motion_events` into ONE reviewable unit with a single
    // severity (alert/detection) and a DB-backed reviewed flag. This is the
    // SERVER-SIDE canonical grouping that Review + the NVR timeline both read —
    // replacing the old client-only `ReviewFeed.buildReviewItems`. `data` is a
    // JSON `ReviewSegmentData` (member event ids + aggregated labels/zones/etc),
    // re-aggregated from members on every upsert so the row is idempotent.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS review_segments (
            id          TEXT PRIMARY KEY,
            cam_id      INTEGER NOT NULL,
            start_time  TEXT NOT NULL,
            end_time    TEXT,
            severity    TEXT NOT NULL,
            thumbnail   TEXT,
            data        TEXT NOT NULL,
            reviewed    INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_rs_cam_start ON review_segments(cam_id, start_time DESC);
        CREATE INDEX IF NOT EXISTS idx_rs_start     ON review_segments(start_time DESC);"
    ).execute(pool).await?;

    // ── Event bookmarks (saved/favorites pattern) ────────────────────────────
    // A user-saved EVENT — one row per bookmarked motion_event. Drives the
    // bookmark toggle on every Review card + the "Bookmarks" tab. The old
    // time-anchored `bookmarks` table (pins/list) is obsolete — drop it.
    sqlx::query("DROP TABLE IF EXISTS bookmarks;").execute(pool).await.ok();
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS event_bookmarks (
            event_id    TEXT PRIMARY KEY,
            cam_id      INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ebm_created ON event_bookmarks(created_at DESC);"
    ).execute(pool).await?;


    // Telegram inline search was removed; its thumbnail cache goes with it so
    // upgraded installs don't keep a table nothing reads.
    sqlx::query("DROP TABLE IF EXISTS telegram_thumb_cache;").execute(pool).await.ok();

    reap_orphaned_open_events(pool).await;

    Ok(())
}

/// Close motion events left OPEN by an unclean shutdown.
///
/// `ended_at IS NULL` means "in progress", and several subsystems trust that:
/// the live-analysis loop, the audio event tracker, and the detection pipeline all
/// select on it. If the process is killed mid-event nothing ever writes `ended_at`,
/// so the row claims to be in progress forever — and because the live-analysis loop
/// takes `ORDER BY started_at DESC LIMIT 3`, a few zombies permanently occupy every
/// slot and genuinely live events stop being analysed at all. Observed on this box:
/// 6 zombies, three of them re-analysed on every launch, one from 14 hours earlier.
///
/// Anything still open at startup predates this process and cannot be live, so it is
/// closed at `started_at`: the real end time died with the process, and dating a
/// stale event to "now" would corrupt the timeline it appears on. `duration_secs`
/// keeps whatever was actually recorded, so nothing is invented here.
async fn reap_orphaned_open_events(pool: &SqlitePool) {
    match sqlx::query(
        "UPDATE motion_events SET ended_at = started_at WHERE ended_at IS NULL"
    ).execute(pool).await {
        Ok(r) if r.rows_affected() > 0 =>
            tracing::info!("closed {} event(s) left open by an unclean shutdown", r.rows_affected()),
        Ok(_)  => {}
        Err(e) => tracing::warn!("could not reap orphaned open events: {e}"),
    }
}

const VALID_AI_MODELS: &[&str] = &[
    // YOLO11 (2024-2026) — Ultralytics latest generation, best accuracy/speed
    "onnx-community/yolo11n-uint8",   // nano — fastest, ~6ms/frame on GPU
    "onnx-community/yolo11s-uint8",   // small — recommended for most setups
    "onnx-community/yolo11m",         // medium — excellent accuracy
    "onnx-community/yolo11l",         // large — maximum precision
    // RF-DETR (Real-Time DEtection TRansformer) — competitive with YOLO11
    "onnx-community/rfdetr_nano-ONNX",
    "onnx-community/rfdetr_small-ONNX",
    "onnx-community/rfdetr_medium-ONNX",
    // Legacy YOLO / DETR models
    "Xenova/yolov9-c",
    "onnx-community/dfine_n_coco-ONNX",
    "onnx-community/rtdetr_r50vd",
    "Xenova/yolos-tiny",
    "Xenova/detr-resnet-50",
];

pub(crate) async fn load_settings_from_db(pool: &SqlitePool) -> Settings {
    let row: Option<(String,)> = sqlx::query_as("SELECT value FROM settings WHERE key='settings'")
        .fetch_optional(pool).await.unwrap_or(None);
    let mut s: Settings = if let Some((v,)) = row {
        serde_json::from_str(&v).unwrap_or_default()
    } else {
        Settings::default()
    };
    // Reset invalid/removed model IDs to the current default
    if !VALID_AI_MODELS.contains(&s.ai_model.as_str()) {
        s.ai_model = default_ai_model();
    }
    // NVR is now on by default — upgrade existing saved settings silently
    // Force 1-minute segments — shorter segments appear faster in the NVR timeline.
    // Migrate anyone still on the old 10-minute default.
    if s.nvr_segment_mins > 5 { s.nvr_segment_mins = 1; }
    s.nvr_enabled = true;
    // Force faster polling so events are analysed promptly
    if s.agent_poll_secs > 15 { s.agent_poll_secs = 10; }
    // Migrate from port 8880 (ghost socket issue) to 8882
    if s.stream_port == 8880 { s.stream_port = 8882; }
    // Face thresholds shipped defaulted to mature NVRs' 0.9/0.8 PROBABILITY scale, but our
    // matcher scores raw ArcFace cosine (same-person ≈ 0.5) — so a real match never
    // reached 0.9 and recognition silently never fired. Migrate ONCE to the cosine
    // scale, coercing ANY leftover probability-era value (the old code only caught
    // values ≥0.85/0.70, so a stale 0.80 silently survived and kept breaking matching).
    // The one-time flag means deliberate cosine values chosen later are respected.
    if !s.face_thresholds_migrated {
        // ArcFace same-person cosine tops out around ~0.6; a recognition bar above
        // 0.65 would reject every real face, so it can only be a stale probability.
        if s.face_recognition_threshold > 0.65 { s.face_recognition_threshold = 0.5; }
        if s.face_unknown_score        > 0.55 { s.face_unknown_score        = 0.4; }
        // The blur floor was 0.20 on the old ÷1500 quality scale (now ÷300) — ~5×
        // too high, silently dropping real (interpolation-softened) crops.
        if s.face_quality_floor        >= 0.20 { s.face_quality_floor        = 0.10; }
        s.face_thresholds_migrated = true;
    }

    // ── Versioned one-time migrations ────────────────────────────────────────
    // The "audio_detection lesson": changing a struct DEFAULT never reaches
    // existing installs, because Save re-writes every field. Intentional
    // behavior flips go here, gated by `settings_version`, applied exactly
    // once and persisted immediately (secrets are still encrypted at this
    // point, so writing `s` back stores the same representation it came from).
    let pre_version = s.settings_version;
    if s.settings_version < 1 {
        // v1 (2026-07): audio detection ON. The whole audio pipeline (mic ring
        // buffer for clip audio + YAMNet sustained sound events) was silently
        // dead on existing installs because a pre-change `false` was baked into
        // saved settings. Users who dislike it can turn it off — that choice
        // then sticks, because this runs only once.
        s.audio_detection = true;
        s.settings_version = 1;
    }
    if s.settings_version < 2 {
        // v2 (2026-07): add "speech" to the DEFAULT listen list (NVR parity —
        // its default audio set includes speech). Only when the user hasn't
        // customized the list; a custom list is their choice and is respected.
        if s.audio_listen == "scream,glass,alarm,gunshot,bark,yell" {
            s.audio_listen = "scream,glass,alarm,gunshot,bark,yell,speech".to_string();
        }
        s.settings_version = 2;
    }
    if s.settings_version < 3 {
        // v3 (2026-07-28): Ollama is gone — the app no longer downloads, spawns or
        // watchdogs a language-model server (it left a 6 GB orphan on a 16 GB
        // machine). The model now runs in-process; see `agent::local_llm`.
        // Existing installs have "ollama" SAVED, so flipping the struct default
        // would never reach them — the audio_detection lesson above, exactly.
        // Cloud providers and a self-hosted OpenAI-compatible endpoint are the
        // user's own choices and are left alone.
        if s.ai_provider == "ollama" || s.ai_provider.is_empty() {
            s.ai_provider = "local".to_string();
        }
        s.settings_version = 3;
    }
    if s.settings_version < 4 {
        // v4 (2026-07-28): v3 switched the PROVIDER to on-device but left
        // `vision_model` holding whatever Ollama tag was last selected. Nothing reads
        // it any more (capability now comes from the provider, not the string), but it
        // still renders in the Arsenal badge as "On-device · qwen3-vl:2b" — a model
        // that isn't installed and couldn't run if it were. Clear it so the UI stops
        // claiming an engine that doesn't exist.
        if s.ai_provider == "local" {
            s.vision_model.clear();
        }
        s.settings_version = 4;
    }
    if s.settings_version != pre_version {
        tracing::info!("settings migrated v{pre_version} → v{}", s.settings_version);
        let _ = sqlx::query("INSERT OR REPLACE INTO settings(key,value) VALUES('settings',?)")
            .bind(serde_json::to_string(&s).unwrap_or_default())
            .execute(pool).await;
    }
    s
}

/// Probe a camera base URL to find the actual MJPEG stream endpoint.
/// Tries common paths (server-side so no CSP issues).
/// Returns the working stream URL or the original if none found.
#[tauri::command]
pub async fn probe_mjpeg_url(base_url: String, user: Option<String>, pass: Option<String>) -> Result<String, String> {
    const PATHS: &[&str] = &[
        "",
        "/video",
        "/stream",
        "/mjpeg",
        "/mjpg/video.mjpg",
        "/cgi-bin/video.cgi",
        "/?action=stream",
        "/videostream.cgi",
        "/axis-cgi/mjpg/video.cgi",
        "/live",
        "/live/0/MJPEG.mjpg",
        "/h264",
        "/cam",
        "/camera",
        "/video.cgi",
        "/image.jpg",
    ];

    let base = base_url.trim_end_matches('/').to_string();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .map_err(|e| e.to_string())?;

    for path in PATHS {
        let candidate = format!("{}{}", base, path);
        let mut req = client.get(&candidate);
        if let (Some(u), _) = (&user, &pass) {
            if !u.is_empty() {
                req = req.basic_auth(u, pass.as_deref().filter(|p| !p.is_empty()));
            }
        }
        if let Ok(resp) = req.send().await {
            if resp.status().is_success() {
                let ct = resp.headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_lowercase();
                // Valid MJPEG stream returns multipart or image content
                if ct.contains("multipart") || ct.contains("mjpeg") || ct.contains("jpeg") || ct.contains("image") {
                    tracing::info!("MJPEG probe found stream at: {}", candidate);
                    return Ok(candidate);
                }
            }
        }
    }

    // Nothing found — return base URL and let user try
    Ok(base_url)
}

pub(crate) async fn save_settings_to_db(pool: &SqlitePool, s: &Settings) -> anyhow::Result<()> {
    sqlx::query("INSERT OR REPLACE INTO settings(key,value) VALUES('settings',?)")
        .bind(serde_json::to_string(s)?).execute(pool).await?;
    Ok(())
}

pub(crate) async fn load_or_create_auth_token(pool: &SqlitePool) -> String {
    let row: Option<(String,)> = sqlx::query_as("SELECT value FROM settings WHERE key='auth_token'")
        .fetch_optional(pool).await.unwrap_or(None);
    if let Some((token,)) = row { if !token.is_empty() { return token; } }
    let token = Uuid::new_v4().simple().to_string();
    sqlx::query("INSERT OR REPLACE INTO settings(key,value) VALUES('auth_token',?)")
        .bind(&token).execute(pool).await.ok();
    token
}

