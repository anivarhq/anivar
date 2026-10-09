//! End-to-end: a generated video through the real recording, indexing, motion
//! and clip paths. Ignored by default because ffmpeg runs in real time (~25 s)
//! and may download the pinned build; CI runs it on Windows:
//!
//!   cargo test --lib -- --ignored e2e
//!
//! `AppState` holds a live Tauri app handle, so the test enters one layer below
//! it: ffmpeg records with the app's segment contract (`segment_output_args`)
//! while the same decoded frames go to the motion score
//! (`compute_motion_masked`) and the event lifecycle's decisions
//! (`motion_lifecycle::step`); the post-processor's pass (`finalise_incoming`)
//! indexes the segments; `concat_window_to_file` cuts the event's clip. Not
//! covered: the event-row writes `tick_motion_event` makes around `step`.

use std::io::Read;
use std::time::Duration;

use crate::motion_lifecycle::{step, LifecycleSettings, Step};

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn e2e_motion_is_recorded_indexed_and_clipped() {
    let dir = std::env::temp_dir().join(format!("anivar-e2e-{}", uuid::Uuid::new_v4()));
    let nvr = dir.join("nvr");
    std::fs::create_dir_all(crate::nvr_pipes::incoming_dir(&nvr)).unwrap();
    let ffmpeg = crate::ffmpeg::ensure_ffmpeg(&dir).await.expect("ffmpeg");
    let db = sqlx::sqlite::SqlitePoolOptions::new().max_connections(1)
        .connect(&format!("sqlite://{}?mode=rwc", dir.join("e2e.db").to_string_lossy().replace('\\', "/")))
        .await.unwrap();
    crate::db::init_db(&db).await.unwrap();

    // 8 s of a moving test pattern, then 14 s of one still frame: one event,
    // opened by the motion and closed by the default 10 s post-buffer.
    let source = "testsrc2=size=320x240:rate=10:duration=8[a];\
                  color=c=gray:size=320x240:rate=10:duration=14[b];[a][b]concat=n=2:v=1:a=0[out0]";
    let mut args: Vec<String> = ["-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi", "-i", source,
        "-map", "0:v", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
        .iter().map(|s| s.to_string()).collect();
    args.extend(crate::nvr_pipes::segment_output_args(0, &nvr));
    args.extend(["-map", "0:v", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "5", "pipe:1"].map(String::from));

    let settings = crate::state::Settings::default();
    let lifecycle = LifecycleSettings::from_settings(&settings, false);
    let (events, scores) = tokio::task::spawn_blocking(move || {
        let mut child = std::process::Command::new(&ffmpeg).args(&args)
            .stdout(std::process::Stdio::piped()).spawn().expect("spawn ffmpeg");
        let mut out = child.stdout.take().unwrap();
        let mut cs = crate::state::PerCamState::default();
        let (mut events, mut scores) = (Vec::new(), Vec::new());
        let (mut buf, mut chunk) = (Vec::new(), [0u8; 65536]);
        loop {
            let n = out.read(&mut chunk).unwrap_or(0);
            if n == 0 { break; }
            buf.extend_from_slice(&chunk[..n]);
            // ffmpeg's JPEGs end at EOI (FF D9), which can't occur inside one.
            while let Some(end) = buf.windows(2).position(|w| w == [0xFF, 0xD9]) {
                let jpeg: Vec<u8> = buf.drain(..end + 2).collect();
                let (w0, h0, gray) = crate::motion::decode_to_gray_bytes(&jpeg).unwrap();
                let (w, h, gray) = crate::motion::downscale_gray(&gray, w0, h0, 320);
                let score = cs.prev_frame.as_ref().map(|prev| crate::motion::compute_motion_masked(
                    prev, &gray, &[], settings.motion_threshold, settings.motion_lightning_threshold, w, h).0)
                    .unwrap_or(0.0);
                cs.prev_frame = Some(gray);
                scores.push(score);
                let now = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
                match step(&mut cs, score, &lifecycle, 10.0).0 {
                    Step::Opened(_) => events.push(("opened", now)),
                    Step::Closed { .. } => events.push(("closed", now)),
                    _ => {}
                }
            }
        }
        assert!(child.wait().unwrap().success(), "ffmpeg failed");
        (events, scores)
    }).await.unwrap();

    let kinds: Vec<&str> = events.iter().map(|e| e.0).collect();
    assert_eq!(kinds, ["opened", "closed"],
        "one event should open and close; scores: {:?}", scores.iter().map(|s| (s * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    let (opened, closed) = (events[0].1, events[1].1);
    assert!((closed - opened - 18.0).abs() < 3.0, "the event spans the motion plus the post-buffer: {:.1}s", closed - opened);

    // ffmpeg has exited, so every segment is finished.
    let names = crate::nvr_pipes::finalise_incoming(&nvr, &db, Duration::ZERO, &mut Default::default()).await;
    let segs: Vec<(String, f64)> = sqlx::query_as("SELECT started_at, duration_secs FROM nvr_segments ORDER BY started_at")
        .fetch_all(&db).await.unwrap();
    assert_eq!(segs.len(), names.len());
    assert!(segs.len() >= 3, "22 s in 10 s segments: {segs:?}");
    for (_, d) in &segs[..segs.len() - 1] {
        assert!((9.5..=10.5).contains(d), "a full segment lasts ~10 s: {segs:?}");
    }
    let total: f64 = segs.iter().map(|s| s.1).sum();
    assert!((total - 22.0).abs() < 1.0, "the segments hold the whole recording: {total:.1}s");

    // The event's clip, cut from the indexed segments.
    let clip = dir.join("clip_e2e.mp4");
    assert!(crate::nvr_stream::concat_window_to_file(&db, &dir, 0, opened, closed, &clip).await, "clip export failed");
    let moov = crate::nvr_pipes::read_moov(&mut std::fs::File::open(&clip).unwrap()).expect("a playable clip");
    let len = crate::nvr_pipes::mp4_duration(&moov).unwrap();
    assert!((len - (closed - opened)).abs() < 1.5, "the clip covers the event: {len:.1}s of {:.1}s", closed - opened);
    eprintln!("e2e: event {:.1}s, segments {:?}, clip {len:.1}s",
        closed - opened, segs.iter().map(|s| s.1).collect::<Vec<_>>());

    db.close().await;
    let _ = std::fs::remove_dir_all(&dir);
}
