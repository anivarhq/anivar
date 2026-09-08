//! Multi-camera correlation + anomaly detection (face-sighting across cameras, duration-based loitering checks).

use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::AppState;


/// Insert one face sighting. Shared by the live `record_face_sighting` command
/// AND the server-side event recognition path (`agent::clip`), so cross-camera
/// context is populated even when NO camera window is open — previously only the
/// frontend recorded sightings, leaving the headless path's cross-camera block empty.
pub async fn insert_face_sighting(
    db: &sqlx::SqlitePool,
    person_name: &str,
    person_id: Option<&str>,
    camera_id: i64,
    event_id: Option<&str>,
    confidence: f32,
    method: &str,
) -> anyhow::Result<()> {
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO face_sightings(id, person_name, person_id, camera_id, event_id, seen_at, confidence, method)
         VALUES(?,?,?,?,?,?,?,?)"
    )
    .bind(&id).bind(person_name).bind(person_id).bind(camera_id)
    .bind(event_id).bind(&now).bind(confidence).bind(method)
    .execute(db).await?;
    Ok(())
}

#[tauri::command]
pub async fn record_face_sighting(
    state: State<'_, Arc<AppState>>,
    person_name: String,
    camera_id: i64,
    event_id: Option<String>,
    confidence: f32,
) -> Result<(), String> {
    // Frontend-originated sighting: resolve the id from the name best-effort
    // (the command's callers predate the id contract).
    let pid: Option<String> = sqlx::query_scalar("SELECT id FROM known_persons WHERE name=? LIMIT 1")
        .bind(&person_name).fetch_optional(&state.db).await.ok().flatten();
    insert_face_sighting(&state.db, &person_name, pid.as_deref(), camera_id, event_id.as_deref(), confidence, "manual")
        .await.map_err(|e| e.to_string())
}

#[derive(Debug, Serialize)]
pub struct CameraCorrelation {
    pub person_name: String,
    pub sightings: Vec<Sighting>,
}

#[derive(Debug, Serialize)]
pub struct Sighting {
    pub camera_id: i64,
    pub seen_at: String,
    pub confidence: f32,
    pub event_id: Option<String>,
}

#[tauri::command]
pub async fn get_camera_correlations(
    state: State<'_, Arc<AppState>>,
    since_hours: Option<i64>,
) -> Result<Vec<CameraCorrelation>, String> {
    let hours = since_hours.unwrap_or(24);
    let rows: Vec<(String, i64, String, f32, Option<String>)> = sqlx::query_as(
        "SELECT person_name, camera_id, seen_at, confidence, event_id
         FROM face_sightings
         WHERE seen_at > datetime('now', ? || ' hours')
         ORDER BY person_name, seen_at DESC"
    )
    .bind(format!("-{hours}"))
    .fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    // Group by person
    let mut map: std::collections::HashMap<String, Vec<Sighting>> = std::collections::HashMap::new();
    for (name, cam, seen_at, conf, eid) in rows {
        map.entry(name).or_default().push(Sighting {
            camera_id: cam, seen_at, confidence: conf, event_id: eid
        });
    }

    // Every person with at least one sighting, newest-first trail intact.
    //
    // The comment here used to say "only persons seen on 2+ cameras" while the
    // predicate was `!cams.is_empty()`, which never filters anything — two
    // readings of the same six lines. Returning everyone is the RIGHT behaviour
    // (a single-camera trail is still the answer to "where was this person?"),
    // so the code stays and the comment is now true. Callers that want the
    // multi-camera view filter on `sightings` themselves.
    let mut result: Vec<CameraCorrelation> = map.into_iter()
        .filter(|(_, sightings)| !sightings.is_empty())
        .map(|(person_name, sightings)| CameraCorrelation { person_name, sightings })
        .collect();
    result.sort_by(|a, b| a.person_name.cmp(&b.person_name));
    Ok(result)
}

#[derive(Debug, Serialize)]
pub struct AnomalyResult {
    pub event_id: String,
    pub started_at: String,
    pub anomaly_type: String,  // "loitering" | "rapid_movement" | "crowd" | "none"
    pub duration_secs: f64,
    pub peak_score: f32,
    pub detail: String,
}

#[tauri::command]
pub async fn detect_anomalies(
    state: State<'_, Arc<AppState>>,
    since_hours: Option<i64>,
) -> Result<Vec<AnomalyResult>, String> {
    let hours = since_hours.unwrap_or(24);
    let rows: Vec<(String, String, Option<f64>, f32, Option<String>)> = sqlx::query_as(
        "SELECT id, started_at, duration_secs, peak_score, detections
         FROM motion_events
         WHERE ended_at IS NOT NULL
           AND started_at > datetime('now', ? || ' hours')
         ORDER BY started_at DESC"
    )
    .bind(format!("-{hours}"))
    .fetch_all(&state.db).await.map_err(|e| e.to_string())?;

    let mut anomalies = Vec::new();
    for (id, started_at, duration, peak, dets_json) in rows {
        let dur = duration.unwrap_or(0.0);

        // Count persons in detections
        let person_count = dets_json.as_deref().map(|j| {
            #[derive(serde::Deserialize)]
            struct D { label: String }
            serde_json::from_str::<Vec<D>>(j).unwrap_or_default()
                .iter().filter(|d| d.label == "person").count()
        }).unwrap_or(0);

        let (anomaly_type, detail) = if dur > 180.0 && peak < 0.15 {
            ("loitering", format!("Subject stationary for {:.0}s with minimal movement (score {:.0}%)", dur, peak * 100.0))
        } else if dur > 120.0 && peak < 0.25 {
            ("loitering", format!("Prolonged presence {:.0}s — possible loitering", dur))
        } else if dur < 20.0 && peak > 0.65 {
            ("rapid_movement", format!("Sudden high-speed movement — peak score {:.0}% in {:.0}s", peak * 100.0, dur))
        } else if person_count >= 3 {
            ("crowd", format!("{person_count} persons detected simultaneously"))
        } else {
            continue; // no anomaly
        };

        anomalies.push(AnomalyResult {
            event_id: id,
            started_at,
            anomaly_type: anomaly_type.to_string(),
            duration_secs: dur,
            peak_score: peak,
            detail,
        });
    }
    Ok(anomalies)
}
