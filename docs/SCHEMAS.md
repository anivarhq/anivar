# Stored data formats

## `motion_events.ai_summary`

What Anivar concluded about an event, as a JSON object in a `TEXT` column. It's written once the event closes and has been analysed (`agent/clip.rs`), and read by the Review feed, search, the assistant and Telegram.

A row holds one of three shapes:

| Shape | When it's written |
|---|---|
| **v3**: the v2 fields plus `attributes` | After a model analysed the event (the normal path). |
| **v2**: the base fields only | When no assistant model is configured: the summary is built from the detections alone (`fallback_detection_summary`). |
| Plain text, not JSON | Rows from before v2. |

Readers handle all three. To get the human-readable text, use `extract_summary_text` (Rust) or `aiText` (`src/lib/eventFormat.ts`). Both prefer `text`, then `description`, and never show raw JSON.

### Fields

v2 fields are the base; v3 adds `attributes`.

| Field | Type | Since | Meaning |
|---|---|---|---|
| `v` | integer | v2 | Format version: `2` or `3`. |
| `title` | string | v2 | Short headline from the model, e.g. "Courier at the front door". Empty on the detections-only path. |
| `risk` | string | v2 | `"normal"`, `"monitor"`, `"suspicious"` or `"critical"`. Alerts are gated on it (`alert_min_risk`). |
| `type` | string | v2 | Threat or event type, e.g. `"person"`, `"vehicle"`, `"loitering"`, `"intrusion"`. |
| `text` | string | v2 | The summary people read. |
| `description` | string | v2 | Description of the person, if one is present (age range, build, clothing). Empty otherwise. |
| `objects` | string[] | v2 | Things the person carried or introduced, plus non-person detections. |
| `confidence` | number | v2 | The model's confidence, 0 to 1. The detections-only path writes 0.4. |
| `fp` | boolean | v2 | The model judged the event a false positive. |
| `persons` | integer | v2 | Number of people seen. |
| `attributes` | object[] | v3 | Every refinement found on the event; see below. The same list is stored as its own column, `motion_events.attributes`. |

### `attributes` entries

| `type` | `value` | Other keys |
|---|---|---|
| `"face"` | A recognised person's name | `score`: recognition score (raw ArcFace cosine) |
| `"plate"` | The plate as read | `score`: OCR confidence. `known_name`: the owner from Settings → Known plates, when it matches. |
| `"color"` | A vehicle's body colour | `score`: share of the crop |
| `"outfit"` | A clothing phrase, e.g. "red top, black trousers" | (none) |
| `"object"` | A carried or security-relevant object | (none) |

### Example (v3)

A test (`agent::types::schema_doc_tests`) parses this example and checks it has exactly the fields the code writes. If the format changes, change it here too.

```json
{
  "v": 3,
  "title": "Courier leaves a parcel",
  "risk": "monitor",
  "type": "person",
  "text": "A courier in a red top left a parcel at the door and walked back to a white van.",
  "description": "adult, medium build, red top, black trousers",
  "objects": ["parcel"],
  "confidence": 0.82,
  "fp": false,
  "persons": 1,
  "attributes": [
    { "type": "face", "value": "Amma", "score": 0.61 },
    { "type": "plate", "value": "KA01AB1234", "score": 0.93, "known_name": "Courier van" },
    { "type": "color", "value": "white", "score": 0.71 },
    { "type": "outfit", "value": "red top, black trousers" },
    { "type": "object", "value": "parcel" }
  ]
}
```
