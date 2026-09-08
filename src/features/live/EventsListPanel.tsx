// v25: Wraps NVR's ThumbnailStrip so the focus-mode Events drawer renders
// the SAME cards as the main NVR tab. We do the camera + date filter here;
// the strip itself handles auto-scroll, hover, active state, risk pill,
// time label, AI summary snippet.

import { useMemo } from "react";

import type { MotionEvent } from "../../types";
import { ThumbnailStrip } from "../nvr/NVRPanel";
import { dayStartMs, dayEndMs } from "../../lib/time";

import styles from "./EventsListPanel.module.css";

interface Props {
  camId: number;
  events: MotionEvent[];
  selectedDate?: Date;
  selectedEventId: string | null;
  onSelectEvent: (ev: MotionEvent) => void;
  /** Event ids with no recorded video (shown dimmed + "No video" badge). */
  noFootageIds?: Set<string>;
  /** 12-hour clock, matching the header toggle. */
  use12h?: boolean;
}

export function EventsListPanel({ camId, events, selectedDate, selectedEventId, onSelectEvent, noFootageIds, use12h = false }: Props) {
  const camEvents = useMemo(() => {
    let dayStart = -Infinity, dayEnd = Infinity;
    if (selectedDate) {
      dayStart = dayStartMs(selectedDate);
      dayEnd   = dayEndMs(selectedDate);
    }
    return events
      .filter(ev => (ev.cam_id ?? 0) === camId)
      // Timelines/strips visualize VIDEO events only; sounds live in the
      // Review feed + Audio tab (an audio event's score is YAMNet confidence,
      // which painted misleading red risk pills here).
      .filter(ev => ev.event_category !== "audio")
      .filter(ev => {
        const t = new Date(ev.started_at).getTime();
        return t >= dayStart && t <= dayEnd;
      })
      .slice(0, 100);
  }, [events, camId, selectedDate]);

  if (camEvents.length === 0) {
    return (
      <div className={styles.empty}>
        <span>No events for this camera on the selected date.</span>
      </div>
    );
  }

  // `use12h` was hardcoded false here, so the header's 12h/24h toggle changed
  // the timeline but not the event cards sitting right beside it.
  return (
    <ThumbnailStrip
      events={camEvents}
      activeEvId={selectedEventId}
      onSelect={onSelectEvent}
      use12h={use12h}
      layout="grid"
      noFootageIds={noFootageIds}
    />
  );
}
