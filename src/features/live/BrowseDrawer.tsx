// v18: Combined Events + Timeline drawer. The two sections are rendered
// independently — the user can show one, the other, or both at once. The
// `panels` prop says which sections are active.

import { MotionEvent } from "../../types";
import type { PanelState } from "./FocusHeader";

import { EventsListPanel } from "./EventsListPanel";
import { HistoryDrawer } from "./HistoryDrawer";

import styles from "./BrowseDrawer.module.css";

interface Props {
  camId: number;
  panels: PanelState;
  selectedDate: Date;
  selectedEventId: string | null;
  positionMs: number | null;
  events: MotionEvent[];
  onSelectEvent: (ev: MotionEvent) => void;
  onSeek: (ms: number) => void;
  viewStart?: number;
  viewEnd?: number;
  onSetView?: (start: number, end: number) => void;
  use12h?: boolean;
}

export function BrowseDrawer({
  camId, panels, selectedDate, selectedEventId, positionMs, events,
  onSelectEvent, onSeek,
  viewStart, viewEnd, onSetView, use12h,
}: Props) {
  return (
    <div className={styles.root}>
      {panels.events && (
        <section className={styles.section}>
          <EventsListPanel
            camId={camId}
            events={events}
            selectedDate={selectedDate}
            selectedEventId={selectedEventId}
            onSelectEvent={onSelectEvent}
            use12h={use12h}
          />
        </section>
      )}
      {panels.timeline && (
        <section className={styles.section}>
          <HistoryDrawer
            camId={camId}
            selectedDate={selectedDate}
            positionMs={positionMs}
            events={events}
            onSeek={onSeek}
            onSelectEvent={onSelectEvent}
            selectedEventId={selectedEventId ?? undefined}
            viewStart={viewStart}
            viewEnd={viewEnd}
            onSetView={onSetView}
            use12h={use12h}
          />
        </section>
      )}
    </div>
  );
}
