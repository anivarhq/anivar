// v14: Export dropdown menu — replaces the Export tab in the (deleted)
// CameraHistoryModal. Modeled on ShareLiveButton (absolute-positioned menu,
// click-outside to close). Preset rows + Custom range expand-in-place.

import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Download } from "lucide-react";

import { useStore } from "../../store";
import { useShallow } from "zustand/react/shallow";
import type { MotionEvent } from "../../types";
import {
  exportCustom,
  exportEvent,
  exportRelative,
  safeName,
  toLocalInput,
} from "./cameraExport";

import styles from "./FocusHeader.module.css";

interface Props {
  camId: number;
  cameraName: string;
  selectedEvent: MotionEvent | null;
  viewportRange?: { start: number; end: number };
}

const PRESETS: Array<{ label: string; secs: number }> = [
  { label: "Last 1 minute",  secs: 60 },
  { label: "Last 10 minutes", secs: 600 },
  { label: "Last hour",      secs: 3600 },
  { label: "Last 24 hours",  secs: 86_400 },
];

export function ExportDropdown({ camId, cameraName, selectedEvent, viewportRange }: Props) {
  const { streamInfo, showToast } = useStore(useShallow(s => ({ streamInfo: s.streamInfo, showToast: s.showToast })));
  const [open, setOpen] = useState(false);
  const [customOpen, setCustomOpen] = useState(false);
  const [customStart, setCustomStart] = useState("");
  const [customEnd, setCustomEnd] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);

  const camLabel = useMemo(() => safeName(cameraName || `cam${camId}`), [cameraName, camId]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
        setCustomOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setOpen(false); setCustomOpen(false); }
    };
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  useEffect(() => {
    if (!customOpen) return;
    const r = viewportRange ?? { start: Date.now() - 3600_000, end: Date.now() };
    setCustomStart(toLocalInput(r.start));
    setCustomEnd(toLocalInput(r.end));
  }, [customOpen, viewportRange]);

  const handlePreset = (secs: number) => {
    if (!streamInfo) { showToast("Server not ready", "error"); return; }
    setOpen(false);
    void exportRelative(streamInfo, camId, camLabel, secs, showToast);
  };

  const handleEvent = () => {
    if (!streamInfo || !selectedEvent) return;
    setOpen(false);
    void exportEvent(streamInfo, camLabel, selectedEvent, showToast);
  };

  const handleCustom = () => {
    if (!streamInfo) return;
    setOpen(false);
    void exportCustom(streamInfo, camId, camLabel, customStart, customEnd, showToast);
  };

  return (
    <div ref={rootRef} className={styles.dropdownRoot}>
      <button
        className={`${styles.headerBtn} ${open ? styles.headerBtnActive : ""}`}
        onClick={() => setOpen(v => !v)}
        title="Export video"
      >
        <Download size={13} />
        <span>Export</span>
        <ChevronDown size={11} style={{ marginLeft: 2, opacity: 0.7 }} />
      </button>
      {open && (
        <div className={styles.dropdownMenu} role="menu">
          {PRESETS.map(p => (
            <button key={p.secs} className={styles.dropdownItem}
              onClick={() => handlePreset(p.secs)}>
              {p.label}
            </button>
          ))}
          <div className={styles.dropdownSep} />
          <button className={styles.dropdownItem}
            disabled={!selectedEvent}
            onClick={handleEvent}>
            {selectedEvent
              ? `This event (${selectedEvent.id.slice(0, 8)})`
              : "This event — none selected"}
          </button>
          <div className={styles.dropdownSep} />
          {!customOpen ? (
            <button className={styles.dropdownItem}
              onClick={() => setCustomOpen(true)}>
              Custom range…
            </button>
          ) : (
            <div className={styles.customPicker}>
              <label className={styles.customLabel}>
                Start
                <input type="datetime-local" value={customStart}
                  onChange={e => setCustomStart(e.target.value)} />
              </label>
              <label className={styles.customLabel}>
                End
                <input type="datetime-local" value={customEnd}
                  onChange={e => setCustomEnd(e.target.value)} />
              </label>
              <div className={styles.customActions}>
                <button className={styles.customCancel}
                  onClick={() => setCustomOpen(false)}>Cancel</button>
                <button className={styles.customSave}
                  onClick={handleCustom}>Stitch &amp; Save</button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
