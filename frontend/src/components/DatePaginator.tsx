"use client";

import type { CSSProperties } from "react";
import useTimeStore from "@/store/useTimeStore";

/**
 * Prev/next day navigation over the run's available_dates. Rendered inside the
 * TimeSlider header. Disables at boundary dates and shows a spinner while the
 * next page.json is loading.
 */
export default function DatePaginator() {
  const availableDates = useTimeStore((s) => s.availableDates);
  const activeDate = useTimeStore((s) => s.activeDate);
  const isLoadingPage = useTimeStore((s) => s.isLoadingPage);
  const navigateDay = useTimeStore((s) => s.navigateDay);

  if (!activeDate || availableDates.length === 0) {
    return <span style={{ color: "#64748b" }}>—</span>;
  }

  const idx = availableDates.indexOf(activeDate);
  const atStart = idx <= 0;
  const atEnd = idx < 0 || idx >= availableDates.length - 1;

  const btn = (disabled: boolean): CSSProperties => ({
    background: "transparent",
    border: "none",
    color: disabled ? "#475569" : "#cbd5e1",
    cursor: disabled ? "default" : "pointer",
    fontSize: 14,
    padding: "0 6px",
    lineHeight: 1,
  });

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
      <button
        type="button"
        onClick={() => navigateDay("prev")}
        disabled={atStart || isLoadingPage}
        aria-label="Previous day"
        style={btn(atStart || isLoadingPage)}
      >
        ◀
      </button>
      <span
        style={{
          minWidth: 92,
          textAlign: "center",
          color: "#e2e8f0",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {isLoadingPage ? "loading…" : activeDate}
      </span>
      <button
        type="button"
        onClick={() => navigateDay("next")}
        disabled={atEnd || isLoadingPage}
        aria-label="Next day"
        style={btn(atEnd || isLoadingPage)}
      >
        ▶
      </button>
    </div>
  );
}
