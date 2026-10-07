"use client";

import { useEffect, useState } from "react";
import useTimeStore from "@/store/useTimeStore";
import { registryUrl, spatialUrl, pageUrl } from "@/lib/payloadSource";
import {
  assertPage,
  assertRegistry,
  assertSpatial,
  errorMessage,
  fetchPayload,
  isAbortError,
} from "@/lib/v5";

/**
 * Three-stage payload loading shared by the map and the analytics page:
 *
 *   1. run_registry.json → runRegistry + default activeRunId (skipped if loaded)
 *   2. spatial.json      → store.spatial + availableDates (on run change)
 *   3. page.json         → store.dayPayload {spatial, page} (on run/date change)
 *
 * Everything lands in the zustand store, so navigating between routes reuses
 * the already-loaded payloads instead of refetching. Returns the latest error.
 */
export default function usePayloadLoader(): string | null {
  const [error, setError] = useState<string | null>(null);

  const registryLoaded = useTimeStore((s) => s.runRegistry.length > 0);
  const activeRunId = useTimeStore((s) => s.activeRunId);
  const activeDate = useTimeStore((s) => s.activeDate);
  const spatial = useTimeStore((s) => s.spatial);

  // ── Stage 1: registry ─────────────────────────────────────────────────────
  useEffect(() => {
    if (registryLoaded) return;
    const controller = new AbortController();
    fetchPayload(registryUrl(), assertRegistry, controller.signal)
      .then((registry) => {
        setError(null);
        const runs = [...registry.runs].sort((a, b) => a.registry.rank - b.registry.rank);
        const s = useTimeStore.getState();
        s.setRunRegistry(runs, registry.selection_rule);
        if (!s.activeRunId) {
          const def = registry.default_run_id ?? runs[0]?.run_id ?? null;
          if (def) s.setActiveRunId(def);
        }
      })
      .catch((err) => {
        if (isAbortError(err)) return;
        setError(`Failed to load run registry: ${errorMessage(err)}`);
      });
    return () => controller.abort();
  }, [registryLoaded]);

  // ── Stage 2: spatial index for the active run ─────────────────────────────
  useEffect(() => {
    if (!activeRunId) return;
    if (useTimeStore.getState().spatial?.metadata.run_id === activeRunId) return;
    const controller = new AbortController();
    fetchPayload(spatialUrl(activeRunId), assertSpatial, controller.signal)
      .then((sp) => {
        setError(null);
        const s = useTimeStore.getState();
        const dates = sp.metadata.available_dates ?? [];
        s.setAvailableDates(dates);
        // Keep the current day when the new run covers it (model comparison).
        if (dates.length > 0 && (!s.activeDate || !dates.includes(s.activeDate))) {
          s.setActiveDate(dates[0]);
        }
        s.setSpatial(sp);
      })
      .catch((err) => {
        if (isAbortError(err)) return;
        setError(`Failed to load spatial index for ${activeRunId}: ${errorMessage(err)}`);
      });
    return () => controller.abort();
  }, [activeRunId]);

  // ── Stage 3: page for the active run + date ───────────────────────────────
  useEffect(() => {
    if (!activeRunId || !activeDate || !spatial) return;
    // Run-switch race: spatial may still be the previous run's.
    if (spatial.metadata.run_id !== activeRunId) return;
    const loaded = useTimeStore.getState().dayPayload;
    if (
      loaded &&
      loaded.runId === activeRunId &&
      loaded.date === activeDate &&
      loaded.spatial === spatial
    ) {
      return;
    }
    const controller = new AbortController();
    const s = useTimeStore.getState();
    s.setIsLoadingPage(true);
    fetchPayload(pageUrl(activeRunId, activeDate), assertPage, controller.signal)
      .then((page) => {
        setError(null);
        const st = useTimeStore.getState();
        st.setDayPayload({ runId: activeRunId, date: activeDate, spatial, page });
        st.setMaxTimesteps(Math.max(0, page.timesteps - 1));
        st.setTimestep(st.currentTimestep); // re-clamp to the new day length
        st.setIsLoadingPage(false);
      })
      .catch((err) => {
        if (isAbortError(err)) return;
        useTimeStore.getState().setIsLoadingPage(false);
        setError(`Failed to load page for ${activeRunId}/${activeDate}: ${errorMessage(err)}`);
      });
    return () => {
      controller.abort();
      useTimeStore.getState().setIsLoadingPage(false);
    };
  }, [activeRunId, activeDate, spatial]);

  return error;
}
