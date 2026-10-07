import { create } from 'zustand';
import type { AttentionEdge, TemporalPage } from '@/types/payload';
import type { RunRegistryEntry, SpatialIndex } from '@/types/registry';

/** One loaded day of one run: spatial index + page, always for the same run. */
export interface DayPayload {
  runId: string;
  date: string;
  spatial: SpatialIndex;
  page: TemporalPage;
}

interface TimeState {
  currentTimestep: number;
  maxTimesteps: number;
  isPlaying: boolean;
  selectedNodeId: number | null;
  globalAttentionThreshold: number;
  egoGraphEdges: AttentionEdge[] | null;

  // --- Multi-run + pagination state ---
  activeRunId: string | null;
  activeDate: string | null;            // "2026-05-01"
  availableDates: string[];             // sorted, from spatial.json metadata
  runRegistry: RunRegistryEntry[];      // from run_registry.json, ordered by rank
  selectionRule: string | null;         // run_registry.json selection_rule
  isLoadingPage: boolean;               // true during a page.json fetch

  // --- Loaded payloads (shared by the map, overlays and analytics page) ---
  spatial: SpatialIndex | null;         // latest spatial.json (may lead dayPayload on run switch)
  dayPayload: DayPayload | null;        // spatial + page of the same run, set atomically

  setTimestep: (step: number) => void;
  setMaxTimesteps: (max: number) => void;
  togglePlayback: () => void;
  incrementTimestep: () => void;
  setSelectedNodeId: (id: number | null) => void;
  setGlobalAttentionThreshold: (v: number) => void;
  setEgoGraphEdges: (edges: AttentionEdge[] | null) => void;

  setActiveRunId: (runId: string) => void;
  setActiveDate: (date: string) => void;
  setAvailableDates: (dates: string[]) => void;
  setRunRegistry: (runs: RunRegistryEntry[], selectionRule?: string | null) => void;
  setIsLoadingPage: (loading: boolean) => void;
  navigateDay: (direction: 'prev' | 'next') => void;
  setSpatial: (spatial: SpatialIndex | null) => void;
  setDayPayload: (payload: DayPayload | null) => void;
}

const useTimeStore = create<TimeState>((set, get) => ({
  currentTimestep: 0,
  maxTimesteps: 95,
  isPlaying: false,
  selectedNodeId: null,
  globalAttentionThreshold: 0.5,
  egoGraphEdges: null,

  activeRunId: null,
  activeDate: null,
  availableDates: [],
  runRegistry: [],
  selectionRule: null,
  isLoadingPage: false,

  spatial: null,
  dayPayload: null,

  setTimestep: (step: number) =>
    set({ currentTimestep: Math.max(0, Math.min(step, get().maxTimesteps)) }),

  setMaxTimesteps: (max: number) => set({ maxTimesteps: max }),

  togglePlayback: () => set((state) => ({ isPlaying: !state.isPlaying })),

  incrementTimestep: () =>
    set((state) => ({
      currentTimestep:
        state.currentTimestep >= state.maxTimesteps
          ? 0
          : state.currentTimestep + 1,
    })),

  setSelectedNodeId: (id: number | null) => set({ selectedNodeId: id }),

  // Clamp threshold to [0, 1] — attention weights live in [0, 1].
  setGlobalAttentionThreshold: (v: number) =>
    set({ globalAttentionThreshold: Math.max(0, Math.min(1, v)) }),

  setEgoGraphEdges: (edges: AttentionEdge[] | null) => set({ egoGraphEdges: edges }),

  setActiveRunId: (runId: string) => set({ activeRunId: runId }),

  setActiveDate: (date: string) => set({ activeDate: date }),

  setAvailableDates: (dates: string[]) => set({ availableDates: dates }),

  setRunRegistry: (runs: RunRegistryEntry[], selectionRule: string | null = null) =>
    set({ runRegistry: runs, selectionRule }),

  setIsLoadingPage: (loading: boolean) => set({ isLoadingPage: loading }),

  navigateDay: (direction: 'prev' | 'next') => {
    const { availableDates, activeDate } = get();
    if (!activeDate || availableDates.length === 0) return;
    const idx = availableDates.indexOf(activeDate);
    if (idx === -1) return;
    const nextIdx = direction === 'next' ? idx + 1 : idx - 1;
    if (nextIdx < 0 || nextIdx >= availableDates.length) return; // boundary
    set({ activeDate: availableDates[nextIdx] });
  },

  setSpatial: (spatial: SpatialIndex | null) => set({ spatial }),

  setDayPayload: (payload: DayPayload | null) => set({ dayPayload: payload }),
}));

export default useTimeStore;
