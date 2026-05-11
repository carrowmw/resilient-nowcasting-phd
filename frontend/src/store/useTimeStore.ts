import { create } from 'zustand';
import type { AttentionEdge } from '@/types/payload';

interface TimeState {
  currentTimestep: number;
  maxTimesteps: number;
  isPlaying: boolean;
  selectedNodeId: number | null;
  globalAttentionThreshold: number;
  egoGraphEdges: AttentionEdge[] | null;

  setTimestep: (step: number) => void;
  setMaxTimesteps: (max: number) => void;
  togglePlayback: () => void;
  incrementTimestep: () => void;
  setSelectedNodeId: (id: number | null) => void;
  setGlobalAttentionThreshold: (v: number) => void;
  setEgoGraphEdges: (edges: AttentionEdge[] | null) => void;
}

const useTimeStore = create<TimeState>((set, get) => ({
  currentTimestep: 0,
  maxTimesteps: 95,
  isPlaying: false,
  selectedNodeId: null,
  globalAttentionThreshold: 0.5,
  egoGraphEdges: null,

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
}));

export default useTimeStore;
