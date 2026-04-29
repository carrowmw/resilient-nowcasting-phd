import { create } from 'zustand';

interface TimeState {
  currentTimestep: number;
  maxTimesteps: number;
  isPlaying: boolean;
  selectedNodeId: number | null;
  attentionThreshold: number;

  setTimestep: (step: number) => void;
  setMaxTimesteps: (max: number) => void;
  togglePlayback: () => void;
  incrementTimestep: () => void;
  setSelectedNodeId: (id: number | null) => void;
  setAttentionThreshold: (v: number) => void;
}

const useTimeStore = create<TimeState>((set, get) => ({
  currentTimestep: 0,
  maxTimesteps: 95,
  isPlaying: false,
  selectedNodeId: null,
  attentionThreshold: 0.05,

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
  setAttentionThreshold: (v: number) =>
    set({ attentionThreshold: Math.max(0, Math.min(1, v)) }),
}));

export default useTimeStore;
