import { create } from 'zustand';

interface TimeState {
  currentTimestep: number;
  maxTimesteps: number;
  isPlaying: boolean;

  setTimestep: (step: number) => void;
  setMaxTimesteps: (max: number) => void;
  togglePlayback: () => void;
  incrementTimestep: () => void;
}

const useTimeStore = create<TimeState>((set, get) => ({
  currentTimestep: 0,
  maxTimesteps: 95,
  isPlaying: false,

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
}));

export default useTimeStore;
