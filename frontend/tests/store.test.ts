/**
 * tests/store.test.ts
 *
 * Unit tests for useTimeStore (Zustand).
 * Tests run against the real store implementation — no mocking needed.
 * Each test resets the store to a known baseline via setTimestep / setMaxTimesteps
 * so tests are order-independent.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import useTimeStore from '@/store/useTimeStore';
import { makeBundle, makeRegistry, makeRegistryEntry } from './fixtures/v5';

// Reset store state before every test so cases are fully isolated.
beforeEach(() => {
  useTimeStore.setState({
    currentTimestep: 0,
    maxTimesteps: 95,
    isPlaying: false,
    selectedNodeId: null,
    globalAttentionThreshold: 0.5,
  });
});

// ── setTimestep ──────────────────────────────────────────────────────────────

describe('setTimestep', () => {
  it('moves to a valid timestep', () => {
    useTimeStore.getState().setTimestep(42);
    expect(useTimeStore.getState().currentTimestep).toBe(42);
  });

  it('clamps to 0 when given a negative value', () => {
    useTimeStore.getState().setTimestep(-5);
    expect(useTimeStore.getState().currentTimestep).toBe(0);
  });

  it('clamps to maxTimesteps when given a value beyond the upper bound', () => {
    useTimeStore.getState().setTimestep(200);
    expect(useTimeStore.getState().currentTimestep).toBe(95);
  });

  it('accepts exactly 0 (lower boundary)', () => {
    useTimeStore.getState().setTimestep(10);
    useTimeStore.getState().setTimestep(0);
    expect(useTimeStore.getState().currentTimestep).toBe(0);
  });

  it('accepts exactly maxTimesteps (upper boundary)', () => {
    useTimeStore.getState().setTimestep(95);
    expect(useTimeStore.getState().currentTimestep).toBe(95);
  });
});

// ── setMaxTimesteps ──────────────────────────────────────────────────────────

describe('setMaxTimesteps', () => {
  it('updates maxTimesteps', () => {
    useTimeStore.getState().setMaxTimesteps(47);
    expect(useTimeStore.getState().maxTimesteps).toBe(47);
  });

  it('setTimestep respects the new max after setMaxTimesteps', () => {
    useTimeStore.getState().setMaxTimesteps(10);
    useTimeStore.getState().setTimestep(20);
    // Should be clamped to new max of 10
    expect(useTimeStore.getState().currentTimestep).toBe(10);
  });
});

// ── togglePlayback ───────────────────────────────────────────────────────────

describe('togglePlayback', () => {
  it('starts as false', () => {
    expect(useTimeStore.getState().isPlaying).toBe(false);
  });

  it('toggles from false → true', () => {
    useTimeStore.getState().togglePlayback();
    expect(useTimeStore.getState().isPlaying).toBe(true);
  });

  it('toggles from true → false', () => {
    useTimeStore.getState().togglePlayback(); // → true
    useTimeStore.getState().togglePlayback(); // → false
    expect(useTimeStore.getState().isPlaying).toBe(false);
  });
});

// ── incrementTimestep ────────────────────────────────────────────────────────

describe('incrementTimestep', () => {
  it('increments by 1', () => {
    useTimeStore.getState().setTimestep(5);
    useTimeStore.getState().incrementTimestep();
    expect(useTimeStore.getState().currentTimestep).toBe(6);
  });

  it('wraps to 0 when at maxTimesteps', () => {
    useTimeStore.getState().setTimestep(95);
    useTimeStore.getState().incrementTimestep();
    expect(useTimeStore.getState().currentTimestep).toBe(0);
  });

  it('does not overshoot maxTimesteps − 1 before wrapping', () => {
    useTimeStore.getState().setTimestep(94);
    useTimeStore.getState().incrementTimestep(); // → 95 (not wrapping yet)
    expect(useTimeStore.getState().currentTimestep).toBe(95);
    useTimeStore.getState().incrementTimestep(); // → wraps to 0
    expect(useTimeStore.getState().currentTimestep).toBe(0);
  });

  it('wraps correctly with a non-default maxTimesteps', () => {
    useTimeStore.getState().setMaxTimesteps(3);
    useTimeStore.getState().setTimestep(3);
    useTimeStore.getState().incrementTimestep();
    expect(useTimeStore.getState().currentTimestep).toBe(0);
  });
});

// ── setSelectedNodeId ────────────────────────────────────────────────────────

describe('setSelectedNodeId', () => {
  it('starts as null', () => {
    expect(useTimeStore.getState().selectedNodeId).toBeNull();
  });

  it('sets a numeric id', () => {
    useTimeStore.getState().setSelectedNodeId(42);
    expect(useTimeStore.getState().selectedNodeId).toBe(42);
  });

  it('clears back to null', () => {
    useTimeStore.getState().setSelectedNodeId(7);
    useTimeStore.getState().setSelectedNodeId(null);
    expect(useTimeStore.getState().selectedNodeId).toBeNull();
  });
});

// ── setGlobalAttentionThreshold ──────────────────────────────────────────────

describe('setGlobalAttentionThreshold', () => {
  it('defaults to 0.5', () => {
    expect(useTimeStore.getState().globalAttentionThreshold).toBeCloseTo(0.5);
  });

  it('accepts an in-range value', () => {
    useTimeStore.getState().setGlobalAttentionThreshold(0.4);
    expect(useTimeStore.getState().globalAttentionThreshold).toBeCloseTo(0.4);
  });

  it('clamps below 0', () => {
    useTimeStore.getState().setGlobalAttentionThreshold(-0.5);
    expect(useTimeStore.getState().globalAttentionThreshold).toBe(0);
  });

  it('clamps above 1', () => {
    useTimeStore.getState().setGlobalAttentionThreshold(2);
    expect(useTimeStore.getState().globalAttentionThreshold).toBe(1);
  });
});

// ── v5 payload state ─────────────────────────────────────────────────────────

describe('v5 payload state', () => {
  it('no longer exposes the v4 simulated-node toggle', () => {
    const s = useTimeStore.getState() as unknown as Record<string, unknown>;
    expect('showSimulatedNodes' in s).toBe(false);
    expect('setShowSimulatedNodes' in s).toBe(false);
  });

  it('setRunRegistry stores runs and the selection rule', () => {
    const reg = makeRegistry([makeRegistryEntry({ runId: 'a', rank: 1 }), makeRegistryEntry({ runId: 'b', rank: 2 })]);
    useTimeStore.getState().setRunRegistry(reg.runs, reg.selection_rule);
    expect(useTimeStore.getState().runRegistry.map((r) => r.run_id)).toEqual(['a', 'b']);
    expect(useTimeStore.getState().selectionRule).toBe(reg.selection_rule);
    useTimeStore.getState().setRunRegistry([]);
    expect(useTimeStore.getState().selectionRule).toBeNull();
  });

  it('setSpatial / setDayPayload hold one run-consistent bundle', () => {
    const { spatial, page } = makeBundle({ runId: 'v5_full-s42' });
    useTimeStore.getState().setSpatial(spatial);
    useTimeStore.getState().setDayPayload({ runId: 'v5_full-s42', date: page.date, spatial, page });
    const st = useTimeStore.getState();
    expect(st.spatial).toBe(spatial);
    expect(st.dayPayload?.page).toBe(page);
    expect(st.dayPayload?.spatial.metadata.run_id).toBe('v5_full-s42');
    useTimeStore.getState().setDayPayload(null);
    expect(useTimeStore.getState().dayPayload).toBeNull();
  });

  it('navigateDay steps through availableDates and stops at the bounds', () => {
    const dates = ['2026-05-01', '2026-05-02', '2026-05-03'];
    useTimeStore.setState({ availableDates: dates, activeDate: dates[0] });
    const { navigateDay } = useTimeStore.getState();
    navigateDay('prev');
    expect(useTimeStore.getState().activeDate).toBe(dates[0]);
    navigateDay('next');
    navigateDay('next');
    expect(useTimeStore.getState().activeDate).toBe(dates[2]);
    navigateDay('next');
    expect(useTimeStore.getState().activeDate).toBe(dates[2]);
  });
});
