/**
 * tests/slider.test.tsx
 *
 * React Testing Library tests for the TimeSlider component.
 *
 * Strategy: mock the Zustand store so this test is decoupled from real store
 * state and we can assert exactly which action gets dispatched on slider change.
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';

// ── Store mock ────────────────────────────────────────────────────────────────
// We mock the *module* so TimeSlider consumes our controlled values.

const mockSetTimestep = vi.fn();
const mockTogglePlayback = vi.fn();
const mockIncrementTimestep = vi.fn();

const storeState = {
  currentTimestep: 0,
  maxTimesteps: 95,
  isPlaying: false,
  setTimestep: mockSetTimestep,
  togglePlayback: mockTogglePlayback,
  incrementTimestep: mockIncrementTimestep,
};

vi.mock('@/store/useTimeStore', () => {
  const selector = (fn: (s: typeof storeState) => unknown) => fn(storeState);
  return { default: selector };
});

// ── Helpers ────────────────────────────────────────────────────────────────────
// Import component AFTER mocks are registered.
import TimeSlider from '@/components/TimeSlider';

function renderSlider() {
  return render(<TimeSlider />);
}

beforeEach(() => {
  vi.clearAllMocks();
  storeState.currentTimestep = 0;
  storeState.isPlaying = false;
});

// ── Tests ──────────────────────────────────────────────────────────────────────

describe('TimeSlider rendering', () => {
  it('renders the time label for timestep 0 as "00:00"', () => {
    renderSlider();
    expect(screen.getByText('00:00')).toBeInTheDocument();
  });

  it('renders the play button when not playing', () => {
    renderSlider();
    expect(screen.getByRole('button', { name: /play time animation/i })).toBeInTheDocument();
  });

  it('renders the pause button when playing', () => {
    storeState.isPlaying = true;
    renderSlider();
    expect(screen.getByRole('button', { name: /pause playback/i })).toBeInTheDocument();
  });

  it('shows the correct t= counter text', () => {
    storeState.currentTimestep = 12;
    renderSlider();
    expect(screen.getByText(/t = 12 \/ 95/)).toBeInTheDocument();
  });
});

describe('TimeSlider interactions', () => {
  it('dispatches setTimestep with the new numeric value when the slider changes', () => {
    renderSlider();
    const slider = screen.getByRole('slider', { name: /timestep selector/i });
    fireEvent.change(slider, { target: { value: '24' } });
    expect(mockSetTimestep).toHaveBeenCalledOnce();
    expect(mockSetTimestep).toHaveBeenCalledWith(24);
  });

  it('dispatches togglePlayback when the play button is clicked', () => {
    renderSlider();
    fireEvent.click(screen.getByRole('button', { name: /play time animation/i }));
    expect(mockTogglePlayback).toHaveBeenCalledOnce();
  });

  it('dispatches togglePlayback when the pause button is clicked', () => {
    storeState.isPlaying = true;
    renderSlider();
    fireEvent.click(screen.getByRole('button', { name: /pause playback/i }));
    expect(mockTogglePlayback).toHaveBeenCalledOnce();
  });

  it('slider min/max attributes match maxTimesteps from the store', () => {
    renderSlider();
    const slider = screen.getByRole('slider', { name: /timestep selector/i });
    expect(slider).toHaveAttribute('min', '0');
    expect(slider).toHaveAttribute('max', '95');
  });

  it('slider value reflects currentTimestep from the store', () => {
    storeState.currentTimestep = 48;
    renderSlider();
    const slider = screen.getByRole('slider', { name: /timestep selector/i });
    expect(slider).toHaveAttribute('value', '48');
  });
});
