/**
 * tests/analytics.test.tsx
 *
 * Capability gating in the analytics tabs: null fields render an explicit
 * "not available" state, never a zero-filled chart.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { render, screen } from '@testing-library/react';
import CalibrationTab from '@/components/analytics/CalibrationTab';
import NetworkTab from '@/components/analytics/NetworkTab';
import ForecastTab from '@/components/analytics/ForecastTab';
import { BASELINE_CAPABILITIES, FULL_CAPABILITIES, makeBundle } from './fixtures/v5';

beforeAll(() => {
  // recharts' ResponsiveContainer needs ResizeObserver (absent in jsdom).
  globalThis.ResizeObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
});

describe('CalibrationTab', () => {
  it('shows N/A for every calibration view of a baseline model', () => {
    const { page, spatial } = makeBundle({ capabilities: BASELINE_CAPABILITIES });
    render(
      <CalibrationTab
        calibration={page.calibration}
        dayMetrics={page.day_metrics}
        windowMetrics={spatial.metadata.window_metrics}
        capabilities={BASELINE_CAPABILITIES}
      />,
    );
    expect(screen.getByText('PIT histogram not available for this model.')).toBeInTheDocument();
    expect(screen.getByText('p_zero reliability not available for this model.')).toBeInTheDocument();
    expect(screen.getByText('Outage reliability not available for this model.')).toBeInTheDocument();
    // Null coverage renders as an em dash, not 0%.
    expect(screen.getAllByText('— · —').length).toBeGreaterThanOrEqual(4);
  });

  it('renders charts instead of N/A when the model has the capabilities', () => {
    const { page, spatial } = makeBundle();
    render(
      <CalibrationTab
        calibration={page.calibration}
        dayMetrics={page.day_metrics}
        windowMetrics={spatial.metadata.window_metrics}
        capabilities={FULL_CAPABILITIES}
      />,
    );
    expect(screen.queryByText(/not available for this model/)).toBeNull();
  });

  it('gates outage reliability on capabilities.outage even if data is present', () => {
    const { page, spatial } = makeBundle();
    const caps = { ...FULL_CAPABILITIES, outage: false };
    render(
      <CalibrationTab
        calibration={page.calibration}
        dayMetrics={page.day_metrics}
        windowMetrics={spatial.metadata.window_metrics}
        capabilities={caps}
      />,
    );
    expect(screen.getByText('Outage reliability not available for this model.')).toBeInTheDocument();
  });
});

describe('NetworkTab', () => {
  it('shows N/A for attention views without a graph', () => {
    const { page } = makeBundle({ capabilities: BASELINE_CAPABILITIES });
    render(<NetworkTab page={{ ...page, tripwire_audit_trail: [] }} capabilities={BASELINE_CAPABILITIES} onSelectNode={() => {}} />);
    expect(screen.getByText('Attention leaderboard not available for this model.')).toBeInTheDocument();
    expect(screen.getByText('Attention not available for this model.')).toBeInTheDocument();
    expect(screen.getByText(`No tripwire events on ${page.date}.`)).toBeInTheDocument();
  });
});

describe('ForecastTab', () => {
  it('prompts for a node selection when none is selected', () => {
    const { page } = makeBundle();
    render(<ForecastTab page={page} node={null} capabilities={FULL_CAPABILITIES} />);
    expect(screen.getByText(/Select a node/)).toBeInTheDocument();
  });

  it('renders node metrics with em dashes for missing coverage', () => {
    const { page, spatial } = makeBundle({ capabilities: BASELINE_CAPABILITIES });
    render(<ForecastTab page={page} node={spatial.nodes[0]} capabilities={BASELINE_CAPABILITIES} />);
    expect(screen.getByText(spatial.nodes[0].location_name)).toBeInTheDocument();
    expect(screen.getByText('no interval')).toBeInTheDocument();
    expect(screen.queryByLabelText('Probability chart')).toBeNull();
  });
});
