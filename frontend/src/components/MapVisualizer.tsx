'use client';

import { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { MapboxOverlay } from '@deck.gl/mapbox';
import { ScatterplotLayer } from '@deck.gl/layers';

import useTimeStore from '@/store/useTimeStore';
import type { UIPayload, NodeEntry } from '@/types/payload';

// Newcastle upon Tyne city centre
const INITIAL_VIEW = {
  longitude: -1.6178,
  latitude: 54.9783,
  zoom: 11,
  pitch: 45,
  bearing: 0,
};

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

export default function MapVisualizer() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);
  const setMaxTimesteps = useTimeStore((s) => s.setMaxTimesteps);
  const currentTimestep = useTimeStore((s) => s.currentTimestep);

  // ─── Initialise MapLibre + deck.gl overlay once ───────────────────────────
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: MAP_STYLE,
      center: [INITIAL_VIEW.longitude, INITIAL_VIEW.latitude],
      zoom: INITIAL_VIEW.zoom,
      pitch: INITIAL_VIEW.pitch,
      bearing: INITIAL_VIEW.bearing,
    });

    map.addControl(new maplibregl.NavigationControl(), 'top-right');

    const overlay = new MapboxOverlay({ layers: [] });
    map.addControl(overlay as unknown as maplibregl.IControl);

    mapRef.current = map;
    overlayRef.current = overlay;

    // ─── Fetch payload and wire up layers ─────────────────────────────────
    map.on('load', async () => {
      try {
        const res = await fetch('/data/ui_payload.json');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload: UIPayload = await res.json();

        // Update global clock upper bound
        setMaxTimesteps(payload.metadata.timesteps - 1);

        // Log simulated nodes for verification
        const simNodes = payload.nodes.filter((n) => n.is_simulated);
        console.info(
          `[MapVisualizer] Payload loaded: ${payload.nodes.length} nodes ` +
            `(${payload.metadata.num_real_nodes} real + ${simNodes.length} simulated), ` +
            `${payload.edges.length} edges, ${payload.metadata.timesteps} timesteps.`
        );

        // Static proof-of-concept layer: red dot at every node location
        const proofLayer = new ScatterplotLayer<NodeEntry>({
          id: 'nodes-proof',
          data: payload.nodes,
          getPosition: (d) => [d.lon, d.lat],
          getRadius: (d) => (d.is_simulated ? 120 : 80),
          getFillColor: (d) =>
            d.is_simulated ? [255, 200, 0, 220] : [220, 40, 40, 200],
          radiusUnits: 'meters',
          pickable: true,
          onHover: ({ object }) => {
            if (object) {
              console.log(
                `[Hover] ${object.location_name} ` +
                  `| simulated=${object.is_simulated} ` +
                  `| t=${currentTimestep} ` +
                  `| prediction=${object.timeseries[currentTimestep]?.prediction ?? 'N/A'}`
              );
            }
          },
        });

        overlay.setProps({ layers: [proofLayer] });
      } catch (err) {
        console.warn('[MapVisualizer] Failed to load ui_payload.json', err);
      }
    });

    return () => {
      overlay.finalize();
      map.remove();
      mapRef.current = null;
      overlayRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      ref={mapContainerRef}
      style={{ width: '100%', height: '100%' }}
      aria-label="Geospatial digital twin map"
    />
  );
}
