"use client";

import { useEffect, useRef, useCallback } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { ScatterplotLayer, ArcLayer } from "@deck.gl/layers";

import useTimeStore from "@/store/useTimeStore";
import type { UIPayload, NodeEntry, EdgeEntry } from "@/types/payload";

// Newcastle upon Tyne city centre
const INITIAL_VIEW = {
  longitude: -1.6178,
  latitude: 54.9783,
  zoom: 11,
  pitch: 45,
  bearing: 0,
};

const MAP_STYLE =
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

// Top-N attention edges to render per timestep (avoids "hairball" clutter)
const TOP_N_EDGES = 60;

/**
 * Maps a normalised value t∈[0,1] to a Green→Yellow→Red RGBA tuple.
 * t=0 → dim green, t=0.5 → yellow, t=1 → red.
 */
function heatColor(t: number): [number, number, number, number] {
  const c = Math.min(1, Math.max(0, t));
  if (c < 0.5) {
    const s = c * 2; // 0→1
    return [Math.round(s * 255), 200, 0, 220];
  }
  const s = (c - 0.5) * 2; // 0→1
  return [255, Math.round(200 * (1 - s)), 0, 220];
}

/**
 * Construct a ScatterplotLayer + ArcLayer for the given payload snapshot at
 * `timestep`.  Called both on initial load and on every timestep change.
 */
function buildLayers(
  payload: UIPayload,
  nodeMap: Map<number, NodeEntry>,
  timestep: number,
) {
  // Compute per-timestep max prediction for colour normalisation.
  let maxPrediction = 1;
  for (const node of payload.nodes) {
    const entry = node.timeseries[timestep];
    if (entry && entry.prediction > maxPrediction)
      maxPrediction = entry.prediction;
  }

  // ── Node Layer ─────────────────────────────────────────────────────────────
  const nodeLayer = new ScatterplotLayer<NodeEntry>({
    id: "nodes-layer",
    data: payload.nodes,
    getPosition: (d) => [d.lon, d.lat],

    getFillColor: (d) => {
      const entry = d.timeseries[timestep];
      if (!entry) return [128, 128, 128, 180]; // Gray for missing data
      // Normalizing uncertainty for the color scale.
      // You may need to define a maxUncertainty based on your data distribution.
      const maxUncertainty = 50;
      const normalizedUncertainty = Math.min(
        1,
        entry.uncertainty / maxUncertainty,
      );

      // heatColor(0) is Green (Low Uncertainty), heatColor(1) is Red (High Uncertainty)
      return heatColor(normalizedUncertainty);
    },

    // Radius represents prediction (count)
    getRadius: (d) => {
      const entry = d.timeseries[timestep];
      const base = d.is_simulated ? 4 : 2;
      if (!entry) return base;

      return base + entry.prediction;
    },

    // ★ Deck.gl updateTriggers — forces accessor re-evaluation on timestep change.
    updateTriggers: {
      getFillColor: [timestep],
      getRadius: [timestep],
    },

    radiusUnits: "meters",
    pickable: true,
    onHover: ({ object }) => {
      if (object) {
        const entry = object.timeseries[timestep];
        console.log(
          `[Hover] ${object.location_name}` +
            ` | simulated=${object.is_simulated}` +
            ` | t=${timestep}` +
            ` | prediction=${entry?.prediction ?? "N/A"}` +
            ` | uncertainty=${entry?.uncertainty ?? "N/A"}`,
        );
      }
    },
  });

  // ── Topology Layer (attention arcs) ────────────────────────────────────────
  // Select top-N edges by attention weight at this timestep to avoid clutter.
  type WeightedEdge = { edge: EdgeEntry; weight: number };
  const topEdges: WeightedEdge[] = payload.edges
    .map((e) => ({ edge: e, weight: e.attention_sequence[timestep] ?? 0 }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, TOP_N_EDGES);

  const arcLayer = new ArcLayer<WeightedEdge>({
    id: "topology-layer",
    data: topEdges,

    getSourcePosition: (d) => {
      const src = nodeMap.get(d.edge.source);
      return src ? [src.lon, src.lat] : [0, 0];
    },
    getTargetPosition: (d) => {
      const tgt = nodeMap.get(d.edge.target);
      return tgt ? [tgt.lon, tgt.lat] : [0, 0];
    },

    getSourceColor: (d) => [255, 165, 0, Math.round(d.weight * 255)],
    getTargetColor: (d) => [255, 69, 0, Math.round(d.weight * 255)],
    getWidth: (d) => Math.max(1, d.weight * 80),

    // ★ Deck.gl updateTriggers — forces accessor re-evaluation on timestep change.
    updateTriggers: {
      getSourceColor: [timestep],
      getTargetColor: [timestep],
      getWidth: [timestep],
      // data itself changes (top-N re-sorted), but adding this is defensive.
      data: [timestep],
    },
  });

  return [nodeLayer, arcLayer];
}

export default function MapVisualizer() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);

  // Payload and derived node-lookup map kept in refs so the timestep effect
  // can read them without being listed as dependencies (they never change after
  // the initial fetch).
  const payloadRef = useRef<UIPayload | null>(null);
  const nodeMapRef = useRef<Map<number, NodeEntry>>(new Map());

  const setMaxTimesteps = useTimeStore((s) => s.setMaxTimesteps);
  const currentTimestep = useTimeStore((s) => s.currentTimestep);

  // ─── Initialise MapLibre + deck.gl overlay once ────────────────────────────
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

    map.addControl(new maplibregl.NavigationControl(), "top-right");

    const overlay = new MapboxOverlay({ layers: [] });
    map.addControl(overlay as unknown as maplibregl.IControl);

    mapRef.current = map;
    overlayRef.current = overlay;

    // Fetch payload, wire up initial layers, set clock upper-bound.
    map.on("load", async () => {
      try {
        const res = await fetch("/data/ui_payload.json");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload: UIPayload = await res.json();

        setMaxTimesteps(payload.metadata.timesteps - 1);

        // Build O(1) node-lookup map for ArcLayer position accessors.
        const nodeMap = new Map<number, NodeEntry>();
        for (const node of payload.nodes) nodeMap.set(node.id, node);

        payloadRef.current = payload;
        nodeMapRef.current = nodeMap;

        const simNodes = payload.nodes.filter((n) => n.is_simulated);
        console.info(
          `[MapVisualizer] Payload loaded: ${payload.nodes.length} nodes` +
            ` (${payload.metadata.num_real_nodes} real + ${simNodes.length} simulated),` +
            ` ${payload.edges.length} edges, ${payload.metadata.timesteps} timesteps.`,
        );

        // Render initial layers at whatever timestep the store currently holds.
        const t0 = useTimeStore.getState().currentTimestep;
        overlay.setProps({ layers: buildLayers(payload, nodeMap, t0) });
      } catch (err) {
        console.warn("[MapVisualizer] Failed to load ui_payload.json", err);
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

  // ─── Rebuild layers on every timestep change ──────────────────────────────
  // Deck.gl uses shallow reference equality to detect changes; we must pass
  // new layer instances (with correct updateTriggers) on each tick.
  useEffect(() => {
    const payload = payloadRef.current;
    const overlay = overlayRef.current;
    if (!payload || !overlay) return;
    overlay.setProps({
      layers: buildLayers(payload, nodeMapRef.current, currentTimestep),
    });
  }, [currentTimestep]);

  return (
    <div
      ref={mapContainerRef}
      style={{ width: "100%", height: "100%" }}
      aria-label="Geospatial digital twin map"
    />
  );
}
