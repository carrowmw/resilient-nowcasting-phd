"use client";

import { useEffect, useRef } from "react";
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

// Defensive cap on the global edge set when no node is selected. With ego-graph
// isolation active, ALL edges touching the selected node are shown regardless.
const GLOBAL_EDGE_CAP = 80;

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

function nodeFillColor(d: NodeEntry, timestep: number): [number, number, number, number] {
  const entry = d.timeseries[timestep];
  if (!entry) return [128, 128, 128, 180];
  const maxUncertainty = 50;
  const normalized = Math.min(1, entry.uncertainty / maxUncertainty);
  return heatColor(normalized);
}

function nodeRadius(d: NodeEntry, timestep: number): number {
  const entry = d.timeseries[timestep];
  const base = d.is_simulated ? 4 : 2;
  if (!entry) return base;
  return base + entry.prediction;
}

interface FilterState {
  selectedNodeId: number | null;
  attentionThreshold: number;
}

/**
 * Construct ScatterplotLayers (real + simulated) and an ArcLayer for the given
 * payload snapshot. Ego-graph isolation: when selectedNodeId is set, only edges
 * incident to that node are kept.
 */
function buildLayers(
  payload: UIPayload,
  nodeMap: Map<number, NodeEntry>,
  timestep: number,
  filters: FilterState,
  onSelectNode: (id: number | null) => void,
) {
  const { selectedNodeId, attentionThreshold } = filters;

  // ── Real-node layer (solid filled disc) ────────────────────────────────────
  const realNodes = payload.nodes.filter((n) => !n.is_simulated);
  const realLayer = new ScatterplotLayer<NodeEntry>({
    id: "nodes-real",
    data: realNodes,
    getPosition: (d) => [d.lon, d.lat],
    getFillColor: (d) => nodeFillColor(d, timestep),
    getRadius: (d) => nodeRadius(d, timestep),
    radiusUnits: "meters",
    stroked: false,
    filled: true,
    pickable: true,
    onClick: ({ object }) => onSelectNode(object ? object.id : null),
    updateTriggers: {
      getFillColor: [timestep],
      getRadius: [timestep],
    },
  });

  // ── Simulated-node layer (hollow ring) ─────────────────────────────────────
  // NSTAR_FRONTEND_ARCHITECTURE.md §A: simulated nodes must be visually distinct.
  // Deck.gl's stroked/filled flags are layer-wide, so we split into two layers.
  const simNodes = payload.nodes.filter((n) => n.is_simulated);
  const simLayer = new ScatterplotLayer<NodeEntry>({
    id: "nodes-simulated",
    data: simNodes,
    getPosition: (d) => [d.lon, d.lat],
    // Ring inherits the same uncertainty heat colour for visual coherence.
    getLineColor: (d) => nodeFillColor(d, timestep),
    getRadius: (d) => nodeRadius(d, timestep),
    radiusUnits: "meters",
    stroked: true,
    filled: false,
    lineWidthMinPixels: 3,
    pickable: true,
    onClick: ({ object }) => onSelectNode(object ? object.id : null),
    updateTriggers: {
      getLineColor: [timestep],
      getRadius: [timestep],
    },
  });

  // ── Topology Layer (attention arcs) ────────────────────────────────────────
  type WeightedEdge = { edge: EdgeEntry; weight: number };
  let weighted: WeightedEdge[] = payload.edges
    .map((e) => ({ edge: e, weight: e.attention_sequence[timestep] ?? 0 }))
    .filter((x) => x.weight >= attentionThreshold);

  if (selectedNodeId !== null) {
    // Ego-graph: keep only edges touching the selected node.
    weighted = weighted.filter(
      (x) =>
        x.edge.source === selectedNodeId || x.edge.target === selectedNodeId,
    );
  } else {
    // Defensive cap to prevent visual hairball when nothing is selected.
    weighted = weighted
      .sort((a, b) => b.weight - a.weight)
      .slice(0, GLOBAL_EDGE_CAP);
  }

  const arcLayer = new ArcLayer<WeightedEdge>({
    id: "topology-layer",
    data: weighted,
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
    updateTriggers: {
      getSourceColor: [timestep],
      getTargetColor: [timestep],
      getWidth: [timestep],
      data: [timestep, selectedNodeId, attentionThreshold],
    },
  });

  return [realLayer, simLayer, arcLayer];
}

export default function MapVisualizer() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);

  const payloadRef = useRef<UIPayload | null>(null);
  const nodeMapRef = useRef<Map<number, NodeEntry>>(new Map());

  const setMaxTimesteps = useTimeStore((s) => s.setMaxTimesteps);
  const setSelectedNodeId = useTimeStore((s) => s.setSelectedNodeId);
  const currentTimestep = useTimeStore((s) => s.currentTimestep);
  const selectedNodeId = useTimeStore((s) => s.selectedNodeId);
  const attentionThreshold = useTimeStore((s) => s.attentionThreshold);

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

    // Click on empty map area → deselect node (Deck.gl onClick fires first when
    // a pickable object is hit, so this only runs on background clicks).
    map.on("click", () => {
      if (useTimeStore.getState().selectedNodeId !== null) {
        useTimeStore.getState().setSelectedNodeId(null);
      }
    });

    map.on("load", async () => {
      try {
        const res = await fetch("/data/ui_payload.json");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload: UIPayload = await res.json();

        setMaxTimesteps(payload.metadata.timesteps - 1);

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

        const state = useTimeStore.getState();
        overlay.setProps({
          layers: buildLayers(
            payload,
            nodeMap,
            state.currentTimestep,
            {
              selectedNodeId: state.selectedNodeId,
              attentionThreshold: state.attentionThreshold,
            },
            setSelectedNodeId,
          ),
        });
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

  // Rebuild layers whenever the timestep, selection, or threshold changes.
  useEffect(() => {
    const payload = payloadRef.current;
    const overlay = overlayRef.current;
    if (!payload || !overlay) return;
    overlay.setProps({
      layers: buildLayers(
        payload,
        nodeMapRef.current,
        currentTimestep,
        { selectedNodeId, attentionThreshold },
        setSelectedNodeId,
      ),
    });
  }, [currentTimestep, selectedNodeId, attentionThreshold, setSelectedNodeId]);

  return (
    <div
      ref={mapContainerRef}
      style={{ width: "100%", height: "100%" }}
      aria-label="Geospatial digital twin map"
    />
  );
}
