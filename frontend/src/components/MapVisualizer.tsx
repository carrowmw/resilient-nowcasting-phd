"use client";

import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { ScatterplotLayer, ArcLayer } from "@deck.gl/layers";

import useTimeStore from "@/store/useTimeStore";
import type { UIPayload, NodeEntry, EdgeEntry, AttentionEdge } from "@/types/payload";

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

// Defensive cap on the global edge set when no node is selected.
const GLOBAL_EDGE_CAP = 80;

/**
 * Maps a normalised value t∈[0,1] to a Green→Yellow→Red RGBA tuple.
 */
function heatColor(t: number): [number, number, number, number] {
  const c = Math.min(1, Math.max(0, t));
  if (c < 0.5) {
    const s = c * 2;
    return [Math.round(s * 255), 200, 0, 220];
  }
  const s = (c - 0.5) * 2;
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
  hoveredNodeId: number | null;
  egoGraphEdges: AttentionEdge[] | null;
}

/**
 * Build all Deck.gl layers for the current frame.
 *
 * Layers (bottom to top):
 *   1. nodes-real       — solid ScatterplotLayer for physical sensors
 *   2. nodes-simulated  — hollow ring ScatterplotLayer for synthetic nodes
 *   3. topology-layer   — global or ego-graph ArcLayer (master_index edges)
 *   4. ego-graph-layer  — localized top-K ArcLayer from node_{id}.json
 */
function buildLayers(
  payload: UIPayload,
  nodeMap: Map<number, NodeEntry>,
  timestep: number,
  filters: FilterState,
  onSelectNode: (id: number | null) => void,
  onHoverNode: (id: number | null) => void,
) {
  const { selectedNodeId, attentionThreshold, hoveredNodeId, egoGraphEdges } = filters;

  // ── Real-node layer ────────────────────────────────────────────────────────
  const realNodes = payload.nodes.filter((n) => !n.is_simulated);
  const realLayer = new ScatterplotLayer<NodeEntry>({
    id: "nodes-real",
    data: realNodes,
    getPosition: (d) => [d.lon, d.lat],
    getFillColor: (d) => nodeFillColor(d, timestep),
    getRadius: (d) => nodeRadius(d, timestep) * (d.id === hoveredNodeId ? 1.6 : 1.0),
    radiusUnits: "meters",
    stroked: false,
    filled: true,
    pickable: true,
    autoHighlight: true,
    highlightColor: [255, 255, 255, 55],
    onClick: ({ object }) => onSelectNode(object ? object.id : null),
    onHover: ({ object }) => { onHoverNode(object ? object.id : null); return true; },
    updateTriggers: {
      getFillColor: [timestep],
      getRadius: [timestep, hoveredNodeId],
    },
  });

  // ── Simulated-node layer (hollow ring) ─────────────────────────────────────
  const simNodes = payload.nodes.filter((n) => n.is_simulated);
  const simLayer = new ScatterplotLayer<NodeEntry>({
    id: "nodes-simulated",
    data: simNodes,
    getPosition: (d) => [d.lon, d.lat],
    getLineColor: (d) => nodeFillColor(d, timestep),
    getRadius: (d) => nodeRadius(d, timestep) * (d.id === hoveredNodeId ? 1.6 : 1.0),
    radiusUnits: "meters",
    stroked: true,
    filled: false,
    lineWidthMinPixels: 3,
    pickable: true,
    autoHighlight: true,
    highlightColor: [255, 255, 255, 55],
    onClick: ({ object }) => onSelectNode(object ? object.id : null),
    onHover: ({ object }) => { onHoverNode(object ? object.id : null); return true; },
    updateTriggers: {
      getLineColor: [timestep],
      getRadius: [timestep, hoveredNodeId],
    },
  });

  // ── Global topology layer (master_index edges) ─────────────────────────────
  type WeightedEdge = { edge: EdgeEntry; weight: number };
  let weighted: WeightedEdge[] = payload.edges
    .map((e) => ({ edge: e, weight: e.attention_sequence[timestep] ?? 0 }))
    .filter((x) => x.weight >= attentionThreshold);

  if (selectedNodeId !== null) {
    weighted = weighted.filter(
      (x) =>
        x.edge.source === selectedNodeId || x.edge.target === selectedNodeId,
    );
  } else {
    weighted = weighted.sort((a, b) => b.weight - a.weight).slice(0, GLOBAL_EDGE_CAP);
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

  // ── Localized ego-graph layer (top-K edges from node_{id}.json) ────────────
  // Blue arcs = incoming attention; rose arcs = outgoing attention.
  // Width and alpha scale with weight_sequence[timestep]. Visible only when a node is selected
  // and ego-graph data has been fetched by NodeInspector.
  const egoArcLayer = new ArcLayer<AttentionEdge>({
    id: "ego-graph-layer",
    data: egoGraphEdges ?? [],
    visible: selectedNodeId !== null && (egoGraphEdges?.length ?? 0) > 0,
    getSourcePosition: (d) => [d.source_lon, d.source_lat],
    getTargetPosition: (d) => [d.target_lon, d.target_lat],
    getSourceColor: (d) => {
      const w = d.weight_sequence[timestep] ?? 0;
      const alpha = Math.round(80 + w * 175);
      return d.direction === "incoming" ? [96, 165, 250, alpha] : [251, 113, 133, alpha];
    },
    getTargetColor: (d) => {
      const w = d.weight_sequence[timestep] ?? 0;
      const alpha = Math.round(50 + w * 110);
      return d.direction === "incoming" ? [96, 165, 250, alpha] : [251, 113, 133, alpha];
    },
    getWidth: (d) => Math.max(2, (d.weight_sequence[timestep] ?? 0) * 14),
    widthUnits: "pixels",
    updateTriggers: {
      data: [selectedNodeId],
      getSourceColor: [timestep],
      getTargetColor: [timestep],
      getWidth: [timestep],
    },
  });

  return [realLayer, simLayer, arcLayer, egoArcLayer];
}

export default function MapVisualizer() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);

  const payloadRef = useRef<UIPayload | null>(null);
  const nodeMapRef = useRef<Map<number, NodeEntry>>(new Map());

  // Hover state is kept in a ref (not Zustand) to avoid React re-renders on
  // every pointer movement — only the WebGL overlay needs to rebuild.
  const hoveredNodeIdRef = useRef<number | null>(null);

  const setMaxTimesteps = useTimeStore((s) => s.setMaxTimesteps);
  const setSelectedNodeId = useTimeStore((s) => s.setSelectedNodeId);
  const currentTimestep = useTimeStore((s) => s.currentTimestep);
  const selectedNodeId = useTimeStore((s) => s.selectedNodeId);
  const attentionThreshold = useTimeStore((s) => s.attentionThreshold);
  const egoGraphEdges = useTimeStore((s) => s.egoGraphEdges);

  // ── Initialise MapLibre + deck.gl overlay once ────────────────────────────
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

    map.on("click", () => {
      if (useTimeStore.getState().selectedNodeId !== null) {
        useTimeStore.getState().setSelectedNodeId(null);
      }
    });

    map.on("load", async () => {
      try {
        const res = await fetch("/data/master_index.json");
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
              hoveredNodeId: hoveredNodeIdRef.current,
              egoGraphEdges: state.egoGraphEdges,
            },
            setSelectedNodeId,
            (id) => {
              if (hoveredNodeIdRef.current !== id) {
                hoveredNodeIdRef.current = id;
                const s = useTimeStore.getState();
                overlay.setProps({
                  layers: buildLayers(
                    payload,
                    nodeMap,
                    s.currentTimestep,
                    {
                      selectedNodeId: s.selectedNodeId,
                      attentionThreshold: s.attentionThreshold,
                      hoveredNodeId: id,
                      egoGraphEdges: s.egoGraphEdges,
                    },
                    setSelectedNodeId,
                    () => {},
                  ),
                });
              }
            },
          ),
        });
      } catch (err) {
        console.warn("[MapVisualizer] Failed to load master_index.json", err);
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

  // Rebuild layers whenever timestep, selection, threshold, or ego-graph data changes.
  useEffect(() => {
    const payload = payloadRef.current;
    const overlay = overlayRef.current;
    if (!payload || !overlay) return;
    overlay.setProps({
      layers: buildLayers(
        payload,
        nodeMapRef.current,
        currentTimestep,
        {
          selectedNodeId,
          attentionThreshold,
          hoveredNodeId: hoveredNodeIdRef.current,
          egoGraphEdges,
        },
        setSelectedNodeId,
        (id) => {
          if (hoveredNodeIdRef.current !== id) {
            hoveredNodeIdRef.current = id;
            const p = payloadRef.current;
            const ov = overlayRef.current;
            if (!p || !ov) return;
            const s = useTimeStore.getState();
            ov.setProps({
              layers: buildLayers(
                p,
                nodeMapRef.current,
                s.currentTimestep,
                {
                  selectedNodeId: s.selectedNodeId,
                  attentionThreshold: s.attentionThreshold,
                  hoveredNodeId: id,
                  egoGraphEdges: s.egoGraphEdges,
                },
                setSelectedNodeId,
                () => {},
              ),
            });
          }
        },
      ),
    });
  }, [currentTimestep, selectedNodeId, attentionThreshold, egoGraphEdges, setSelectedNodeId]);

  return (
    <div
      ref={mapContainerRef}
      style={{ width: "100%", height: "100%" }}
      aria-label="Geospatial digital twin map"
    />
  );
}
