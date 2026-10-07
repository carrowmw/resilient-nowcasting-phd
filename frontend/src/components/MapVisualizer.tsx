"use client";

import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { ScatterplotLayer, ArcLayer } from "@deck.gl/layers";

import useTimeStore, { type DayPayload } from "@/store/useTimeStore";
import type { AttentionEdge, EdgeAttention, NodeSeries } from "@/types/payload";
import type { SpatialNode } from "@/types/registry";
import usePayloadLoader from "@/lib/usePayloadLoader";
import { intervalColor, isGhost, relativeIntervalWidth, topEdgesByMeanWeight, type RGBA } from "@/lib/v5";

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

// Violet accent marks zero-shot holdout sensors distinctly from observed ones.
const HOLDOUT_ACCENT: RGBA = [167, 139, 250, 255];
const MISSING_SERIES_COLOR: RGBA = [128, 128, 128, 180];
// Ghost nodes (no observation in the analysis window): hollow rings, so they never
// read as the solid grey "no interval" marker. Role colour is kept, faded.
const GHOST_OBSERVED_RING: RGBA = [148, 163, 184, 150];
const GHOST_HOLDOUT_RING: RGBA = [167, 139, 250, 130];
const GHOST_RADIUS_M = 8;

/** Node joined with its day series; built once per loaded page. */
interface MapNode extends SpatialNode {
  series: NodeSeries | undefined;
}

function nodeFillColor(d: MapNode, timestep: number): RGBA {
  if (!d.series) return MISSING_SERIES_COLOR;
  // Neutral (not a fabricated heat value) when the model has no lo/hi interval.
  return intervalColor(relativeIntervalWidth(d.series, timestep));
}

function nodeRadius(d: MapNode, timestep: number): number {
  const base = d.node_role === "observed" ? 2 : 4;
  const p = d.series?.point[timestep];
  return p == null ? base : base + Math.max(0, p);
}

interface FilterState {
  selectedNodeId: number | null;
  attentionThreshold: number;
  hoveredNodeId: number | null;
  egoGraphEdges: AttentionEdge[] | null;
}

interface MapData {
  nodes: MapNode[];
  nodeMap: Map<number, MapNode>;
  stableEdges: EdgeAttention[];
}

/**
 * Build all Deck.gl layers for the current frame.
 *
 * Layers (bottom to top):
 *   0. nodes-ghost        — hollow fixed-size rings: no data in the analysis window
 *   1. nodes-observed     — filled by relative interval width
 *   2. nodes-holdout      — same fill + violet ring (zero-shot sensors)
 *   3. nodes-holdout-halo — outer violet target ring
 *   4. topology-layer     — global top-N attention arcs (page.edge_attention)
 *   5. ego-graph-layer    — selected node's ego graph from node_{id}.json
 */
function buildLayers(
  data: MapData,
  timestep: number,
  filters: FilterState,
  onSelectNode: (id: number | null) => void,
  onHoverNode: (id: number | null) => void,
) {
  const { selectedNodeId, attentionThreshold, hoveredNodeId, egoGraphEdges } = filters;
  const hoverScale = (d: MapNode) => (d.id === hoveredNodeId ? 1.6 : 1.0);

  const interaction = {
    pickable: true,
    autoHighlight: true,
    highlightColor: [255, 255, 255, 55] as RGBA,
    onClick: ({ object }: { object?: MapNode }) => {
      if (!object) return false;
      onSelectNode(object.id);
      return true;
    },
    onHover: ({ object }: { object?: MapNode }) => {
      onHoverNode(object ? object.id : null);
      return true;
    },
  };

  // The forecast at a ghost cannot be checked, so its radius does not encode it.
  const ghostLayer = new ScatterplotLayer<MapNode>({
    id: "nodes-ghost",
    data: data.nodes.filter(isGhost),
    getPosition: (d) => [d.lon, d.lat],
    getFillColor: [0, 0, 0, 0],
    getLineColor: (d) => (d.node_role === "holdout" ? GHOST_HOLDOUT_RING : GHOST_OBSERVED_RING),
    getRadius: (d) => GHOST_RADIUS_M * hoverScale(d),
    radiusUnits: "meters",
    radiusMinPixels: 3,
    stroked: true,
    filled: true,   // transparent fill keeps the whole disc pickable
    lineWidthMinPixels: 1.5,
    ...interaction,
    updateTriggers: {
      getRadius: [hoveredNodeId],
    },
  });

  const liveNodes = data.nodes.filter((n) => !isGhost(n));
  const observedLayer = new ScatterplotLayer<MapNode>({
    id: "nodes-observed",
    data: liveNodes.filter((n) => n.node_role === "observed"),
    getPosition: (d) => [d.lon, d.lat],
    getFillColor: (d) => nodeFillColor(d, timestep),
    getRadius: (d) => nodeRadius(d, timestep) * hoverScale(d),
    radiusUnits: "meters",
    stroked: false,
    filled: true,
    ...interaction,
    updateTriggers: {
      getFillColor: [timestep],
      getRadius: [timestep, hoveredNodeId],
    },
  });

  const holdoutNodes = liveNodes.filter((n) => n.node_role === "holdout");
  const holdoutLayer = new ScatterplotLayer<MapNode>({
    id: "nodes-holdout",
    data: holdoutNodes,
    getPosition: (d) => [d.lon, d.lat],
    getFillColor: (d) => nodeFillColor(d, timestep),
    getLineColor: HOLDOUT_ACCENT,
    getRadius: (d) => nodeRadius(d, timestep) * hoverScale(d),
    radiusUnits: "meters",
    stroked: true,
    filled: true,
    lineWidthMinPixels: 2,
    ...interaction,
    updateTriggers: {
      getFillColor: [timestep],
      getRadius: [timestep, hoveredNodeId],
    },
  });

  const holdoutHaloLayer = new ScatterplotLayer<MapNode>({
    id: "nodes-holdout-halo",
    data: holdoutNodes,
    getPosition: (d) => [d.lon, d.lat],
    getFillColor: [0, 0, 0, 0],
    getLineColor: [167, 139, 250, 180],
    getRadius: (d) => nodeRadius(d, timestep) * 1.35 * hoverScale(d),
    radiusUnits: "meters",
    stroked: true,
    filled: false,
    lineWidthMinPixels: 1.5,
    pickable: false,
    updateTriggers: {
      getRadius: [timestep, hoveredNodeId],
    },
  });

  // Global topology: data array is stable per page; threshold + per-tick weights
  // go through accessors. A null weight (edge dropped) renders fully transparent.
  const position = (id: number): [number, number] => {
    const n = data.nodeMap.get(id);
    return n ? [n.lon, n.lat] : [0, 0];
  };
  const arcAlpha = (w: number | null) =>
    w != null && w >= attentionThreshold ? Math.round(10 + w * 245) : 0;
  const arcLayer = new ArcLayer<EdgeAttention>({
    id: "topology-layer",
    data: data.stableEdges,
    visible: selectedNodeId === null,
    getSourcePosition: (d) => position(d.source),
    getTargetPosition: (d) => position(d.target),
    getSourceColor: (d) => [255, 165, 0, arcAlpha(d.weights[timestep] ?? null)],
    getTargetColor: (d) => [255, 69, 0, arcAlpha(d.weights[timestep] ?? null)],
    getWidth: (d) => {
      const w = d.weights[timestep];
      return w == null ? 0 : Math.max(1, w * 80);
    },
    updateTriggers: {
      getSourceColor: [timestep, attentionThreshold],
      getTargetColor: [timestep, attentionThreshold],
      getWidth: [timestep],
    },
  });

  // Ego graph: blue = incoming, rose = outgoing; null weight = absent.
  const egoColor = (d: AttentionEdge, base: number, span: number): RGBA => {
    const w = d.weight_sequence[timestep];
    const alpha = w == null ? 0 : Math.round(base + w * span);
    return d.direction === "incoming" ? [96, 165, 250, alpha] : [251, 113, 133, alpha];
  };
  const egoArcLayer = new ArcLayer<AttentionEdge>({
    id: "ego-graph-layer",
    data: egoGraphEdges ?? [],
    visible: selectedNodeId !== null && (egoGraphEdges?.length ?? 0) > 0,
    getSourcePosition: (d) => [d.source_lon, d.source_lat],
    getTargetPosition: (d) => [d.target_lon, d.target_lat],
    getSourceColor: (d) => egoColor(d, 80, 175),
    getTargetColor: (d) => egoColor(d, 50, 110),
    getWidth: (d) => (d.weight_sequence[timestep] ?? 0) * 200,
    widthUnits: "pixels",
    updateTriggers: {
      getWidth: [timestep],
      getSourceColor: [timestep],
      getTargetColor: [timestep],
      data: [egoGraphEdges],
    },
  });

  return [ghostLayer, observedLayer, holdoutLayer, holdoutHaloLayer, arcLayer, egoArcLayer];
}

function buildMapData(payload: DayPayload): MapData {
  const nodes: MapNode[] = payload.spatial.nodes.map((n) => ({
    ...n,
    series: payload.page.node_series[String(n.id)],
  }));
  return {
    nodes,
    nodeMap: new Map(nodes.map((n) => [n.id, n])),
    stableEdges: topEdgesByMeanWeight(payload.page.edge_attention, GLOBAL_EDGE_CAP),
  };
}

export default function MapVisualizer() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);
  const [mapReady, setMapReady] = useState(false);

  // Registry → spatial → page; data lands in the store as dayPayload.
  const fetchError = usePayloadLoader();
  const dayPayload = useTimeStore((s) => s.dayPayload);

  const mapDataRef = useRef<MapData | null>(null);

  // currentTimestep is tracked via a ref + subscribe to avoid React re-renders
  // on every playback tick.
  const currentTimestepRef = useRef(useTimeStore.getState().currentTimestep);

  // Hover state lives in a ref: only the WebGL overlay needs to rebuild.
  const hoveredNodeIdRef = useRef<number | null>(null);

  // Rebuild functions behind refs so the subscribe/hover callbacks always call
  // the current closure (avoids the "dead callback" bug).
  const rebuildHoverRef = useRef<(id: number | null) => void>(() => {});
  const rebuildRef = useRef<() => void>(() => {});

  const setSelectedNodeId = useTimeStore((s) => s.setSelectedNodeId);
  const selectedNodeId = useTimeStore((s) => s.selectedNodeId);
  const globalAttentionThreshold = useTimeStore((s) => s.globalAttentionThreshold);
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

    // Only deselect on a bare map click: the map click fires AFTER the layer
    // click, so without this guard it would null out a fresh selection.
    map.on("click", () => {
      if (hoveredNodeIdRef.current !== null) return;
      if (useTimeStore.getState().selectedNodeId !== null) {
        useTimeStore.getState().setSelectedNodeId(null);
      }
    });

    map.on("load", () => setMapReady(true));

    return () => {
      overlay.finalize();
      map.remove();
      mapRef.current = null;
      overlayRef.current = null;
    };
  }, []);

  // ── State-driven rebuild (selection, threshold, ego graph, new page) ──────
  useEffect(() => {
    const overlay = overlayRef.current;
    if (!dayPayload || !overlay || !mapReady) return;

    const data = buildMapData(dayPayload);
    mapDataRef.current = data;

    const render = (hoveredNodeId: number | null) => {
      const ov = overlayRef.current;
      const d = mapDataRef.current;
      if (!ov || !d) return;
      const s = useTimeStore.getState();
      ov.setProps({
        layers: buildLayers(
          d,
          currentTimestepRef.current,
          {
            selectedNodeId: s.selectedNodeId,
            attentionThreshold: s.globalAttentionThreshold,
            hoveredNodeId,
            egoGraphEdges: s.egoGraphEdges,
          },
          setSelectedNodeId,
          (id) => rebuildHoverRef.current(id),
        ),
      });
    };

    rebuildHoverRef.current = (id: number | null) => {
      if (hoveredNodeIdRef.current === id) return;
      hoveredNodeIdRef.current = id;
      render(id);
    };
    rebuildRef.current = () => render(hoveredNodeIdRef.current);
    rebuildRef.current();
  }, [selectedNodeId, globalAttentionThreshold, egoGraphEdges, dayPayload, mapReady, setSelectedNodeId]);

  // ── Timestep subscribe — bypasses the React render cycle ─────────────────
  useEffect(() => {
    currentTimestepRef.current = useTimeStore.getState().currentTimestep;
    let prevTimestep = currentTimestepRef.current;
    const unsub = useTimeStore.subscribe((state) => {
      if (state.currentTimestep === prevTimestep) return;
      prevTimestep = state.currentTimestep;
      currentTimestepRef.current = state.currentTimestep;
      rebuildRef.current();
    });
    return unsub;
  }, []);

  return (
    <>
      {fetchError && (
        <div
          style={{
            position: "absolute",
            top: 8,
            left: 8,
            zIndex: 20,
            background: "rgba(239,68,68,0.9)",
            color: "#fff",
            padding: "8px 14px",
            borderRadius: 8,
            fontSize: 13,
            maxWidth: 360,
            backdropFilter: "blur(4px)",
          }}
          role="alert"
        >
          {fetchError}
        </div>
      )}
      <div
        ref={mapContainerRef}
        style={{ width: "100%", height: "100%" }}
        aria-label="Geospatial digital twin map"
      />
    </>
  );
}
