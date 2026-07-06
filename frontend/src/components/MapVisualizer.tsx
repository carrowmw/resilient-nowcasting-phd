"use client";

import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { ScatterplotLayer, ArcLayer } from "@deck.gl/layers";

import useTimeStore from "@/store/useTimeStore";
import type {
  UIPayload,
  NodeEntry,
  NodeRole,
  EdgeEntry,
  AttentionEdge,
} from "@/types/payload";
import type {
  RunRegistry,
  SpatialIndex,
  TemporalPage,
} from "@/types/registry";
import { registryUrl, spatialUrl, pageUrl } from "@/lib/payloadSource";

/**
 * Merge a per-run spatial index with a per-day temporal page into the legacy
 * UIPayload shape, so buildLayers() / nodeMap logic stay unchanged. The split
 * payload (Phase 1) keeps coordinates + topology in spatial.json and the 96-step
 * timeseries + edge attention in page.json; this stitches them back together.
 */
function mergePayload(spatial: SpatialIndex, page: TemporalPage): UIPayload {
  const m = spatial.metadata;
  const nodes: NodeEntry[] = spatial.nodes.map((n) => ({
    id: n.id,
    location_name: n.location_name,
    lat: n.lat,
    lon: n.lon,
    node_role: n.node_role,
    is_simulated: n.is_simulated,
    timeseries: page.node_timeseries[String(n.id)] ?? [],
    semantic_tags: n.semantic_tags,
  }));
  return {
    metadata: {
      num_nodes: m.num_nodes,
      num_real_nodes: m.num_real_nodes,
      num_simulated_nodes: m.num_simulated_nodes,
      num_holdout_nodes: m.num_holdout_nodes,
      holdout_manifest_sha: m.holdout_manifest_sha,
      timesteps: page.timesteps,
      seq_len: page.seq_len,
      veh_class: m.veh_class,
      dir: m.dir,
      start_date: page.date,
      end_date: page.date,
      wandb_run_id: m.wandb_run_id,
      git_commit_hash: m.git_commit_hash,
      hyperparameters: m.hyperparameters,
    },
    nodes,
    edges: page.edge_attention as EdgeEntry[],
    pit_histogram: page.pit_histogram,
    zero_inflation_reliability: page.zero_inflation_reliability,
    global_imputation_burden: page.global_imputation_burden,
    global_healing_quality: page.global_healing_quality,
    attention_entropy_series: page.attention_entropy_series,
    tripwire_audit_trail: page.tripwire_audit_trail,
    alpha_node_leaderboard: page.alpha_node_leaderboard,
  };
}

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

/** Resolve a node's display role, deriving it from is_simulated for pre-harness
 * payloads that predate the node_role field. */
function roleOf(n: { node_role?: NodeRole; is_simulated: boolean }): NodeRole {
  return n.node_role ?? (n.is_simulated ? "simulated" : "observed");
}

// Neutral slate for nodes whose model produces no uncertainty (baselines): we
// have nothing to heat-map, so we must not paint a fabricated colour.
const NO_UNCERTAINTY_COLOR: [number, number, number, number] = [100, 116, 139, 200];
// Violet accent marks zero-shot holdout sensors distinctly from observed ones.
const HOLDOUT_ACCENT: [number, number, number, number] = [167, 139, 250, 255];

function nodeFillColor(
  d: NodeEntry,
  timestep: number,
): [number, number, number, number] {
  const entry = d.timeseries[timestep];
  if (!entry) return [128, 128, 128, 180];
  // A baseline has no σ — colour it neutrally rather than inventing a heat value.
  if (entry.uncertainty == null) return NO_UNCERTAINTY_COLOR;
  // Stabilised Coefficient of Variation: σ/(μ+1)
  // Prevents high-volume nodes from always appearing red due to high absolute
  // variance. A CV > 1.0 (uncertainty exceeds predicted volume) is capped at red.
  const relativeUncertainty = entry.uncertainty / (entry.prediction + 1);
  const normalized = Math.min(1, relativeUncertainty);
  return heatColor(normalized);
}

function nodeRadius(d: NodeEntry, timestep: number): number {
  const entry = d.timeseries[timestep];
  const base = roleOf(d) === "observed" ? 2 : 4;
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
 *   1. nodes-real       — solid ScatterplotLayer for observed physical sensors
 *   2. nodes-holdout    — heat-filled + violet-ringed layer for zero-shot sensors
 *   3. nodes-simulated  — hollow ring ScatterplotLayer for synthetic nodes
 *   4. topology-layer   — global or ego-graph ArcLayer (master_index edges)
 *   5. ego-graph-layer  — localized top-K ArcLayer from node_{id}.json
 */
function buildLayers(
  payload: UIPayload,
  nodeMap: Map<number, NodeEntry>,
  timestep: number,
  filters: FilterState,
  onSelectNode: (id: number | null) => void,
  onHoverNode: (id: number | null) => void,
) {
  const { selectedNodeId, attentionThreshold, hoveredNodeId, egoGraphEdges } =
    filters;

  // ── Observed-node layer ────────────────────────────────────────────────────
  const realNodes = payload.nodes.filter((n) => roleOf(n) === "observed");
  const realLayer = new ScatterplotLayer<NodeEntry>({
    id: "nodes-real",
    data: realNodes,
    getPosition: (d) => [d.lon, d.lat],
    getFillColor: (d) => nodeFillColor(d, timestep),
    getRadius: (d) =>
      nodeRadius(d, timestep) * (d.id === hoveredNodeId ? 1.6 : 1.0),
    radiusUnits: "meters",
    stroked: false,
    filled: true,
    pickable: true,
    autoHighlight: true,
    highlightColor: [255, 255, 255, 55],
    onClick: ({ object }) => {
      if (!object) return false;
      onSelectNode(object.id);
      return true;
    },
    onHover: ({ object }) => {
      onHoverNode(object ? object.id : null);
      return true;
    },
    updateTriggers: {
      getFillColor: [timestep],
      getRadius: [timestep, hoveredNodeId],
    },
  });

  // ── Holdout-node layer (zero-shot sensors) ─────────────────────────────────
  // Real sensors masked throughout training: heat-mapped fill (they DO have a
  // prediction + ground truth) plus a violet ring so the map marks them as the
  // provably-never-trained-on nodes, distinct from ordinary observed sensors.
  const holdoutNodes = payload.nodes.filter((n) => roleOf(n) === "holdout");
  const holdoutLayer = new ScatterplotLayer<NodeEntry>({
    id: "nodes-holdout",
    data: holdoutNodes,
    getPosition: (d) => [d.lon, d.lat],
    getFillColor: (d) => nodeFillColor(d, timestep),
    getLineColor: HOLDOUT_ACCENT,
    getRadius: (d) =>
      nodeRadius(d, timestep) * (d.id === hoveredNodeId ? 1.6 : 1.0),
    radiusUnits: "meters",
    stroked: true,
    filled: true,
    lineWidthMinPixels: 2,
    pickable: true,
    autoHighlight: true,
    highlightColor: [255, 255, 255, 55],
    onClick: ({ object }) => {
      if (!object) return false;
      onSelectNode(object.id);
      return true;
    },
    onHover: ({ object }) => {
      onHoverNode(object ? object.id : null);
      return true;
    },
    updateTriggers: {
      getFillColor: [timestep],
      getRadius: [timestep, hoveredNodeId],
    },
  });

  // ── Simulated-node layer (hollow ring with transparent fill) ──────────────
  // filled: true + getFillColor: [0,0,0,0] creates an invisible solid disk that
  // covers the hollow centre, giving the full circle — not just the stroke — a
  // pickable hit-area. Without this the centre gap of the ring passes all
  // pointer events through to the map, making synthetic nodes unclickable.
  const simNodes = payload.nodes.filter((n) => roleOf(n) === "simulated");
  const simLayer = new ScatterplotLayer<NodeEntry>({
    id: "nodes-simulated",
    data: simNodes,
    getPosition: (d) => [d.lon, d.lat],
    getLineColor: (d) => nodeFillColor(d, timestep),
    getFillColor: [0, 0, 0, 0],
    getRadius: (d) =>
      nodeRadius(d, timestep) * (d.id === hoveredNodeId ? 1.6 : 1.0),
    radiusUnits: "meters",
    stroked: true,
    filled: true,
    lineWidthMinPixels: 3,
    pickable: true,
    autoHighlight: true,
    highlightColor: [255, 255, 255, 55],
    onClick: ({ object }) => {
      if (!object) return false;
      onSelectNode(object.id);
      return true;
    },
    onHover: ({ object }) => {
      onHoverNode(object ? object.id : null);
      return true;
    },
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
    weighted = weighted
      .sort((a, b) => b.weight - a.weight)
      .slice(0, GLOBAL_EDGE_CAP);
  }

  const arcLayer = new ArcLayer<WeightedEdge>({
    id: "topology-layer",
    data: weighted,
    visible: selectedNodeId === null,
    getSourcePosition: (d) => {
      const src = nodeMap.get(d.edge.source);
      return src ? [src.lon, src.lat] : [0, 0];
    },
    getTargetPosition: (d) => {
      const tgt = nodeMap.get(d.edge.target);
      return tgt ? [tgt.lon, tgt.lat] : [0, 0];
    },
    getSourceColor: (d) => [255, 165, 0, Math.round(10 + d.weight * 245)],
    getTargetColor: (d) => [255, 69, 0, Math.round(10 + d.weight * 245)],
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
  // Width and alpha scale with weight_sequence[currentTimestep]. Visible only when a
  // node is selected and ego-graph data has been fetched by NodeInspector.
  const currentTimestep = timestep;
  const egoArcLayer = new ArcLayer<AttentionEdge>({
    id: "ego-graph-layer",
    data: egoGraphEdges ?? [],
    visible: selectedNodeId !== null && (egoGraphEdges?.length ?? 0) > 0,
    getSourcePosition: (d) => [d.source_lon, d.source_lat],
    getTargetPosition: (d) => [d.target_lon, d.target_lat],
    getSourceColor: (d) => {
      const w = d.weight_sequence[currentTimestep] ?? 0;
      const alpha = Math.round(80 + w * 175);
      return d.direction === "incoming"
        ? [96, 165, 250, alpha]
        : [251, 113, 133, alpha];
    },
    getTargetColor: (d) => {
      const w = d.weight_sequence[currentTimestep] ?? 0;
      const alpha = Math.round(50 + w * 110);
      return d.direction === "incoming"
        ? [96, 165, 250, alpha]
        : [251, 113, 133, alpha];
    },
    getWidth: (d) => {
      const w = d.weight_sequence[currentTimestep] ?? 0;
      // Tripwire: must fire continuously while scrubbing TimeSlider. If this
      // logs only once at mount, the layer's updateTriggers are not wired and
      // accessors are closing over a stale currentTimestep.
      console.log(
        `[ego-arc] getWidth t=${currentTimestep} edge=${d.source_id}->${d.target_id} w=${w.toFixed(3)}`,
      );
      return w * 200;
    },
    widthUnits: "pixels",
    updateTriggers: {
      getWidth: [currentTimestep],
      getSourceColor: [currentTimestep],
      getTargetColor: [currentTimestep],
      // Re-evaluate the data binding when timestep advances or the underlying
      // edges array changes (e.g. a different node was selected).
      data: [currentTimestep, egoGraphEdges],
    },
  });

  return [realLayer, holdoutLayer, simLayer, arcLayer, egoArcLayer];
}

export default function MapVisualizer() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);

  const payloadRef = useRef<UIPayload | null>(null);
  const nodeMapRef = useRef<Map<number, NodeEntry>>(new Map());
  const spatialRef = useRef<SpatialIndex | null>(null);

  // Reactive triggers so the layer-rebuild effect fires when the map finishes
  // loading or a new day's page is merged into payloadRef.
  const [mapReady, setMapReady] = useState(false);
  const [payloadVersion, setPayloadVersion] = useState(0);

  // Hover state is kept in a ref (not Zustand) to avoid React re-renders on
  // every pointer movement — only the WebGL overlay needs to rebuild.
  const hoveredNodeIdRef = useRef<number | null>(null);

  // Self-referential hover rebuild function. Set once after data loads, updated
  // on every state-effect run. Always reads live values from useTimeStore.getState()
  // and from stable refs, so the closure never goes stale.
  //
  // The indirection through a ref (rather than a plain closure) is what breaks
  // the "dead callback" bug: every buildLayers call passes
  //   onHoverNode: (id) => rebuildHoverRef.current(id)
  // so the next hover invocation always calls the *current* rebuild function,
  // not a snapshot of it from the previous render.
  const rebuildHoverRef = useRef<(id: number | null) => void>(() => {});

  const setMaxTimesteps = useTimeStore((s) => s.setMaxTimesteps);
  const setTimestep = useTimeStore((s) => s.setTimestep);
  const setSelectedNodeId = useTimeStore((s) => s.setSelectedNodeId);
  const currentTimestep = useTimeStore((s) => s.currentTimestep);
  const selectedNodeId = useTimeStore((s) => s.selectedNodeId);
  const globalAttentionThreshold = useTimeStore(
    (s) => s.globalAttentionThreshold,
  );
  const egoGraphEdges = useTimeStore((s) => s.egoGraphEdges);

  // Multi-run + pagination state/actions (Phase 1)
  const activeRunId = useTimeStore((s) => s.activeRunId);
  const activeDate = useTimeStore((s) => s.activeDate);
  const setRunRegistry = useTimeStore((s) => s.setRunRegistry);
  const setActiveRunId = useTimeStore((s) => s.setActiveRunId);
  const setAvailableDates = useTimeStore((s) => s.setAvailableDates);
  const setActiveDate = useTimeStore((s) => s.setActiveDate);
  const setIsLoadingPage = useTimeStore((s) => s.setIsLoadingPage);

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

    // Only deselect on a bare map click (no node under the cursor).
    // When the cursor is on a node, hoveredNodeIdRef is non-null and the
    // layer's own onClick handler is responsible for setting the selection.
    // Without this guard the map click fires AFTER the layer click and
    // immediately nulls out the selection the layer just made.
    map.on("click", () => {
      if (hoveredNodeIdRef.current !== null) return;
      if (useTimeStore.getState().selectedNodeId !== null) {
        useTimeStore.getState().setSelectedNodeId(null);
      }
    });

    // Layer construction is handled by the rebuild effect once both the map is
    // ready and a day's payload has been merged (three-stage loading below).
    map.on("load", () => setMapReady(true));

    return () => {
      overlay.finalize();
      map.remove();
      mapRef.current = null;
      overlayRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Stage 1: on mount, fetch the run registry → pick the default run ───────
  useEffect(() => {
    let cancelled = false;
    fetch(registryUrl())
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<RunRegistry>;
      })
      .then((registry) => {
        if (cancelled) return;
        setRunRegistry(registry.runs);
        const def = registry.default_run_id ?? registry.runs[0]?.run_id ?? null;
        if (def) setActiveRunId(def);
      })
      .catch((err) =>
        console.warn("[MapVisualizer] Failed to load run_registry.json", err),
      );
    return () => {
      cancelled = true;
    };
  }, [setRunRegistry, setActiveRunId]);

  // ── Stage 2: on run change, fetch spatial.json → node coords + dates ───────
  useEffect(() => {
    if (!activeRunId) return;
    let cancelled = false;
    fetch(spatialUrl(activeRunId))
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<SpatialIndex>;
      })
      .then((spatial) => {
        if (cancelled) return;
        spatialRef.current = spatial;
        const dates = spatial.metadata.available_dates ?? [];
        setAvailableDates(dates);
        if (dates.length > 0) setActiveDate(dates[0]);
        console.info(
          `[MapVisualizer] spatial.json loaded for ${activeRunId}: ` +
            `${spatial.nodes.length} nodes, ${spatial.edge_topology.length} edges, ` +
            `${dates.length} days.`,
        );
      })
      .catch((err) =>
        console.warn(`[MapVisualizer] Failed to load spatial.json for ${activeRunId}`, err),
      );
    return () => {
      cancelled = true;
    };
  }, [activeRunId, setAvailableDates, setActiveDate]);

  // ── Stage 3: on run/date change, fetch page.json → merge into UIPayload ────
  useEffect(() => {
    if (!activeRunId || !activeDate) return;
    const spatial = spatialRef.current;
    // Guard the run-switch race: spatialRef may still hold the previous run when
    // activeRunId has just changed. Wait until stage 2 loads the matching spatial.
    if (!spatial || spatial.metadata.run_id !== activeRunId) return;
    let cancelled = false;
    setIsLoadingPage(true);
    fetch(pageUrl(activeRunId, activeDate))
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<TemporalPage>;
      })
      .then((page) => {
        if (cancelled) return;
        const payload = mergePayload(spatial, page);
        const nodeMap = new Map<number, NodeEntry>();
        for (const node of payload.nodes) nodeMap.set(node.id, node);
        payloadRef.current = payload;
        nodeMapRef.current = nodeMap;
        setMaxTimesteps(page.timesteps - 1);
        setTimestep(0);
        setPayloadVersion((v) => v + 1);
        setIsLoadingPage(false);
        console.info(
          `[MapVisualizer] page.json loaded: ${activeRunId}/${activeDate} ` +
            `(${page.timesteps} timesteps, ${payload.edges.length} edges).`,
        );
      })
      .catch((err) => {
        if (cancelled) return;
        setIsLoadingPage(false);
        console.warn(
          `[MapVisualizer] Failed to load page.json for ${activeRunId}/${activeDate}`,
          err,
        );
      });
    return () => {
      cancelled = true;
    };
  }, [activeRunId, activeDate, setMaxTimesteps, setTimestep, setIsLoadingPage]);

  // Rebuild layers whenever timestep, selection, threshold, or ego-graph data changes.
  // Also refreshes rebuildHoverRef.current so hover callbacks always see the latest
  // Zustand snapshot (selectedNodeId, attentionThreshold, egoGraphEdges).
  useEffect(() => {
    const payload = payloadRef.current;
    const overlay = overlayRef.current;
    if (!payload || !overlay) return;

    // Keep the hover rebuild function in sync with the latest closed-over state.
    rebuildHoverRef.current = (id: number | null) => {
      if (hoveredNodeIdRef.current === id) return;
      hoveredNodeIdRef.current = id;
      const ov = overlayRef.current;
      if (!ov) return;
      const s = useTimeStore.getState();
      ov.setProps({
        layers: buildLayers(
          payloadRef.current!,
          nodeMapRef.current,
          s.currentTimestep,
          {
            selectedNodeId: s.selectedNodeId,
            attentionThreshold: s.globalAttentionThreshold,
            hoveredNodeId: id,
            egoGraphEdges: s.egoGraphEdges,
          },
          setSelectedNodeId,
          (id2) => rebuildHoverRef.current(id2),
        ),
      });
    };

    overlay.setProps({
      layers: buildLayers(
        payload,
        nodeMapRef.current,
        currentTimestep,
        {
          selectedNodeId,
          attentionThreshold: globalAttentionThreshold,
          hoveredNodeId: hoveredNodeIdRef.current,
          egoGraphEdges,
        },
        setSelectedNodeId,
        (id) => rebuildHoverRef.current(id),
      ),
    });
  }, [
    currentTimestep,
    selectedNodeId,
    globalAttentionThreshold,
    egoGraphEdges,
    setSelectedNodeId,
    payloadVersion,
    mapReady,
  ]);

  return (
    <div
      ref={mapContainerRef}
      style={{ width: "100%", height: "100%" }}
      aria-label="Geospatial digital twin map"
    />
  );
}
