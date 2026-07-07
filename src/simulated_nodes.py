import os
import random
import logging
from typing import Dict, List, Any
import torch
import numpy as np
import osmnx as ox

from features import StaticFeatureLoader, REQUIRED_KEYS, OSM_CENTER, OSM_DIST

logger = logging.getLogger(__name__)

class SimulatedNodeInjector:
    """
    Handles the deterministic generation, static feature extraction, and dynamic feature
    injection for simulated (zero-shot) nodes in the UI payload.
    """
    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        bbox: Dict[str, float],
        static_scaler: Any,
        device: torch.device
    ):
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.device = device
        
        if self.num_nodes > 0:
            logger.info(f"Dynamically sampling {self.num_nodes} simulated nodes from road graph...")
            self.coords = self._sample_osm_nodes(self.num_nodes, bbox)
            self.nodes_meta = self._create_nodes_meta(self.coords)
            self.sim_static = self._build_simulated_static(static_scaler, self.nodes_meta).to(self.device)
            self.sim_struct_masks = self._build_all_missing_masks().to(self.device)
            # Dynamic features are completely unknown (NaN) -> will be zeroed out
            self.sim_dynamic = torch.full((self.num_nodes, self.seq_len, 3), float("nan"), device=self.device)
        else:
            self.coords = []
            self.nodes_meta = []
            # Empty tensors if no simulated nodes
            self.sim_static = torch.empty((0, len(REQUIRED_KEYS)), device=self.device)
            self.sim_struct_masks = torch.empty((0, self.seq_len, 2), device=self.device)
            self.sim_dynamic = torch.empty((0, self.seq_len, 3), device=self.device)

    def _sample_osm_nodes(self, n: int, bbox: Dict[str, float]) -> List[tuple]:
        """Fetch road graph, filter by bounding box, and deterministically sample n intersections."""
        # Use cache if possible or fetch
        G = ox.graph_from_point(OSM_CENTER, dist=OSM_DIST, network_type='walk')
        
        all_nodes = []
        for node, data in G.nodes(data=True):
            lat = data.get('y')
            lon = data.get('x')
            if lat is None or lon is None:
                continue
            if bbox['lat_min'] <= lat <= bbox['lat_max'] and bbox['lon_min'] <= lon <= bbox['lon_max']:
                all_nodes.append((lat, lon))
        
        assert len(all_nodes) >= n, f"CRITICAL: Found only {len(all_nodes)} nodes in bounding box, but {n} requested."
        
        # Deterministic sampling
        random.seed(42)
        return random.sample(all_nodes, n)
        
    def _create_nodes_meta(self, coords: List[tuple]) -> List[Dict]:
        meta = []
        for j, (lat, lon) in enumerate(coords):
            meta.append({
                "location_name": f"Simulated_Node_{j}",
                "lat": lat,
                "lon": lon,
                "is_simulated": True
            })
        return meta

    def _build_simulated_static(self, static_scaler: Any, nodes_meta: List[Dict]) -> torch.Tensor:
        """Create static features fetching spatial features from Google Places and OSMnx."""
        loader = StaticFeatureLoader(cache_path="data/external/simulated_places_cache.json")

        # Query missing Google features
        for meta in nodes_meta:
            loc = meta["location_name"]
            if loc not in loader.cache:
                logger.info(f"Fetching live data for simulated node: {loc}")
                loader.cache[loc] = loader._fetch_location_data(meta["lat"], meta["lon"])
                loader._save_cache()

        # Query OSM features
        sim_sensors_data = [{"location": m["location_name"], "lat": m["lat"], "lon": m["lon"]} for m in nodes_meta]
        osm_cache = loader._build_osm_cache(sim_sensors_data)

        rows = []
        for meta in nodes_meta:
            loc = meta["location_name"]
            feats = dict(loader.cache[loc])
            feats.update(osm_cache.get(loc, {}))

            if 'is_school_zone' in feats:
                feats['is_school_zone'] = float(feats['is_school_zone'])

            row = [float(feats[k]) for k in REQUIRED_KEYS]
            rows.append(row)

        raw_static_np = np.array(rows, dtype=np.float32)
        scaled_static_np = static_scaler.transform(raw_static_np)
        return torch.tensor(scaled_static_np, dtype=torch.float32)

    def _build_all_missing_masks(self) -> torch.Tensor:
        """
        Builds the [num_simulated, seq_len, 2] structural masks.
        Simulated nodes have no physical sensor, so total MNAR missingness is 1.0.
        Night missingness could technically be temporal, but for simplicity of purely simulated
        (zero-shot) nodes, they are treated as entirely structurally missing everywhere.
        """
        # We fill it with 1.0 across both dimensions.
        return torch.ones((self.num_nodes, self.seq_len, 2), dtype=torch.float32)

    def inject_static(self, real_static: torch.Tensor) -> torch.Tensor:
        """Concatenates real and simulated static features."""
        return torch.cat([real_static, self.sim_static], dim=0)

    def inject_dynamic_item(self, raw_item: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Takes a single dataset batch item (for real nodes) and appends simulated tensors.
        Returns the mutated dictionary.
        """
        combined = dict(raw_item)
        
        real_dynamic = combined["dynamic_features"].to(self.device)
        real_struct_masks = combined["structural_masks"].to(self.device)
        
        num_real = real_dynamic.shape[0]
        num_total = num_real + self.num_nodes
        
        combined["dynamic_features"] = torch.cat([real_dynamic, self.sim_dynamic], dim=0)
        combined["structural_masks"] = torch.cat([real_struct_masks, self.sim_struct_masks], dim=0)
        
        # simulated_nodes_mask marks simulated nodes so they don't send corrupt messages
        sim_mask = torch.zeros(num_total, dtype=torch.bool, device=self.device)
        if self.num_nodes > 0:
            sim_mask[num_real:] = True
        combined["simulated_nodes_mask"] = sim_mask
        
        return combined
