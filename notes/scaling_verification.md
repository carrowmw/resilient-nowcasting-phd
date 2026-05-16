# Scaling Strategy Verification Report

This report summarizes the verification of how data scaling is applied to static and dynamic features in the `ResilientIoTPipeline`.

## 1. Dynamic Features
- **Mechanism**: `StandardScaler` from `sklearn.preprocessing`.
- **Scaling Scope**: **Global (Cross-Node and Cross-Time)**.
- **Implementation**: The dynamic data tensor of shape `[num_times, num_nodes, n_features]` is flattened into `[-1, n_features]` before `fit_transform` is called.
- **Impact**: 
    - **Pros**: Preserves the absolute magnitude differences between different geographical locations and different times.
    - **Cons**: As noted in your query, it is susceptible to extreme outliers. A single sensor reporting an erroneous 10,000% spike would compress the variance of all other sensors.

## 2. Static Features
- **Mechanism**: `StandardScaler`.
- **Scaling Scope**: **Global (Cross-Node)**.
- **Features Scaled**: Currently includes `nearby_poi_count` and `dist_to_nearest_intersection`.
- **Coordinates Verification**: 
    - **Finding**: Latitude and Longitude (coordinates) are **not** currently used as static input features in the `RealIoTDataset`. 
    - **Usage**: They are utilized during the preprocessing phase (`src/preprocess.py`) for spatial joining with weather sensors and fetching POI data, but they are not passed to the model as features.
    - **POI Density**: `nearby_poi_count` acts as the POI density feature and is scaled globally across all nodes.

## 3. Target Scaling (Forecast)
- **Mechanism**: **Unscaled**.
- **Rationale**: The pipeline uses a Zero-Inflated Negative Binomial (ZINB) loss. This loss function requires non-negative integer-like counts. Applying `StandardScaler` to the target would produce signed floats, breaking the ZINB distribution assumptions.

## 4. Local vs. Global Scaling Comparison

| Feature Type | Current Implementation | Effect |
| :--- | :--- | :--- |
| **Dynamic** | **Global** | Preserves spatial variance; sensitive to global outliers. |
| **Static** | **Global** | Normalizes metadata across the network; preserves relative "importance" of nodes. |
| **Target** | **None** | Maintains raw count domain for ZINB loss. |

### Verification Details (src/dataset.py)
```python
# Scaling dynamic features (Global)
num_times, num_nodes_d, n_dyn_features = self.dynamic_data.shape
dyn_flat = self.dynamic_data.reshape(-1, n_dyn_features)
self.scaler = StandardScaler()
dyn_scaled = self.scaler.fit_transform(dyn_flat)

# Scaling static features (Global)
stat_np = stat_df.values  # [num_nodes, static_feat_dim]
self.static_scaler = StandardScaler()
stat_np_scaled = self.static_scaler.fit_transform(stat_np)
```

## Recommendations
1. **Coordinate Injection**: If coordinates are meant to be features for the `DynamicGraphLearner`, they need to be explicitly added to the `stat_df` in `src/dataset.py`.
2. **Robust Scaling**: To handle the outlier concern while maintaining global context, consider `RobustScaler` (quantile-based) instead of `StandardScaler`.
3. **Local Scaling Option**: If the model struggles to converge because some nodes are significantly busier than others, a "Local-Global" hybrid (scaling per-node but passing the node's mean/std as static features) might be worth exploring.
