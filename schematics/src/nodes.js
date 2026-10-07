// Sensor-field layout shared by both schematics, normalised to 0..1.
// SENSORS: 30 well-spread real station positions from the Newcastle pedestrian
// network (farthest-point sample, far-north outlier removed). UNSENSED: places
// in the gaps between them, where no sensor exists.
window.SCHEMATIC_NODES = {
  sensors: [
    [0.363, 0.822], [1, 0.178], [0.351, 0], [0.003, 0.404], [0.819, 0.697], [0.547, 0.371],
    [0, 0.825], [0.262, 0.291], [0.608, 0.881], [0.136, 0.617], [0.806, 0.303], [0.597, 0.653],
    [0.259, 1], [0.79, 0.496], [0.431, 0.504], [0.21, 0.769], [0.639, 0.504], [0.323, 0.43],
    [0.713, 0.602], [0.659, 0.323], [0.687, 0.733], [0.398, 0.932], [0.245, 0.864], [0.208, 0.656],
    [0.357, 0.745], [0.637, 0.588], [0.724, 0.294], [0.683, 0.659], [0.749, 0.718], [0.314, 0.967],
  ],
  unsensed: [[0.5, 0.15], [0.15, 0.2], [0.92, 0.45], [0.47, 0.65], [0.08, 1.0], [0.9, 0.92]],
  // Index into `sensors` for each failure state shown in the "reality" panel.
  states: {
    intermittent: [5, 14, 18, 22],
    silent: [9, 8, 29],
    lost: [1, 7, 26],
  },
};
