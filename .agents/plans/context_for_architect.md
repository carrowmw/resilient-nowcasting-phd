# Architectural Context: Frontend Attention Edges

## Inquiry
Determine the origin layer of the "top attention edges" displayed in the digital twin frontend (whether they originate from the decoders, the temporal transformer, or the GATConv layer).

## Findings
The top attention edges rendered in the frontend originate explicitly from the **GATConv** (Graph Attention Network) layer within the Spatial Encoder, not from the temporal transformer or any of the decoders.

### Execution Trace
1. **`src/encoder.py` (Line 105):** Inside `MaskedSpatiotemporalEncoder`, the PyTorch Geometric layer `self.gat` (which is a `GATv2Conv`) is called with `return_attention_weights=True`. This returns the `alpha` tensor, representing the spatial attention coefficients between neighboring nodes.
2. **`src/pipeline.py` (Line 112):** The `ResilientIoTPipeline` runs the encoder in analysis mode and catches the `alpha` tensor, passing it through to the final output dictionary as `attention_weights`.
3. **`src/export_ui_payload.py` (Lines 1057-1073):** The payload exporter extracts this `alpha` tensor. It averages the attention weights across the multiple attention heads for the final timestep of the historical sequence. These scalar weights are then attached to the JSON graph payload as `attention_sequence` and aggregated to calculate the `total_outgoing_attention` centrality rankings displayed in the frontend.

## Conclusion for Architect / User
The visual links on the map and the edge thickness correspond directly to the **spatial attention mechanism** (`GATv2Conv`), demonstrating how much information the model allows to flow between specific spatial nodes during the encoding phase.
