"""
Graph Neural Network model for predicting layout sizes and alignments
across different device resolutions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Any, Tuple
import numpy as np


class LayoutGNN(nn.Module):
    """
    Graph Neural Network for predicting layout properties (sizes, alignments)
    across different resolutions.
    """
    
    def __init__(
        self,
        input_dim: int = 20,  # Features: x, y, width, height, flex props, etc.
        hidden_dim: int = 64,
        output_dim: int = 8,  # width, height, imageWidth, imageHeight, alignItems, justifyContent, flexDirection, alignContent
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super(LayoutGNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layers for different property types - PER NODE predictions
        # Each node gets its own size and alignment predictions
        self.size_head = nn.Linear(hidden_dim, 4)  # width, height, imageWidth, imageHeight
        self.alignment_head = nn.Linear(hidden_dim, 4)  # alignItems, justifyContent, flexDirection, alignContent
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the GNN.
        Returns PER-NODE predictions (not graph-level).
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector (not used for per-node predictions)
        
        Returns:
            Tuple of (size_predictions [num_nodes, 4], alignment_predictions [num_nodes, 4])
        """
        # Graph convolutions - learn node representations
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.batch_norm(x)
                x = F.relu(x)
                x = self.dropout(x)
        
        # Predict sizes and alignments FOR EACH NODE
        # x is [num_nodes, hidden_dim], so outputs are [num_nodes, 4]
        sizes = self.size_head(x)  # [num_nodes, 4]
        alignments = self.alignment_head(x)  # [num_nodes, 4]
        
        return sizes, alignments


def build_graph_from_items(items: List[Dict[str, Any]], canvas_width: int = 1920, canvas_height: int = 1080) -> Data:
    """
    Build a graph representation from desktop items.
    Note: Items should already be filtered to only include static items (position="static").
    Absolute items are ignored by the GNN.
    
    Nodes represent items, edges represent parent-child relationships
    and spatial proximity.
    
    Args:
        items: List of desktop items (should already be filtered to static items only)
        canvas_width: Canvas width for normalization
        canvas_height: Canvas height for normalization
    
    Returns:
        PyTorch Geometric Data object
    """
    if not items:
        # Return empty graph
        return Data(x=torch.zeros((0, 20)), edge_index=torch.zeros((2, 0), dtype=torch.long))
    
    # Build item index map
    item_map = {item['id']: idx for idx, item in enumerate(items)}
    
    # Extract node features
    node_features = []
    edges = []
    
    def extract_features(item: Dict[str, Any], parent_item: Dict[str, Any] = None) -> List[float]:
        """Extract normalized features for a single item."""
        # Normalize coordinates and sizes
        x_norm = item.get('x', 0) / canvas_width
        y_norm = item.get('y', 0) / canvas_height
        width_norm = item.get('width', 0) / canvas_width if item.get('width', 0) > 0 else -1
        height_norm = item.get('height', 0) / canvas_height if item.get('height', 0) > 0 else -1
        
        # Flex properties (one-hot encoded)
        flex_direction_map = {'row': 0, 'column': 1, 'row-reverse': 2, 'column-reverse': 3, None: 0}
        justify_map = {'flex-start': 0, 'flex-end': 1, 'center': 2, 'space-between': 3, 'space-around': 4, 'space-evenly': 5, None: 0}
        align_map = {'flex-start': 0, 'flex-end': 1, 'center': 2, 'stretch': 3, 'baseline': 4, None: 0}
        
        flex_dir = flex_direction_map.get(item.get('flexDirection'), 0)
        justify = justify_map.get(item.get('justifyContent'), 0)
        align = align_map.get(item.get('alignItems'), 0)
        
        # Position type
        position_static = 1.0 if item.get('position') == 'static' else 0.0
        position_absolute = 1.0 if item.get('position') == 'absolute' else 0.0
        
        # Image properties
        image_enabled = 1.0 if item.get('imageEnabled') else 0.0
        image_width_norm = item.get('imageWidth', 0) / canvas_width if item.get('imageWidth', 0) > 0 else 0
        image_height_norm = item.get('imageHeight', 0) / canvas_height if item.get('imageHeight', 0) > 0 else 0
        
        # Relative to parent
        rel_x = 0.0
        rel_y = 0.0
        rel_width = 0.0
        rel_height = 0.0
        
        if parent_item:
            parent_width = parent_item.get('width', canvas_width)
            parent_height = parent_item.get('height', canvas_height)
            if parent_width > 0:
                rel_x = (item.get('x', 0) - parent_item.get('x', 0)) / parent_width
                rel_width = item.get('width', 0) / parent_width if item.get('width', 0) > 0 else 0
            if parent_height > 0:
                rel_y = (item.get('y', 0) - parent_item.get('y', 0)) / parent_height
                rel_height = item.get('height', 0) / parent_height if item.get('height', 0) > 0 else 0
        
        # Depth in tree (for hierarchical understanding)
        depth = item.get('_depth', 0) / 10.0  # Normalize depth
        
        features = [
            x_norm, y_norm, width_norm, height_norm,
            flex_dir / 3.0, justify / 5.0, align / 4.0,
            position_static, position_absolute,
            image_enabled, image_width_norm, image_height_norm,
            rel_x, rel_y, rel_width, rel_height,
            depth,
            item.get('zIndex', 0) / 100.0,  # Normalized z-index
            item.get('opacity', 1.0),
            len(item.get('children', [])) / 10.0  # Number of children (normalized)
        ]
        
        return features
    
    def process_item(item: Dict[str, Any], parent: Dict[str, Any] = None, depth: int = 0):
        """Recursively process items and build graph."""
        item['_depth'] = depth
        features = extract_features(item, parent)
        node_features.append(features)
        
        current_idx = len(node_features) - 1
        
        # Add edge to parent if exists
        if parent and parent['id'] in item_map:
            parent_idx = item_map[parent['id']]
            edges.append([parent_idx, current_idx])
            edges.append([current_idx, parent_idx])  # Undirected
        
        # Process children
        for child in item.get('children', []):
            process_item(child, item, depth + 1)
    
    # Process all root items
    for item in items:
        process_item(item, None, 0)
    
    # Convert to tensors
    if not node_features:
        return Data(x=torch.zeros((0, 20)), edge_index=torch.zeros((2, 0), dtype=torch.long))
    
    x = torch.tensor(node_features, dtype=torch.float32)
    
    # Add spatial proximity edges (items close to each other)
    if len(node_features) > 1:
        for i in range(len(node_features)):
            for j in range(i + 1, len(node_features)):
                # Calculate distance
                dist_x = abs(node_features[i][0] - node_features[j][0])
                dist_y = abs(node_features[i][1] - node_features[j][1])
                dist = (dist_x ** 2 + dist_y ** 2) ** 0.5
                
                # Add edge if close enough (threshold: 0.3 normalized distance)
                if dist < 0.3:
                    edges.append([i, j])
                    edges.append([j, i])
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)


def predict_resolution_properties(
    model: LayoutGNN,
    desktop_items: List[Dict[str, Any]],
    target_width: int,
    target_height: int,
    desktop_width: int = 1920,
    desktop_height: int = 1080,
    use_model_predictions: bool = False
) -> List[Dict[str, Any]]:
    """
    Predict properties for target resolution based on desktop layout.
    Note: Only processes static items (position="static"). Absolute items are ignored.
    
    Args:
        model: Trained GNN model (or untrained for rule-based fallback)
        desktop_items: Desktop items (should already be filtered to static items only)
        target_width: Target resolution width
        target_height: Target resolution height
        desktop_width: Desktop canvas width
        desktop_height: Desktop canvas height
        use_model_predictions: If True, use model predictions; if False, use rule-based
    
    Returns:
        List of items with predicted properties for target resolution
    """
    model.eval()
    
    # Build graph from desktop items
    graph = build_graph_from_items(desktop_items, desktop_width, desktop_height)
    
    if graph.x.shape[0] == 0:
        return []
    
    # Try to use model predictions if available and requested
    if use_model_predictions:
        try:
            from decode_predictions import decode_model_predictions
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            graph_gpu = graph.to(device)
            model_gpu = model.to(device)
            
            with torch.no_grad():
                sizes_pred, alignments_pred = model_gpu(graph_gpu.x, graph_gpu.edge_index)
            
            # Decode model predictions
            predicted_items = decode_model_predictions(
                sizes_pred, alignments_pred, desktop_items, target_width, target_height,
                desktop_width, desktop_height
            )
            
            return predicted_items
        except Exception as e:
            print(f"Model prediction failed, falling back to rules: {e}")
            # Fall through to rule-based
    
    # Rule-based prediction (current implementation)
    # Scale factor
    width_scale = target_width / desktop_width
    height_scale = target_height / desktop_height
    
    def predict_item(item: Dict[str, Any], parent_item: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recursively predict properties for an item and its children."""
        # For now, use rule-based approach with GNN enhancement
        # In production, you'd use the actual model predictions
        
        predicted = {}
        
        # Scale sizes
        desktop_width_val = item.get('width', 0)
        desktop_height_val = item.get('height', 0)
        
        # Handle special width values (-1 = full width, -2 = flex grow, etc.)
        # Always include width property
        if desktop_width_val > 0:
            predicted['width'] = max(1, int(desktop_width_val * width_scale))
        elif desktop_width_val == -1:  # Full width
            # For root items, convert -1 to actual screen width
            # For children, preserve -1 to fill parent
            if parent_item is None:
                predicted['width'] = target_width  # Root item gets screen width
            else:
                predicted['width'] = -1  # Child preserves -1 to fill parent
        elif desktop_width_val == -2:  # Flex grow (50% width)
            # If parent is switching to column layout (mobile), -2 should become -1 (full width)
            # because items stack vertically and should take full width
            if target_width < 768 and parent_item is not None:
                # Check if parent will be column (row -> column conversion)
                parent_flex_dir = parent_item.get('flexDirection')
                parent_display = parent_item.get('display')
                if parent_flex_dir == 'row' or (parent_flex_dir is None and parent_display == 'flex'):
                    # Parent will switch to column, so child should be full width
                    predicted['width'] = -1
                else:
                    predicted['width'] = -2  # Keep 50% if parent stays row or already column
            else:
                predicted['width'] = -2  # Preserve for larger screens
        else:
            # Preserve other special values, or default to -1 if missing
            predicted['width'] = desktop_width_val if desktop_width_val != 0 else -1
        
        # Handle special height values
        # Always include height property
        if desktop_height_val > 0:
            predicted['height'] = max(1, int(desktop_height_val * height_scale))
        elif desktop_height_val == -1:  # Full height
            # For root items, convert -1 to scaled height based on desktop
            # For children, preserve -1 to fill parent
            if parent_item is None:
                # Scale desktop height proportionally
                desktop_actual_height = desktop_height if desktop_height > 0 else 1080
                predicted['height'] = max(1, int(desktop_actual_height * height_scale))
            else:
                predicted['height'] = -1  # Child preserves -1 to fill parent
        elif desktop_height_val == -2:  # Flex grow (maintain)
            predicted['height'] = -2
        else:
            # Preserve other special values, or scale if it's a positive value
            if desktop_height_val > 0:
                predicted['height'] = max(1, int(desktop_height_val * height_scale))
            else:
                predicted['height'] = desktop_height_val
        
        # Image dimensions
        if item.get('imageEnabled'):
            img_w = item.get('imageWidth', 0)
            img_h = item.get('imageHeight', 0)
            if img_w > 0:
                predicted['imageWidth'] = max(1, int(img_w * width_scale))
            if img_h > 0:
                predicted['imageHeight'] = max(1, int(img_h * height_scale))
        
        # Alignment properties - adapt based on orientation
        # For mobile/tablet, often switch to column layout
        if target_width < 768:  # Mobile
            if item.get('flexDirection') == 'row':
                predicted['flexDirection'] = 'column'
            elif item.get('flexDirection') is None and item.get('display') == 'flex':
                predicted['flexDirection'] = 'column'
            else:
                # Preserve existing flexDirection if it's already column or other
                if item.get('flexDirection'):
                    predicted['flexDirection'] = item.get('flexDirection')
            
            # Adjust justifyContent for vertical layouts
            if item.get('justifyContent') == 'space-between':
                predicted['justifyContent'] = 'flex-start'
            else:
                # Preserve other justifyContent values
                if item.get('justifyContent'):
                    predicted['justifyContent'] = item.get('justifyContent')
            
            # Preserve alignItems if it exists
            if item.get('alignItems'):
                predicted['alignItems'] = item.get('alignItems')
        else:
            # Preserve desktop alignments for larger screens
            if item.get('flexDirection'):
                predicted['flexDirection'] = item.get('flexDirection')
            if item.get('justifyContent'):
                predicted['justifyContent'] = item.get('justifyContent')
            if item.get('alignItems'):
                predicted['alignItems'] = item.get('alignItems')
            if item.get('alignContent'):
                predicted['alignContent'] = item.get('alignContent')
        
        # Process children
        if item.get('children'):
            predicted['children'] = []
            # Determine if this item will be column layout (for child width adjustments)
            item_will_be_column = False
            if target_width < 768:  # Mobile
                item_flex_dir = item.get('flexDirection')
                item_display = item.get('display')
                # Check if item will switch from row to column
                if item_flex_dir == 'row' or (item_flex_dir is None and item_display == 'flex'):
                    item_will_be_column = True
                # Or if it's already predicted as column
                elif predicted.get('flexDirection') == 'column':
                    item_will_be_column = True
            
            for child in item['children']:
                child_predicted = predict_item(child, item)
                # Preserve child ID
                child_predicted['id'] = child.get('id')
                
                # If parent is switching to column layout and child has -2 width, change to -1
                # This ensures children take full width when stacked vertically
                if item_will_be_column and child_predicted.get('width') == -2:
                    child_predicted['width'] = -1
                
                predicted['children'].append(child_predicted)
        
        return predicted
    
    # Predict for all items
    predicted_items = []
    for item in desktop_items:
        predicted = predict_item(item)
        # Merge with original item ID
        predicted['id'] = item.get('id')
        predicted_items.append(predicted)
    
    return predicted_items

