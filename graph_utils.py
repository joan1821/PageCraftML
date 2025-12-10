"""
Graph construction utilities for converting Item hierarchy to graph representation.
"""
from typing import List, Dict, Tuple, Optional
import torch
from torch_geometric.data import Data
import numpy as np
from nn_server import Item


def item_to_features(item: Item, parent_width: float = 1920.0, parent_height: float = 1080.0) -> torch.Tensor:
    """
    Convert an Item to a feature vector for node representation.
    
    Features include:
    - Position (normalized x, y)
    - Size (normalized width, height)
    - Aspect ratio
    - Styling properties (normalized)
    - Layout properties (padding, margin, etc.)
    """
    # Normalize position and size relative to parent
    norm_x = item.x / parent_width if parent_width > 0 else 0.0
    norm_y = item.y / parent_height if parent_height > 0 else 0.0
    norm_width = item.width / parent_width if parent_width > 0 else 0.0
    norm_height = item.height / parent_height if parent_height > 0 else 0.0
    
    # Aspect ratio
    aspect_ratio = item.width / item.height if item.height > 0 else 1.0
    
    # Normalize styling properties
    # Color as RGB (simple hex to normalized values)
    color_rgb = hex_to_rgb_normalized(item.color)
    
    # Padding and margin (normalized)
    padding_avg = (item.paddingTop + item.paddingRight + item.paddingBottom + item.paddingLeft) / 4.0 / max(parent_width, parent_height)
    margin_avg = (item.marginTop + item.marginRight + item.marginBottom + item.marginLeft) / 4.0 / max(parent_width, parent_height)
    
    # Border and shadow properties (normalized)
    border_norm = item.borderWidth / max(parent_width, parent_height)
    shadow_blur_norm = item.boxShadowBlur / max(parent_width, parent_height)
    
    # Build feature vector
    features = torch.tensor([
        norm_x, norm_y, norm_width, norm_height,
        aspect_ratio,
        item.rotation / 360.0,  # Normalize rotation
        item.borderRadius / max(item.width, item.height, 1.0),
        item.opacity,
        *color_rgb,
        padding_avg,
        margin_avg,
        border_norm,
        shadow_blur_norm,
        float(item.backgroundGradientEnabled),
        item.filterBlur / 100.0,
        item.filterBrightness / 200.0,  # Normalize around 100%
        item.filterContrast / 200.0,
        item.zIndex / 1000.0,  # Normalize z-index
        float(item.backgroundImageEnabled) if item.backgroundImageEnabled is not None else 0.0,
    ], dtype=torch.float32)
    
    return features


def hex_to_rgb_normalized(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color to normalized RGB values."""
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            return (r, g, b)
    except:
        pass
    return (0.0, 0.0, 0.0)  # Default to black


def build_graph_from_items(items: List[Item], canvas_width: float = 1920.0, canvas_height: float = 1080.0) -> Data:
    """
    Build a PyTorch Geometric graph from a list of Items.
    
    Creates:
    - Node features from item properties
    - Edges representing parent-child relationships
    - Edge features representing spatial relationships
    """
    # Collect all items (including nested) with their indices
    all_items: List[Tuple[Item, int, Optional[int]]] = []  # (item, node_idx, parent_idx)
    node_to_item: Dict[int, Item] = {}
    
    def collect_items(item_list: List[Item], parent_idx: Optional[int] = None, start_idx: int = 0) -> int:
        """Recursively collect all items and assign indices."""
        current_idx = start_idx
        for item in item_list:
            all_items.append((item, current_idx, parent_idx))
            node_to_item[current_idx] = item
            current_idx += 1
            if item.children:
                current_idx = collect_items(item.children, current_idx - 1, current_idx)
        return current_idx
    
    collect_items(items)
    num_nodes = len(all_items)
    
    if num_nodes == 0:
        # Return empty graph
        return Data(x=torch.empty((0, 20), dtype=torch.float32), edge_index=torch.empty((2, 0), dtype=torch.long))
    
    # Build node features
    node_features = []
    for item, node_idx, parent_idx in all_items:
        # Get parent dimensions for normalization
        if parent_idx is not None and parent_idx in node_to_item:
            parent = node_to_item[parent_idx]
            parent_w, parent_h = parent.width, parent.height
        else:
            parent_w, parent_h = canvas_width, canvas_height
        
        features = item_to_features(item, parent_w, parent_h)
        node_features.append(features)
    
    x = torch.stack(node_features)
    
    # Build edges (parent-child relationships)
    edge_list = []
    edge_attrs = []
    
    for item, node_idx, parent_idx in all_items:
        if parent_idx is not None:
            edge_list.append([parent_idx, node_idx])
            # Edge features: relative position and size
            parent = node_to_item[parent_idx]
            rel_x = (item.x - parent.x) / max(parent.width, 1.0)
            rel_y = (item.y - parent.y) / max(parent.height, 1.0)
            rel_w = item.width / max(parent.width, 1.0)
            rel_h = item.height / max(parent.height, 1.0)
            edge_attrs.append([rel_x, rel_y, rel_w, rel_h])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

