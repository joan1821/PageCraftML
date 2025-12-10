"""
Responsive behavior prediction and application logic.
"""
from typing import List, Dict, Tuple
import torch
from nn_server import Item
from graph_utils import build_graph_from_items
from gnn_model import ResponsiveGNN, predict_responsive_behavior


def apply_responsive_predictions(
    items: List[Item],
    predictions: Dict,
    target_width: float = 375.0,
    target_height: float = 667.0,
    source_width: float = 1920.0,
    source_height: float = 1080.0
) -> List[Item]:
    """
    Apply GNN predictions to items, creating optimized mobile layouts.
    
    Adaptation types:
    0 = Scale (proportional resize)
    1 = Stack (vertical stacking)
    2 = Hide (hide on mobile)
    3 = Reposition (move to new position)
    """
    # Flatten items to match prediction indices
    all_items_flat: List[Tuple[Item, int]] = []
    
    def flatten_items(item_list: List[Item], start_idx: int = 0) -> int:
        """Flatten nested items into a list."""
        current_idx = start_idx
        for item in item_list:
            all_items_flat.append((item, current_idx))
            current_idx += 1
            if item.children:
                current_idx = flatten_items(item.children, current_idx)
        return current_idx
    
    flatten_items(items)
    
    # Apply predictions
    scales = predictions['scales']
    layouts = predictions['layouts']
    adaptations = predictions['adaptations']
    
    def apply_to_items(item_list: List[Item], parent_x: float = 0, parent_y: float = 0, start_idx: int = 0) -> Tuple[List[Item], int]:
        """Recursively apply predictions to items."""
        result_items = []
        current_idx = start_idx
        
        for item in item_list:
            if current_idx >= len(all_items_flat):
                result_items.append(item)
                continue
            
            item_idx = current_idx
            adaptation_type = adaptations[item_idx]
            
            # Get predictions for this item
            scale_x, scale_y = scales[item_idx]
            pred_x, pred_y, pred_w, pred_h = layouts[item_idx]
            
            # Create updated item
            updated_item = item.model_copy(deep=True)
            
            if adaptation_type == 0:  # Scale
                # Proportional scaling
                scale_factor = min(target_width / source_width, target_height / source_height)
                updated_item.width = item.width * scale_factor
                updated_item.height = item.height * scale_factor
                updated_item.x = item.x * scale_factor
                updated_item.y = item.y * scale_factor
                # Scale padding and margins proportionally
                updated_item.padding = item.padding * scale_factor
                updated_item.paddingTop = item.paddingTop * scale_factor
                updated_item.paddingRight = item.paddingRight * scale_factor
                updated_item.paddingBottom = item.paddingBottom * scale_factor
                updated_item.paddingLeft = item.paddingLeft * scale_factor
                updated_item.margin = item.margin * scale_factor
                updated_item.marginTop = item.marginTop * scale_factor
                updated_item.marginRight = item.marginRight * scale_factor
                updated_item.marginBottom = item.marginBottom * scale_factor
                updated_item.marginLeft = item.marginLeft * scale_factor
                updated_item.borderRadius = item.borderRadius * scale_factor
                updated_item.borderWidth = item.borderWidth * scale_factor
                updated_item.boxShadowBlur = item.boxShadowBlur * scale_factor
                
            elif adaptation_type == 1:  # Stack
                # Vertical stacking - maintain width, adjust position
                updated_item.width = min(item.width, target_width * 0.95)
                updated_item.height = item.height * (target_width / source_width)
                updated_item.x = target_width * 0.025  # Small margin
                # Y position will be set by stacking logic
                updated_item.y = pred_y * target_height
                
            elif adaptation_type == 2:  # Hide
                # Hide by setting opacity to 0 and size to 0
                updated_item.opacity = 0.0
                updated_item.width = 0
                updated_item.height = 0
                
            elif adaptation_type == 3:  # Reposition
                # Use predicted layout
                updated_item.x = pred_x * target_width
                updated_item.y = pred_y * target_height
                updated_item.width = max(pred_w * target_width, 10)  # Minimum width
                updated_item.height = max(pred_h * target_height, 10)  # Minimum height
            
            # Apply to children recursively
            if updated_item.children:
                updated_children, next_idx = apply_to_items(
                    updated_item.children,
                    updated_item.x,
                    updated_item.y,
                    current_idx + 1
                )
                updated_item.children = updated_children
                current_idx = next_idx
            else:
                current_idx += 1
            
            result_items.append(updated_item)
        
        return result_items, current_idx
    
    return apply_to_items(items)[0]


def predict_and_apply_responsive(
    items: List[Item],
    model: ResponsiveGNN,
    target_width: float = 375.0,
    target_height: float = 667.0,
    source_width: float = 1920.0,
    source_height: float = 1080.0
) -> List[Item]:
    """
    Complete pipeline: build graph, predict, and apply responsive behavior.
    """
    # Build graph from items
    graph = build_graph_from_items(items, source_width, source_height)
    
    # Add batch dimension if needed
    if graph.batch is None:
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    
    # Get predictions
    predictions = predict_responsive_behavior(
        model, graph, target_width, target_height, source_width, source_height
    )
    
    # Apply predictions to items
    optimized_items = apply_responsive_predictions(
        items, predictions, target_width, target_height, source_width, source_height
    )
    
    return optimized_items

