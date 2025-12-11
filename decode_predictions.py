"""
Decode model predictions back to item properties.
Converts normalized model outputs to actual width/height/alignment values.
"""
import torch
from typing import Dict, List, Any, Tuple


def decode_property_from_tensor(normalized_value: float, prop_name: str, 
                                target_width: int = None, target_height: int = None,
                                desktop_value: Any = None) -> Any:
    """
    Decode a normalized tensor value back to an actual property value.
    Inverse of encode_property_to_tensor.
    
    Args:
        normalized_value: Normalized value from model [0, 1]
        prop_name: Property name ('width', 'height', 'flexDirection', etc.)
        target_width: Target resolution width (for width properties)
        target_height: Target resolution height (for height properties)
        desktop_value: Original desktop value (for reference)
    
    Returns:
        Decoded property value
    """
    if prop_name in ['width', 'height', 'imageWidth', 'imageHeight']:
        # Numeric properties
        if normalized_value >= 0.95:  # Close to 1.0 -> full width/height
            return -1
        elif 0.45 <= normalized_value <= 0.55:  # Close to 0.5 -> half width/height
            return -2
        else:
            # Regular numeric value - denormalize
            if prop_name in ['width', 'imageWidth']:
                if target_width:
                    return max(1, int(normalized_value * target_width))
                else:
                    return max(1, int(normalized_value * 1920))
            else:
                if target_height:
                    return max(1, int(normalized_value * target_height))
                else:
                    return max(1, int(normalized_value * 1080))
    else:
        # Categorical properties - decode from normalized index
        if prop_name == 'flexDirection':
            options = ['row', 'column', 'row-reverse', 'column-reverse']
        elif prop_name == 'justifyContent':
            options = ['flex-start', 'flex-end', 'center', 'space-between', 'space-around', 'space-evenly']
        elif prop_name == 'alignItems':
            options = ['flex-start', 'flex-end', 'center', 'stretch', 'baseline']
        elif prop_name == 'alignContent':
            options = ['flex-start', 'flex-end', 'center', 'space-between', 'space-around', 'stretch']
        else:
            return None
        
        # Convert normalized value to index
        idx = int(normalized_value * len(options))
        idx = max(0, min(idx, len(options) - 1))  # Clamp to valid range
        
        return options[idx] if idx < len(options) else None


def decode_model_predictions(
    sizes_pred: torch.Tensor,  # [num_nodes, 4]
    alignments_pred: torch.Tensor,  # [num_nodes, 4]
    desktop_items: List[Dict[str, Any]],
    target_width: int,
    target_height: int
) -> List[Dict[str, Any]]:
    """
    Decode model predictions back to item properties.
    
    Args:
        sizes_pred: Model size predictions [num_nodes, 4] - [width, height, imageWidth, imageHeight]
        alignments_pred: Model alignment predictions [num_nodes, 4] - [alignItems, justifyContent, flexDirection, alignContent]
        desktop_items: Original desktop items (for structure and IDs)
        target_width: Target resolution width
        target_height: Target resolution height
    
    Returns:
        List of predicted items with decoded properties
    """
    # Convert to numpy/cpu for processing
    sizes_np = sizes_pred.cpu().numpy() if isinstance(sizes_pred, torch.Tensor) else sizes_pred
    alignments_np = alignments_pred.cpu().numpy() if isinstance(alignments_pred, torch.Tensor) else alignments_pred
    
    # Build item list matching node order
    items_list = []
    item_to_idx = {}
    idx = 0
    
    def collect_items(item: Dict[str, Any]):
        """Collect all items in the same order as graph nodes."""
        nonlocal idx
        items_list.append(item)
        item_to_idx[item.get('id')] = idx
        idx += 1
        if 'children' in item and item['children']:
            for child in item['children']:
                collect_items(child)
    
    for item in desktop_items:
        collect_items(item)
    
    # Decode predictions for each item
    predicted_items = []
    for i, item in enumerate(items_list):
        if i >= len(sizes_np):
            break
        
        predicted = {'id': item.get('id')}
        
        # Decode size properties
        width_norm = sizes_np[i][0]
        height_norm = sizes_np[i][1]
        image_width_norm = sizes_np[i][2]
        image_height_norm = sizes_np[i][3]
        
        # Decode width
        desktop_width_val = item.get('width', 0)
        if desktop_width_val == -1 and i == 0:  # Root item with -1 width
            predicted['width'] = target_width
        else:
            predicted['width'] = decode_property_from_tensor(width_norm, 'width', target_width, None, desktop_width_val)
        
        # Decode height
        desktop_height_val = item.get('height', 0)
        if desktop_height_val == -1 and i == 0:  # Root item with -1 height
            # Scale desktop height proportionally
            predicted['height'] = max(1, int(desktop_height * (target_height / desktop_height)))
        elif desktop_height_val > 0:
            # For positive heights, scale proportionally based on desktop height
            height_scale = target_height / desktop_height if desktop_height > 0 else (target_height / 1080)
            predicted['height'] = max(1, int(desktop_height_val * height_scale))
        elif desktop_height_val == -2:
            predicted['height'] = -2
        else:
            predicted['height'] = -1  # Default to full height
        
        # Decode image dimensions
        if item.get('imageEnabled'):
            img_w = decode_property_from_tensor(image_width_norm, 'imageWidth', target_width)
            img_h = decode_property_from_tensor(image_height_norm, 'imageHeight', target_height)
            if img_w > 0:
                predicted['imageWidth'] = img_w
            if img_h > 0:
                predicted['imageHeight'] = img_h
        
        # Decode alignment properties
        alignItems_norm = alignments_np[i][0]
        justifyContent_norm = alignments_np[i][1]
        flexDirection_norm = alignments_np[i][2]
        alignContent_norm = alignments_np[i][3]
        
        alignItems = decode_property_from_tensor(alignItems_norm, 'alignItems')
        if alignItems:
            predicted['alignItems'] = alignItems
        
        justifyContent = decode_property_from_tensor(justifyContent_norm, 'justifyContent')
        if justifyContent:
            predicted['justifyContent'] = justifyContent
        
        flexDirection = decode_property_from_tensor(flexDirection_norm, 'flexDirection')
        if flexDirection:
            predicted['flexDirection'] = flexDirection
        
        alignContent = decode_property_from_tensor(alignContent_norm, 'alignContent')
        if alignContent:
            predicted['alignContent'] = alignContent
        
        # Process children recursively
        if 'children' in item and item['children']:
            predicted['children'] = []
            child_start_idx = i + 1
            for child_idx, child in enumerate(item['children']):
                # Find child in items_list
                child_node_idx = child_start_idx + child_idx
                if child_node_idx < len(sizes_np):
                    # Decode child (simplified - would need recursive call)
                    child_predicted = {'id': child.get('id')}
                    # Decode child properties
                    child_width_norm = sizes_np[child_node_idx][0]
                    child_height_norm = sizes_np[child_node_idx][1]
                    child_flex_dir_norm = alignments_np[child_node_idx][2]
                    
                    child_width_val = child.get('width', 0)
                    # If parent switches to column and child has -2 width, change to -1
                    if is_mobile and parent_was_row and child_width_val == -2:
                        child_predicted['width'] = -1
                    else:
                        child_predicted['width'] = decode_property_from_tensor(child_width_norm, 'width', target_width, None, child_width_val)
                    
                    child_height_val = child.get('height', 0)
                    if child_height_val > 0:
                        # Scale child height proportionally
                        height_scale = target_height / desktop_height if desktop_height > 0 else (target_height / 1080)
                        child_predicted['height'] = max(1, int(child_height_val * height_scale))
                    elif child_height_val == -1:
                        child_predicted['height'] = -1
                    elif child_height_val == -2:
                        child_predicted['height'] = -2
                    else:
                        child_predicted['height'] = -1  # Default
                    
                    child_flex_dir = decode_property_from_tensor(child_flex_dir_norm, 'flexDirection')
                    if child_flex_dir:
                        child_predicted['flexDirection'] = child_flex_dir
                    
                    predicted['children'].append(child_predicted)
        
        predicted_items.append(predicted)
    
    return predicted_items

