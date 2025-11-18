"""
Responsive behavior prediction and application logic.
"""
from typing import List, Dict, Tuple
import torch
from item_types import Item
from graph_utils import build_graph_from_items
from gnn_model import ResponsiveGNN, predict_responsive_behavior


def apply_responsive_predictions(
    items: List[Item],
    predictions: Dict,
    target_width: float = 375.0,
    target_height: float = 667.0,
    source_width: float = 1920.0,
    source_height: float = 1080.0
) -> List[Dict]:
    """
    Apply responsive resizing to items - ONLY resizes components and images.
    NEVER changes position property (static/absolute).
    
    Rules:
    - Preserve position property (static stays static, absolute stays absolute)
    - Only resize width/height proportionally
    - For static items: don't modify x/y coordinates
    - For absolute items: scale x/y proportionally
    - Scale images (imageWidth, imageHeight) if they exist
    - Scale padding, margins, borders proportionally
    """
    # Calculate proportional scale factor
    scale_factor = min(target_width / source_width, target_height / source_height)
    print(f"[RESPONSIVE] Scale factor: {scale_factor:.4f} (target: {target_width}x{target_height}, source: {source_width}x{source_height})")
    print(f"[RESPONSIVE] Processing {len(items)} items")
    
    def apply_to_items(item_list: List[Item]) -> List[Item]:
        """Recursively apply proportional resizing to items."""
        result_items = []
        
        # MINIMAL: Only include size and alignment properties that changed
        # Everything else (styling, colors, etc.) stays the same as desktop
        minimal_fields = {
            'id',           # Required to identify item
            'position',     # Position type (static/absolute)
            'x', 'y',       # Coordinates (only for absolute, or if changed)
            'width', 'height',  # Resized dimensions
            'padding', 'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',  # Scaled padding
            'margin', 'marginTop', 'marginRight', 'marginBottom', 'marginLeft',  # Scaled margin
            'borderRadius', 'borderWidth',  # Scaled border properties
            'boxShadowBlur', 'boxShadowOffsetX', 'boxShadowOffsetY', 'boxShadowSpread',  # Scaled shadow
            'imageWidth', 'imageHeight',  # Scaled image dimensions (if applicable)
            'children'     # Recursive children
        }
        
        for item in item_list:
            # Convert to dict and filter to only minimal fields (size/alignment only)
            item_dict = item.model_dump() if hasattr(item, 'model_dump') else item.dict()
            
            # Filter to only minimal fields - size and alignment properties
            filtered_dict = {k: v for k, v in item_dict.items() if k in minimal_fields}
            
            # Get position (handle None case)
            item_position = filtered_dict.get('position') or "absolute"
            original_width = filtered_dict['width']
            original_height = filtered_dict['height']
            
            # CRITICAL: Never change the position property
            filtered_dict['position'] = item_position
            
            # Resize dimensions proportionally
            filtered_dict['width'] = filtered_dict['width'] * scale_factor
            filtered_dict['height'] = filtered_dict['height'] * scale_factor
            
            print(f"[RESPONSIVE] Item {filtered_dict['id']}: {original_width}x{original_height} -> {filtered_dict['width']:.1f}x{filtered_dict['height']:.1f} (position: {item_position})")
            
            # Only modify x/y for absolute positioned items
            # Static items use margins, so x/y should not be changed
            if item_position == "absolute":
                filtered_dict['x'] = filtered_dict['x'] * scale_factor
                filtered_dict['y'] = filtered_dict['y'] * scale_factor
            # For static items, x/y are not used, so we leave them as-is
            
            # Scale padding proportionally (only include if non-zero)
            padding_keys = ['padding', 'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft']
            has_padding = any(filtered_dict.get(k, 0) != 0 for k in padding_keys)
            if has_padding:
                for key in padding_keys:
                    if filtered_dict.get(key, 0) != 0:
                        filtered_dict[key] = filtered_dict[key] * scale_factor
            else:
                for key in padding_keys:
                    filtered_dict.pop(key, None)
            
            # Scale margins proportionally (only include if non-zero)
            margin_keys = ['margin', 'marginTop', 'marginRight', 'marginBottom', 'marginLeft']
            has_margin = any(filtered_dict.get(k, 0) != 0 for k in margin_keys)
            if has_margin:
                for key in margin_keys:
                    if filtered_dict.get(key, 0) != 0:
                        filtered_dict[key] = filtered_dict[key] * scale_factor
            else:
                for key in margin_keys:
                    filtered_dict.pop(key, None)
            
            # Scale border and shadow properties (only include if non-zero)
            if filtered_dict.get('borderRadius', 0) != 0:
                filtered_dict['borderRadius'] = filtered_dict['borderRadius'] * scale_factor
            else:
                filtered_dict.pop('borderRadius', None)
            
            if filtered_dict.get('borderWidth', 0) != 0:
                filtered_dict['borderWidth'] = filtered_dict['borderWidth'] * scale_factor
            else:
                filtered_dict.pop('borderWidth', None)
            
            # Only include shadow if it has non-zero values
            shadow_keys = ['boxShadowBlur', 'boxShadowOffsetX', 'boxShadowOffsetY', 'boxShadowSpread']
            has_shadow = any(filtered_dict.get(k, 0) != 0 for k in shadow_keys)
            if has_shadow:
                for key in shadow_keys:
                    if filtered_dict.get(key, 0) != 0:
                        filtered_dict[key] = filtered_dict[key] * scale_factor
            else:
                for key in shadow_keys:
                    filtered_dict.pop(key, None)
            
            # Scale images if they exist
            if filtered_dict.get('imageWidth'):
                filtered_dict['imageWidth'] = filtered_dict['imageWidth'] * scale_factor
            if filtered_dict.get('imageHeight'):
                filtered_dict['imageHeight'] = filtered_dict['imageHeight'] * scale_factor
            else:
                # Remove image fields if not applicable
                filtered_dict.pop('imageWidth', None)
                filtered_dict.pop('imageHeight', None)
            
            # Recursively apply to children
            if filtered_dict.get('children'):
                filtered_dict['children'] = apply_to_items(filtered_dict['children'])
            else:
                # Remove empty children array
                filtered_dict.pop('children', None)
            
            # Remove x/y for static items (not used)
            if item_position == "static":
                filtered_dict.pop('x', None)
                filtered_dict.pop('y', None)
            
            # Remove None values and empty arrays
            filtered_dict = {k: v for k, v in filtered_dict.items() if v is not None and v != []}
            
            # Create minimal dict response (don't create Item object, just return dict)
            # This allows partial data - frontend will merge with desktop version
            result_items.append(filtered_dict)
        
        return result_items
    
    return apply_to_items(items)


def predict_and_apply_responsive(
    items: List[Item],
    model: ResponsiveGNN,
    target_width: float = 375.0,
    target_height: float = 667.0,
    source_width: float = 1920.0,
    source_height: float = 1080.0
) -> List[Dict]:
    """
    Complete pipeline: apply proportional responsive resizing.
    
    Note: Currently uses simple proportional scaling. The GNN model is loaded
    but predictions are not used - we only resize components and images,
    preserving all position properties.
    """
    # For now, we use simple proportional scaling
    # The GNN model is available for future enhancements (e.g., smart breakpoints)
    # but we don't use its predictions to avoid changing position properties
    
    # Create empty predictions dict (not used, but kept for API compatibility)
    predictions = {
        'scales': [],
        'layouts': [],
        'breakpoints': [],
        'adaptations': []
    }
    
    # Apply proportional resizing (ignores predictions, just does scaling)
    optimized_items = apply_responsive_predictions(
        items, predictions, target_width, target_height, source_width, source_height
    )
    
    return optimized_items

