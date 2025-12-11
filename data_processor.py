"""
Data processing utilities for cleaning and merging layout data.
"""
from typing import Dict, List, Any, Set
import copy


# Properties relevant for sizes and alignments (from RESOLUTION_OVERRIDE_PROPERTIES)
SIZE_ALIGNMENT_PROPERTIES: Set[str] = {
    'width',
    'height',
    'imageWidth',
    'imageHeight',
    'alignItems',
    'justifyContent',
    'flexDirection',
    'alignContent',
    'id',  # Always keep ID for merging
    'children'  # Keep children structure
}

# Properties to preserve but not use for prediction
STRUCTURAL_PROPERTIES: Set[str] = {
    'id',
    'children',
    'parentId',
    'name',
    'position',
    'display'
}


def extract_target_properties(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only size/alignment properties from an item.
    This is what we want the model to predict.
    
    Args:
        item: Item dictionary
    
    Returns:
        Item with only size/alignment properties
    """
    target = {'id': item.get('id')}
    
    size_align_props = ['width', 'height', 'imageWidth', 'imageHeight', 
                        'alignItems', 'justifyContent', 'flexDirection', 'alignContent']
    
    for prop in size_align_props:
        if prop in item:
            target[prop] = item[prop]
    
    # Recursively process children
    if 'children' in item and item['children']:
        target['children'] = [extract_target_properties(child) for child in item['children']]
    
    return target


def clean_item_for_prediction(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean an item to keep only properties relevant for size/alignment prediction.
    Removes styling, content, and other irrelevant properties.
    
    Args:
        item: Original item dictionary
    
    Returns:
        Cleaned item with only size/alignment relevant properties
    """
    cleaned = {}
    
    # Always keep ID and children structure
    cleaned['id'] = item.get('id')
    
    # Keep size/alignment properties
    for prop in SIZE_ALIGNMENT_PROPERTIES:
        if prop in item:
            cleaned[prop] = item[prop]
    
    # Keep structural properties needed for graph building
    for prop in STRUCTURAL_PROPERTIES:
        if prop in item:
            cleaned[prop] = item[prop]
    
    # Recursively clean children
    if 'children' in item and item['children']:
        cleaned['children'] = [
            clean_item_for_prediction(child) for child in item['children']
        ]
    
    return cleaned


def filter_static_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter to keep only items with position="static" and their static children.
    Ignores items with position="absolute" or other non-static positions.
    
    Args:
        items: List of item dictionaries
    
    Returns:
        List of static items only (absolute items are excluded)
    """
    def filter_item(item: Dict[str, Any]) -> Dict[str, Any] | None:
        """Recursively filter to keep only static items."""
        # Skip non-static items (absolute, etc.)
        if item.get('position') != 'static':
            return None
        
        # Process children - filter to keep only static children
        filtered_children = []
        if 'children' in item and item['children']:
            for child in item['children']:
                filtered_child = filter_item(child)
                if filtered_child is not None:
                    filtered_children.append(filtered_child)
        
        # Create filtered item
        filtered_item = copy.deepcopy(item)
        if filtered_children:
            filtered_item['children'] = filtered_children
        elif 'children' in filtered_item:
            filtered_item['children'] = []
        
        return filtered_item
    
    filtered_items = []
    for item in items:
        filtered = filter_item(item)
        if filtered is not None:
            filtered_items.append(filtered)
    
    return filtered_items


def clean_items_for_prediction(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean a list of items for prediction.
    Filters to keep only static items first, then cleans the remaining items.
    
    Args:
        items: List of item dictionaries
    
    Returns:
        List of cleaned static items only
    """
    # First filter to keep only static items
    static_items = filter_static_items(items)
    # Then clean the remaining items
    return [clean_item_for_prediction(item) for item in static_items]


def merge_predicted_properties(
    original_item: Dict[str, Any],
    predicted_properties: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a minimal item with ONLY size/alignment properties from predictions.
    Does NOT include other properties like color, padding, margin, etc.
    
    Args:
        original_item: Original full item (used for ID and structure)
        predicted_properties: Predicted properties (from GNN)
    
    Returns:
        Minimal item with only size/alignment properties
    """
    # Start with minimal structure - only ID and predicted properties
    merged = {
        'id': original_item.get('id')
    }
    
    # Add only size/alignment properties that were predicted
    for prop in SIZE_ALIGNMENT_PROPERTIES:
        if prop in predicted_properties:
            merged[prop] = predicted_properties[prop]
    
    # Recursively merge children (only size/alignment properties)
    # Always include children if they exist in predicted_properties OR original_item
    merged_children = []
    
    # Create map of predicted children by ID
    predicted_children_map = {}
    if 'children' in predicted_properties and predicted_properties['children']:
        predicted_children_map = {
            child.get('id'): child 
            for child in predicted_properties['children']
        }
    
    # Process all children from original item
    if 'children' in original_item and original_item['children']:
        for original_child in original_item['children']:
            child_id = original_child.get('id')
            if child_id in predicted_children_map:
                # Child has prediction - merge it
                merged_child = merge_predicted_properties(
                    original_child,
                    predicted_children_map[child_id]
                )
                merged_children.append(merged_child)
            elif original_child.get('position') == 'static':
                # Child is static but no prediction - create minimal item with just ID
                # This ensures children aren't lost
                merged_children.append({'id': child_id})
    
    if merged_children:
        merged['children'] = merged_children
    
    return merged


def merge_predicted_items(
    original_items: List[Dict[str, Any]],
    predicted_items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Create minimal items with ONLY size/alignment properties from predictions.
    Only processes static items; absolute items are excluded.
    
    Args:
        original_items: Original full items (used for ID and structure)
        predicted_items: Predicted items with size/alignment properties
    
    Returns:
        List of minimal items with only size/alignment properties
    """
    # Create map of predicted items by ID
    predicted_map = {}
    
    def build_predicted_map(items: List[Dict[str, Any]]):
        for item in items:
            item_id = item.get('id')
            if item_id:
                predicted_map[item_id] = item
            if 'children' in item and item['children']:
                build_predicted_map(item['children'])
    
    build_predicted_map(predicted_items)
    
    def create_minimal_item(original_item: Dict[str, Any]) -> Dict[str, Any] | None:
        """Create minimal item with only size/alignment properties for static items."""
        # Skip non-static items
        if original_item.get('position') != 'static':
            return None
        
        # Get prediction for this item
        item_id = original_item.get('id')
        if item_id and item_id in predicted_map:
            # Create minimal item with only size/alignment properties
            minimal_item = merge_predicted_properties(
                original_item,
                predicted_map[item_id]
            )
            
            # Recursively process children
            if 'children' in original_item and original_item['children']:
                minimal_children = []
                for child in original_item['children']:
                    minimal_child = create_minimal_item(child)
                    if minimal_child is not None:
                        minimal_children.append(minimal_child)
                if minimal_children:
                    minimal_item['children'] = minimal_children
            
            return minimal_item
        
        return None
    
    # Create minimal items for each original static item
    merged_items = []
    for original_item in original_items:
        minimal_item = create_minimal_item(original_item)
        if minimal_item is not None:
            merged_items.append(minimal_item)
    
    return merged_items


def get_resolution_dimensions(resolution_label: str) -> tuple[int, int]:
    """
    Extract width and height from resolution label.
    
    Args:
        resolution_label: e.g., "Desktop (1920x1080)", "Mobile (375x667)"
    
    Returns:
        Tuple of (width, height)
    """
    try:
        # Extract dimensions from label like "Desktop (1920x1080)"
        import re
        match = re.search(r'\((\d+)x(\d+)\)', resolution_label)
        if match:
            return int(match.group(1)), int(match.group(2))
    except:
        pass
    
    # Default fallback dimensions
    defaults = {
        'Desktop': (1920, 1080),
        'Laptop': (1366, 768),
        'Tablet': (768, 1024),
        'iPad Pro': (1024, 1366),
        'iPhone 14 Pro': (393, 852),
        'Pixel 7': (412, 915),
        'Galaxy S22': (360, 780),
        'Mobile': (375, 667)
    }
    
    for key, dims in defaults.items():
        if key in resolution_label:
            return dims
    
    return (1920, 1080)  # Default desktop

