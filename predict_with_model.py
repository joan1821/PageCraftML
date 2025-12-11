"""
Prediction function that uses the trained GNN model instead of rule-based logic.
"""
import torch
from gnn_model import LayoutGNN, build_graph_from_items
from data_processor import clean_items_for_prediction, get_resolution_dimensions
from typing import Dict, List, Any
import os


def predict_with_trained_model(
    model: LayoutGNN,
    desktop_items: List[Dict[str, Any]],
    target_width: int,
    target_height: int,
    desktop_width: int = 1920,
    desktop_height: int = 1080,
    device: torch.device = None
) -> List[Dict[str, Any]]:
    """
    Predict properties using the trained GNN model.
    
    Args:
        model: Trained LayoutGNN model
        desktop_items: Desktop items (cleaned, static only)
        target_width: Target resolution width
        target_height: Target resolution height
        desktop_width: Desktop canvas width
        desktop_height: Desktop canvas height
        device: PyTorch device
    
    Returns:
        List of predicted items with size/alignment properties
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Build graph from desktop items
    graph = build_graph_from_items(desktop_items, desktop_width, desktop_height)
    
    if graph.x.shape[0] == 0:
        return []
    
    # Move graph to device
    graph = graph.to(device)
    
    # Get model predictions
    with torch.no_grad():
        sizes, alignments = model(graph.x, graph.edge_index)
    
    # Convert model outputs to item properties
    # Note: This is a simplified version - full implementation would need
    # to map node-level predictions back to items
    
    # For now, use rule-based as fallback until model is fully trained
    # In production, you'd decode the model outputs properly
    return predict_with_rules_fallback(desktop_items, target_width, target_height, 
                                      desktop_width, desktop_height)


def predict_with_rules_fallback(
    desktop_items: List[Dict[str, Any]],
    target_width: int,
    target_height: int,
    desktop_width: int = 1920,
    desktop_height: int = 1080
) -> List[Dict[str, Any]]:
    """
    Fallback rule-based prediction (current implementation).
    This will be replaced by model predictions once training is complete.
    """
    # Import the current prediction function
    from gnn_model import predict_resolution_properties
    from gnn_model import LayoutGNN
    
    # Create a dummy model (won't be used in rule-based mode)
    dummy_model = LayoutGNN()
    
    return predict_resolution_properties(
        dummy_model, desktop_items, target_width, target_height,
        desktop_width, desktop_height
    )

