"""
Training script for LayoutGNN model.
Learns to predict layout sizes and alignments from desktop to other resolutions.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from gnn_model import LayoutGNN, build_graph_from_items
from data_processor import clean_items_for_prediction, get_resolution_dimensions
from typing import Dict, List, Any, Tuple
import json
import os
from pathlib import Path


class LayoutDataset(Dataset):
    """
    Dataset for training the GNN model.
    Each sample contains:
    - Desktop items (input)
    - Target resolution items with only size/alignment properties (target)
    """
    
    def __init__(self, examples: List[Dict[str, Any]], desktop_width: int = 1920, desktop_height: int = 1080):
        """
        Args:
            examples: List of training examples, each containing:
                {
                    'desktop': [...],  # Desktop items
                    'target_resolution': 'iPhone 14 Pro (393x852)',
                    'target': [...]  # Target items with only size/alignment properties
                }
            desktop_width: Desktop canvas width
            desktop_height: Desktop canvas height
        """
        self.examples = examples
        self.desktop_width = desktop_width
        self.desktop_height = desktop_height
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Clean desktop items
        desktop_items = clean_items_for_prediction(example['desktop'])
        
        # Build graph from desktop
        graph = build_graph_from_items(desktop_items, self.desktop_width, self.desktop_height)
        
        # Get target resolution dimensions
        target_resolution = example['target_resolution']
        target_width, target_height = get_resolution_dimensions(target_resolution)
        
        # Extract target properties (size/alignment only)
        target_items = example['target']
        
        return {
            'graph': graph,
            'target_width': target_width,
            'target_height': target_height,
            'target_items': target_items,
            'desktop_items': desktop_items
        }


def extract_target_properties(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only size/alignment properties from an item.
    This is what we want the model to predict.
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


def create_training_example(desktop_items: List[Dict[str, Any]], 
                           target_resolution: str,
                           target_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a training example from desktop and target resolution data.
    
    Args:
        desktop_items: Full desktop items
        target_resolution: Target resolution label (e.g., "iPhone 14 Pro (393x852)")
        target_items: Target items (can be full items, will extract size/alignment only)
    
    Returns:
        Training example dict
    """
    # Extract only size/alignment properties from target
    target_properties = [extract_target_properties(item) for item in target_items]
    
    return {
        'desktop': desktop_items,
        'target_resolution': target_resolution,
        'target': target_properties
    }


def load_training_data(data_dir: str = "training_data") -> List[Dict[str, Any]]:
    """
    Load training examples from JSON files.
    
    Expected structure:
    training_data/
        example_1.json
        example_2.json
        ...
    
    Each JSON file should contain:
    {
        "desktop": [...],  # Desktop items
        "targets": {
            "iPhone 14 Pro (393x852)": [...],  # Target items
            "Tablet (768x1024)": [...],
            ...
        }
    }
    """
    examples = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Training data directory {data_dir} does not exist. Creating example structure...")
        data_path.mkdir(exist_ok=True)
        create_example_training_file(data_path / "example_1.json")
        return []
    
    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            desktop_items = data.get('desktop', [])
            targets = data.get('targets', {})
            
            # Create an example for each target resolution
            for resolution, target_items in targets.items():
                example = create_training_example(desktop_items, resolution, target_items)
                examples.append(example)
        
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    print(f"Loaded {len(examples)} training examples from {data_dir}")
    return examples


def create_example_training_file(file_path: Path):
    """Create an example training file structure."""
    example = {
        "desktop": [
            {
                "id": "box-1",
                "position": "static",
                "width": -1,
                "height": 80,
                "flexDirection": "row",
                "alignItems": "flex-start",
                "children": [
                    {
                        "id": "box-2",
                        "position": "static",
                        "width": -2,
                        "height": -1
                    },
                    {
                        "id": "box-3",
                        "position": "static",
                        "width": -2,
                        "height": -1
                    }
                ]
            }
        ],
        "targets": {
            "iPhone 14 Pro (393x852)": [
                {
                    "id": "box-1",
                    "width": 393,
                    "height": 63,
                    "flexDirection": "column",
                    "children": [
                        {
                            "id": "box-2",
                            "width": -1,
                            "height": -1
                        },
                        {
                            "id": "box-3",
                            "width": -1,
                            "height": -1
                        }
                    ]
                }
            ]
        }
    }
    
    with open(file_path, 'w') as f:
        json.dump(example, f, indent=2)
    
    print(f"Created example training file: {file_path}")


def compute_loss(predicted_items: List[Dict[str, Any]], 
                target_items: List[Dict[str, Any]]) -> torch.Tensor:
    """
    Compute loss between predicted and target items.
    Uses MSE for numeric properties and cross-entropy for categorical properties.
    """
    total_loss = 0.0
    count = 0
    
    def item_loss(pred_item: Dict[str, Any], target_item: Dict[str, Any]) -> Tuple[float, int]:
        """Compute loss for a single item recursively."""
        loss = 0.0
        item_count = 0
        
        # Match items by ID
        if pred_item.get('id') != target_item.get('id'):
            return 0.0, 0
        
        # Numeric properties (width, height, imageWidth, imageHeight)
        numeric_props = ['width', 'height', 'imageWidth', 'imageHeight']
        for prop in numeric_props:
            if prop in target_item:
                target_val = target_item[prop]
                pred_val = pred_item.get(prop, 0)
                
                # Handle special values (-1, -2) - treat as classes
                if target_val == -1:
                    if pred_val != -1:
                        loss += 1.0  # Penalty for wrong special value
                    item_count += 1
                elif target_val == -2:
                    if pred_val != -2:
                        loss += 1.0
                    item_count += 1
                elif isinstance(target_val, (int, float)) and target_val > 0:
                    # Regular numeric value - use MSE
                    if isinstance(pred_val, (int, float)):
                        diff = abs(pred_val - target_val)
                        # Normalize by target value
                        normalized_diff = diff / max(abs(target_val), 1)
                        loss += normalized_diff ** 2
                        item_count += 1
        
        # Categorical properties (alignItems, justifyContent, flexDirection, alignContent)
        categorical_props = ['alignItems', 'justifyContent', 'flexDirection', 'alignContent']
        for prop in categorical_props:
            if prop in target_item:
                target_val = target_item[prop]
                pred_val = pred_item.get(prop)
                if pred_val != target_val:
                    loss += 1.0  # Binary loss for categorical
                item_count += 1
        
        # Recursively process children
        pred_children = {c.get('id'): c for c in pred_item.get('children', [])}
        target_children = {c.get('id'): c for c in target_item.get('children', [])}
        
        for child_id, target_child in target_children.items():
            if child_id in pred_children:
                child_loss, child_count = item_loss(pred_children[child_id], target_child)
                loss += child_loss
                item_count += child_count
        
        return loss, item_count
    
    # Match items by ID and compute loss
    pred_map = {item.get('id'): item for item in predicted_items}
    target_map = {item.get('id'): item for item in target_items}
    
    for item_id, target_item in target_map.items():
        if item_id in pred_map:
            item_loss_val, item_count_val = item_loss(pred_map[item_id], target_item)
            total_loss += item_loss_val
            count += item_count_val
    
    if count == 0:
        return torch.tensor(0.0)
    
    return torch.tensor(total_loss / count)  # Average loss


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader that handles PyTorch Geometric Data objects.
    Since batch_size=1, we just return the single item.
    """
    if len(batch) == 1:
        return batch[0]
    # For batch_size > 1, would need to batch graphs properly
    return batch[0]


def encode_property_to_tensor(prop_value: Any, prop_name: str, target_width: int = None, target_height: int = None) -> float:
    """
    Encode a property value to a normalized float for training.
    Handles special values (-1, -2) and numeric/categorical properties.
    """
    if prop_name in ['width', 'height', 'imageWidth', 'imageHeight']:
        # Numeric properties - normalize to [0, 1]
        if prop_value == -1:
            # Full width/height - encode as 1.0 (normalized)
            return 1.0
        elif prop_value == -2:
            # Half width/height - encode as 0.5
            return 0.5
        elif isinstance(prop_value, (int, float)) and prop_value > 0:
            # Normalize by desktop dimensions (1920x1080)
            if prop_name in ['width', 'imageWidth']:
                return min(prop_value / 1920.0, 1.0)
            else:
                return min(prop_value / 1080.0, 1.0)
        else:
            return 0.0
    else:
        # Categorical properties - encode as normalized index
        if prop_name == 'flexDirection':
            mapping = {'row': 0, 'column': 1, 'row-reverse': 2, 'column-reverse': 3, None: 0}
        elif prop_name == 'justifyContent':
            mapping = {'flex-start': 0, 'flex-end': 1, 'center': 2, 'space-between': 3, 'space-around': 4, 'space-evenly': 5, None: 0}
        elif prop_name == 'alignItems':
            mapping = {'flex-start': 0, 'flex-end': 1, 'center': 2, 'stretch': 3, 'baseline': 4, None: 0}
        elif prop_name == 'alignContent':
            mapping = {'flex-start': 0, 'flex-end': 1, 'center': 2, 'space-between': 3, 'space-around': 4, 'stretch': 5, None: 0}
        else:
            return 0.0
        
        value = mapping.get(prop_value, 0)
        return float(value) / len(mapping)  # Normalize to [0, 1]


def train_epoch(model: LayoutGNN, dataloader: DataLoader, optimizer: optim.Optimizer, 
                device: torch.device) -> float:
    """Train for one epoch - learns from examples."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_data in dataloader:
        # Since batch_size=1, extract single example
        graph = batch_data['graph']
        if isinstance(graph, list):
            graph = graph[0]
        
        target_width = batch_data['target_width']
        target_height = batch_data['target_height']
        if hasattr(target_width, 'item'):
            target_width = target_width.item()
        if hasattr(target_height, 'item'):
            target_height = target_height.item()
        
        target_items = batch_data['target_items']
        desktop_items = batch_data['desktop_items']
        
        # custom_collate_fn returns the dict directly, so these should already be lists
        # But handle edge cases
        if not isinstance(target_items, list):
            if isinstance(target_items, (tuple,)):
                target_items = list(target_items)
            else:
                print(f"Skipping batch: target_items is not a list (type: {type(target_items)})")
                continue
        
        if not isinstance(desktop_items, list):
            if isinstance(desktop_items, (tuple,)):
                desktop_items = list(desktop_items)
            else:
                print(f"Skipping batch: desktop_items is not a list (type: {type(desktop_items)})")
                continue
        
        optimizer.zero_grad()
        
        # Move graph to device
        graph = graph.to(device)
        
        # Forward pass - get per-node predictions
        sizes_pred, alignments_pred = model(graph.x, graph.edge_index)
        
        # Build target tensors from target_items by matching with desktop items
        target_map = {}
        def build_target_map(items):
            for item in items:
                item_id = item.get('id')
                if item_id:
                    target_map[item_id] = item
                if 'children' in item and item['children']:
                    build_target_map(item['children'])
        
        build_target_map(target_items)
        
        # Create target tensors matching the order of nodes in graph
        # Nodes are created in the same order as items are processed
        size_targets = []
        alignment_targets = []
        
        def process_item_for_targets(item):
            """Process item and create target tensors."""
            item_id = item.get('id')
            target_item = target_map.get(item_id, {})
            
            # Size targets: [width, height, imageWidth, imageHeight]
            size_target = torch.tensor([
                encode_property_to_tensor(target_item.get('width', item.get('width', 0)), 'width', target_width),
                encode_property_to_tensor(target_item.get('height', item.get('height', 0)), 'height', target_height),
                encode_property_to_tensor(target_item.get('imageWidth', 0), 'imageWidth', target_width),
                encode_property_to_tensor(target_item.get('imageHeight', 0), 'imageHeight', target_height)
            ])
            size_targets.append(size_target)
            
            # Alignment targets: [alignItems, justifyContent, flexDirection, alignContent]
            alignment_target = torch.tensor([
                encode_property_to_tensor(target_item.get('alignItems'), 'alignItems'),
                encode_property_to_tensor(target_item.get('justifyContent'), 'justifyContent'),
                encode_property_to_tensor(target_item.get('flexDirection'), 'flexDirection'),
                encode_property_to_tensor(target_item.get('alignContent'), 'alignContent')
            ])
            alignment_targets.append(alignment_target)
            
            # Process children recursively (in same order as graph building)
            if 'children' in item and item['children']:
                for child in item['children']:
                    process_item_for_targets(child)
        
        # Process all desktop items in order
        for item in desktop_items:
            process_item_for_targets(item)
        
        # Convert to tensors and move to device
        if size_targets and len(size_targets) == sizes_pred.shape[0]:
            size_target_tensor = torch.stack(size_targets).to(device)
            alignment_target_tensor = torch.stack(alignment_targets).to(device)
            
            # Compute losses
            size_loss = nn.MSELoss()(sizes_pred, size_target_tensor)
            alignment_loss = nn.MSELoss()(alignments_pred, alignment_target_tensor)
            loss = size_loss + alignment_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def train_model(data_dir: str = "training_data", 
                epochs: int = 100,
                batch_size: int = 1,
                learning_rate: float = 0.001,
                save_path: str = "model_checkpoint.pth"):
    """
    Train the LayoutGNN model on examples.
    
    Args:
        data_dir: Directory containing training JSON files
        epochs: Number of training epochs
        batch_size: Batch size (currently 1 due to variable graph sizes)
        learning_rate: Learning rate for optimizer
        save_path: Path to save trained model
    """
    # Load training data
    examples = load_training_data(data_dir)
    
    if len(examples) == 0:
        print("No training examples found. Please add training data to the training_data directory.")
        print("See create_example_training_file() for the expected format.")
        return
    
    # Create dataset
    dataset = LayoutDataset(examples)
    # Use custom collate function to handle PyTorch Geometric Data objects
    # batch_size must be 1 for now
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LayoutGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on {len(examples)} examples for {epochs} epochs...")
    print(f"Using device: {device}")
    
    # Training loop
    for epoch in range(epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LayoutGNN model')
    parser.add_argument('--data_dir', type=str, default='training_data',
                       help='Directory containing training JSON files')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save_path', type=str, default='model_checkpoint.pth',
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=args.save_path
    )

