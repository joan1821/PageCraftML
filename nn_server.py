# nn_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List, Optional, Tuple, Union
import uvicorn
from datetime import datetime
import torch
import os

# Import shared types
from item_types import Item, GalleryImage

# Import GNN components (after types to avoid circular import)
from graph_utils import build_graph_from_items
from gnn_model import ResponsiveGNN, predict_responsive_behavior
from responsive_predictor import predict_and_apply_responsive

app = FastAPI(title="PageCraft NN Proxy Server", version="1.0.0")

class SavedWork(BaseModel):
    model_config = ConfigDict(extra="allow")  # Allow extra fields but we'll clean them
    
    version: int = 1
    savedAt: str
    itemsByResolution: Dict[str, List[Any]]  # Allow dicts for mobile (minimal response)
    gallery: List[GalleryImage]

class NNRequest(BaseModel):
    payload: SavedWork

class NNResponse(BaseModel):
    processedPayload: SavedWork


# Initialize GNN model
_model: Optional[ResponsiveGNN] = None


def get_model() -> ResponsiveGNN:
    """Lazy load the GNN model."""
    global _model
    if _model is None:
        _model = ResponsiveGNN(
            input_dim=20,
            hidden_dim=128,
            output_dim=8,
            num_layers=3,
            dropout=0.2
        )
        # Load pretrained weights if available
        model_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")
        if os.path.exists(model_path):
            _model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded pretrained model from {model_path}")
        else:
            print("No pretrained model found. Using randomly initialized weights.")
            print("Note: For production use, train the model on a dataset of desktop-to-mobile conversions.")
        _model.eval()
    return _model


def extract_resolution_dimensions(resolution_label: str) -> Tuple[float, float]:
    """
    Extract width and height from resolution label like "Desktop (1920x1080)" or "Mobile (375x667)".
    Returns default desktop dimensions if parsing fails.
    """
    try:
        # Look for pattern like "(1920x1080)"
        import re
        match = re.search(r'\((\d+)x(\d+)\)', resolution_label)
        if match:
            return float(match.group(1)), float(match.group(2))
    except:
        pass
    # Default to desktop dimensions
    return 1920.0, 1080.0


@app.post("/process", response_model=NNResponse)
async def process_nn(request: NNRequest):
    """
    Processes the payload using Graph Neural Network to predict optimal responsive resize behavior.
    
    The GNN analyzes the desktop layout structure and predicts:
    - Optimal breakpoints for responsive design
    - Fluid scaling factors
    - Layout adaptations (stacking, repositioning, hiding)
    - Media query values
    
    Returns the payload with optimized mobile/responsive layouts.
    """
    import time
    start_time = time.time()
    
    # Start with the input payload
    processed_payload = request.payload.model_copy(deep=True)
    
    # Get the GNN model (this may take 2-5 seconds on first call due to initialization)
    model_load_start = time.time()
    model = get_model()
    model_load_time = time.time() - model_load_start
    if model_load_time > 0.1:
        print(f"Model loaded in {model_load_time:.2f}s")
    
    # Find desktop resolution (typically the largest or first one)
    desktop_resolution = None
    desktop_items = None
    desktop_width, desktop_height = 1920.0, 1080.0
    
    print(f"[SERVER] Received payload with {len(processed_payload.itemsByResolution)} resolutions")
    for resolution, items in processed_payload.itemsByResolution.items():
        width, height = extract_resolution_dimensions(resolution)
        print(f"[SERVER] Resolution: {resolution} -> {width}x{height} ({len(items)} items)")
        if width >= 1920 or desktop_resolution is None:
            desktop_resolution = resolution
            desktop_items = items
            desktop_width, desktop_height = width, height
    
    if desktop_items is None or len(desktop_items) == 0:
        print(f"[SERVER] No desktop layout found. Returning payload unchanged.")
        return NNResponse(processedPayload=processed_payload)
    
    print(f"[SERVER] Using desktop: {desktop_resolution} ({desktop_width}x{desktop_height}) with {len(desktop_items)} items")
    
    # Target mobile dimensions
    mobile_width, mobile_height = 375.0, 667.0
    
    # Check if mobile resolution already exists
    mobile_resolution_label = f"Mobile ({int(mobile_width)}x{int(mobile_height)})"
    
    try:
        # Predict and apply responsive behavior
        predict_start = time.time()
        optimized_mobile_items = predict_and_apply_responsive(
            desktop_items,
            model,
            target_width=mobile_width,
            target_height=mobile_height,
            source_width=desktop_width,
            source_height=desktop_height
        )
        predict_time = time.time() - predict_start
        
        # Add or update mobile resolution (optimized_mobile_items is List[Dict])
        processed_payload.itemsByResolution[mobile_resolution_label] = optimized_mobile_items
        
        total_time = time.time() - start_time
        print(f"[SERVER] Processed payload at {datetime.now().isoformat()}:")
        print(f"[SERVER]   - Desktop: {desktop_resolution} ({len(desktop_items)} items)")
        print(f"[SERVER]   - Mobile: {mobile_resolution_label} ({len(optimized_mobile_items)} items)")
        if len(optimized_mobile_items) > 0:
            first_item = optimized_mobile_items[0]
            item_id = first_item.get('id', 'unknown') if isinstance(first_item, dict) else getattr(first_item, 'id', 'unknown')
            item_width = first_item.get('width', 0) if isinstance(first_item, dict) else getattr(first_item, 'width', 0)
            item_position = first_item.get('position', 'N/A') if isinstance(first_item, dict) else getattr(first_item, 'position', 'N/A')
            print(f"[SERVER]   - First item: {item_id}, width: {item_width:.1f}, position: {item_position}")
        print(f"[SERVER]   - Prediction time: {predict_time:.3f}s")
        print(f"[SERVER]   - Total time: {total_time:.3f}s")
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"Error during GNN processing (after {total_time:.3f}s): {e}")
        import traceback
        traceback.print_exc()
        # Return payload unchanged on error
        pass
    
    return NNResponse(processedPayload=processed_payload)

@app.get("/")
async def root():
    return {"message": "PageCraft NN Server is running! POST to /process for NN proxy."}

@app.get("/health")
async def health():
    """Health check endpoint to verify server is running and model is loaded."""
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "message": "Server is ready to process requests"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Server has errors"
        }

if __name__ == "__main__":
    print("=" * 60)
    print("Starting PageCraft NN Server...")
    print("=" * 60)
    try:
        # Test imports on startup
        print("Checking dependencies...")
        import torch
        import torch_geometric
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ PyTorch Geometric available")
        print("✓ All dependencies loaded successfully")
        print("=" * 60)
        print("Server starting on http://0.0.0.0:8000")
        print("Health check: http://localhost:8000/health")
        print("=" * 60)
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        exit(1)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)