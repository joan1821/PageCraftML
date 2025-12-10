# nn_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, ForwardRef, Tuple
import uvicorn
from datetime import datetime
import torch
import os

# Import GNN components
from graph_utils import build_graph_from_items
from gnn_model import ResponsiveGNN, predict_responsive_behavior
from responsive_predictor import predict_and_apply_responsive

app = FastAPI(title="PageCraft NN Proxy Server", version="1.0.0")

ItemRef = ForwardRef('Item')

class Item(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float
    rotation: float
    borderRadius: float
    color: str
    opacity: float
    padding: float
    paddingTop: float
    paddingRight: float
    paddingBottom: float
    paddingLeft: float
    margin: float
    marginTop: float
    marginRight: float
    marginBottom: float
    marginLeft: float
    boxShadowOffsetX: float
    boxShadowOffsetY: float
    boxShadowBlur: float
    boxShadowSpread: float
    boxShadowBaseColor: str
    boxShadowOpacity: float
    boxShadowUsesBoxOpacity: bool
    borderWidth: float
    borderColor: str
    borderStyle: str
    backgroundGradientEnabled: bool
    backgroundGradientStart: str
    backgroundGradientEnd: str
    backgroundGradientAngle: float
    filterBlur: float
    filterBrightness: float
    filterContrast: float
    zIndex: float
    # Optional extras
    backgroundImageEnabled: Optional[bool] = None
    backgroundImageSrc: Optional[str] = None
    backgroundImageSourceType: Optional[str] = None  # 'url' | 'gallery'
    backgroundImageGalleryId: Optional[str] = None
    backgroundImageFit: Optional[str] = None  # 'cover' | 'contain'
    backgroundImagePosition: Optional[str] = None  # 'center' | 'top' | etc.
    backgroundImageRepeat: Optional[str] = None  # 'no-repeat' | etc.
    children: Optional[List[ItemRef]] = None

Item.model_rebuild()

class GalleryImage(BaseModel):
    id: str
    name: str
    mimeType: str
    dataBase64: str
    width: Optional[int] = None
    height: Optional[int] = None

class SavedWork(BaseModel):
    version: int = 1
    savedAt: str
    itemsByResolution: Dict[str, List[Item]]
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
    # Start with the input payload
    processed_payload = request.payload.model_copy(deep=True)
    
    # Get the GNN model
    model = get_model()
    
    # Find desktop resolution (typically the largest or first one)
    desktop_resolution = None
    desktop_items = None
    desktop_width, desktop_height = 1920.0, 1080.0
    
    for resolution, items in processed_payload.itemsByResolution.items():
        width, height = extract_resolution_dimensions(resolution)
        if width >= 1920 or desktop_resolution is None:
            desktop_resolution = resolution
            desktop_items = items
            desktop_width, desktop_height = width, height
    
    if desktop_items is None or len(desktop_items) == 0:
        print(f"No desktop layout found. Returning payload unchanged.")
        return NNResponse(processedPayload=processed_payload)
    
    # Target mobile dimensions
    mobile_width, mobile_height = 375.0, 667.0
    
    # Check if mobile resolution already exists
    mobile_resolution_label = f"Mobile ({int(mobile_width)}x{int(mobile_height)})"
    
    try:
        # Predict and apply responsive behavior
        optimized_mobile_items = predict_and_apply_responsive(
            desktop_items,
            model,
            target_width=mobile_width,
            target_height=mobile_height,
            source_width=desktop_width,
            source_height=desktop_height
        )
        
        # Add or update mobile resolution
        processed_payload.itemsByResolution[mobile_resolution_label] = optimized_mobile_items
        
        print(f"Processed payload at {datetime.now().isoformat()}:")
        print(f"  - Desktop: {desktop_resolution} ({len(desktop_items)} items)")
        print(f"  - Mobile: {mobile_resolution_label} ({len(optimized_mobile_items)} items)")
        print(f"  - Applied GNN predictions for responsive behavior")
        
    except Exception as e:
        print(f"Error during GNN processing: {e}")
        import traceback
        traceback.print_exc()
        # Return payload unchanged on error
        pass
    
    return NNResponse(processedPayload=processed_payload)

@app.get("/")
async def root():
    return {"message": "PageCraft NN Server is running! POST to /process for NN proxy."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)