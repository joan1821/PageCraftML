# nn_server.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List, Optional, Tuple
import uvicorn
from datetime import datetime
import json
import copy
import sys
import os

# Import GNN model and data processing utilities
from gnn_model import LayoutGNN, predict_resolution_properties
from data_processor import (
    clean_items_for_prediction,
    merge_predicted_items,
    get_resolution_dimensions
)
import torch

app = FastAPI(title="PageCraft NN Proxy Server", version="1.0.0")

# Backend server configuration
BACKEND_SERVER_URL = os.environ.get("BACKEND_SERVER_URL", "https://pagecraftserver-production-d2dd.up.railway.app")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        BACKEND_SERVER_URL,
        "http://localhost:3000",  # Local development
        "http://localhost:5000",
        "*"  # Allow all for development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models (not used for validation, kept for reference)
class GalleryImage(BaseModel):
    model_config = ConfigDict(extra='allow')
    id: Optional[str] = None
    name: Optional[str] = None
    mimeType: Optional[str] = None
    dataBase64: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

class SavedWork(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: Optional[int] = 1
    savedAt: Optional[str] = None
    itemsByResolution: Dict[str, Any]
    gallery: Optional[List[Any]] = []


# Initialize GNN model - load trained model if available, otherwise use untrained
_model = None
_model_path = "model_checkpoint.pth"
_model_trained = False

def get_model() -> Tuple[LayoutGNN, bool]:
    """Get or create the GNN model instance. Loads trained weights if available.
    
    Returns:
        Tuple of (model, is_trained)
    """
    global _model, _model_trained
    if _model is None:
        _model = LayoutGNN()
        
        # Try to load trained model weights
        if os.path.exists(_model_path):
            try:
                _model.load_state_dict(torch.load(_model_path, map_location='cpu'))
                _model_trained = True
                print(f"Loaded trained model from {_model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {_model_path}: {e}")
                print("Using untrained model (rule-based predictions)")
                _model_trained = False
        else:
            print(f"No trained model found at {_model_path}. Using rule-based predictions.")
            print("Train a model using: python train_model.py")
            _model_trained = False
        
        _model.eval()
    return _model, _model_trained


def find_desktop_resolution(items_by_resolution: Dict[str, Any]) -> Optional[str]:
    """
    Find the desktop resolution key.
    Desktop is typically the one with the most items or contains 'Desktop' in name.
    """
    desktop_candidates = []
    
    for resolution, items in items_by_resolution.items():
        if not isinstance(items, list):
            continue
        
        # Check if it's explicitly desktop
        if 'Desktop' in resolution or 'desktop' in resolution.lower():
            if items:  # Has items
                return resolution
            desktop_candidates.append((resolution, len(items)))
        elif items:  # Has items but not explicitly desktop
            desktop_candidates.append((resolution, len(items)))
    
    # Return the resolution with the most items
    if desktop_candidates:
        desktop_candidates.sort(key=lambda x: x[1], reverse=True)
        return desktop_candidates[0][0]
    
    return None


@app.post("/process")
async def process_nn(request: Request):
    """Processes layout data using PageCraftML GNN. Accepts multiple request structures."""
    
    # Parse JSON body
    try:
        raw_body = await request.body()
        body = json.loads(raw_body.decode('utf-8'))
        
        # Debug: Check raw body structure
        print(f"DEBUG: Raw body type: {type(body)}", flush=True)
        if isinstance(body, dict):
            print(f"DEBUG: Body keys: {list(body.keys())}", flush=True)
            if 'pages' in body:
                print(f"DEBUG: Pages structure: {type(body.get('pages'))}", flush=True)
                if isinstance(body.get('pages'), list) and len(body['pages']) > 0:
                    first_page = body['pages'][0]
                    print(f"DEBUG: First page keys: {list(first_page.keys()) if isinstance(first_page, dict) else 'N/A'}", flush=True)
                    if isinstance(first_page, dict) and 'data' in first_page:
                        data = first_page['data']
                        print(f"DEBUG: Data keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}", flush=True)
                        if isinstance(data, dict) and 'itemsByResolution' in data:
                            print(f"DEBUG: itemsByResolution in data has {len(data['itemsByResolution'])} keys: {list(data['itemsByResolution'].keys())}", flush=True)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid JSON: {str(e)}"}
        )
    
    # Extract payload from different structures:
    # 1. { payload: SavedWork } - from Node.js server
    # 2. SavedWork directly - direct SavedWork structure
    # 3. { restaurant: {...}, pages: [{ page: {...}, data: SavedWork }] } - raw structure
    payload = None
    
    if isinstance(body, dict):
        if 'payload' in body:
            payload = body['payload']
        elif 'itemsByResolution' in body:
            payload = body
        elif 'pages' in body and isinstance(body.get('pages'), list) and len(body['pages']) > 0:
            first_page = body['pages'][0]
            if isinstance(first_page, dict) and 'data' in first_page:
                payload = first_page['data']
    
    if payload is None:
        payload = body
    
    # Ensure it's a dict with required fields
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "Payload must be an object"}
        )
    
    # Add defaults
    if 'version' not in payload:
        payload['version'] = 1
    if 'savedAt' not in payload:
        payload['savedAt'] = datetime.now().isoformat()
    if 'itemsByResolution' not in payload:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'itemsByResolution' field"}
        )
    if 'gallery' not in payload:
        payload['gallery'] = []
    
    if not isinstance(payload['itemsByResolution'], dict):
        return JSONResponse(
            status_code=400,
            content={"error": "'itemsByResolution' must be an object"}
        )
    
    # Process the payload using GNN
    processed_payload = copy.deepcopy(payload)
    
    # Debug: Check what we received
    print(f"DEBUG: Payload keys: {list(payload.keys())}", flush=True)
    if 'itemsByResolution' in payload:
        print(f"DEBUG: itemsByResolution keys: {list(payload['itemsByResolution'].keys())}", flush=True)
        print(f"DEBUG: itemsByResolution values lengths: {[(k, len(v) if isinstance(v, list) else 'N/A') for k, v in payload['itemsByResolution'].items()]}", flush=True)
    
    try:
        items_by_resolution = processed_payload['itemsByResolution']
        print(f"DEBUG: After copy, items_by_resolution keys: {list(items_by_resolution.keys())}", flush=True)
        
        # Step 1: Find desktop resolution (the main version with items)
        desktop_resolution = find_desktop_resolution(items_by_resolution)
        
        if not desktop_resolution:
            # No desktop found, return unchanged
            return {"processedPayload": processed_payload}
        
        desktop_items = items_by_resolution.get(desktop_resolution, [])
        
        if not desktop_items:
            # Desktop has no items, return unchanged
            return {"processedPayload": processed_payload}
        
        # Convert items to dicts if needed
        desktop_items_dicts = []
        for item in desktop_items:
            if hasattr(item, 'model_dump'):
                item_dict = item.model_dump()
            elif hasattr(item, 'dict'):
                item_dict = item.dict()
            elif isinstance(item, dict):
                item_dict = item
            else:
                item_dict = dict(item) if hasattr(item, '__dict__') else item
            desktop_items_dicts.append(item_dict)
        
        # Step 2: Clean desktop items - extract only size/alignment relevant properties
        cleaned_desktop_items = clean_items_for_prediction(desktop_items_dicts)
        
        # Check if we have any static items after cleaning
        if not cleaned_desktop_items:
            print(f"Warning: No static items found after filtering. Desktop had {len(desktop_items_dicts)} items.")
            return {"processedPayload": processed_payload}
        
        print(f"Processing {len(cleaned_desktop_items)} static items for desktop resolution: {desktop_resolution}")
        
        # Get desktop dimensions
        desktop_width, desktop_height = get_resolution_dimensions(desktop_resolution)
        
        # Step 3: Process with GNN for each non-desktop resolution
        model, model_is_trained = get_model()
        
        # Standard resolutions to predict for (if not already present)
        standard_resolutions = [
            "Laptop (1366x768)",
            "Tablet (768x1024)",
            "iPad Pro (1024x1366)",
            "iPhone 14 Pro (393x852)",
            "Pixel 7 (412x915)",
            "Galaxy S22 (360x780)",
            "Mobile (375x667)"
        ]
        
        # Ensure all standard resolutions exist in items_by_resolution (initialize as empty if missing)
        for res in standard_resolutions:
            if res not in items_by_resolution:
                items_by_resolution[res] = []
                processed_payload['itemsByResolution'][res] = []
        
        resolutions_processed = 0
        print(f"Total resolutions to check: {len(items_by_resolution)}", flush=True)
        print(f"Resolution keys: {list(items_by_resolution.keys())}", flush=True)
        
        for resolution, items in items_by_resolution.items():
            # Skip desktop resolution (it's our source)
            if resolution == desktop_resolution:
                print(f"Skipping desktop resolution: {resolution}")
                continue
            
            # Debug: Check what we're getting
            print(f"Checking resolution: '{resolution}'", flush=True)
            print(f"  - Type: {type(items)}", flush=True)
            print(f"  - Is list: {isinstance(items, list)}", flush=True)
            if isinstance(items, list):
                print(f"  - Length: {len(items)}", flush=True)
                print(f"  - Value: {items}", flush=True)
            else:
                print(f"  - Value: {items}", flush=True)
            
            # Only process if resolution is empty or has minimal items
            if isinstance(items, list) and len(items) == 0:
                # Get target resolution dimensions
                target_width, target_height = get_resolution_dimensions(resolution)
                
                print(f"Predicting for resolution: {resolution} ({target_width}x{target_height})")
                
                # Step 4: Predict properties for this resolution using GNN
                predicted_items = predict_resolution_properties(
                    model=model,
                    desktop_items=cleaned_desktop_items,
                    target_width=target_width,
                    target_height=target_height,
                    desktop_width=desktop_width,
                    desktop_height=desktop_height,
                    use_model_predictions=model_is_trained  # Use model if trained, otherwise rules
                )
                
                if not predicted_items:
                    print(f"Warning: No predicted items returned for {resolution}")
                    continue
                
                print(f"DEBUG: Got {len(predicted_items)} predicted items for {resolution}")
                if predicted_items:
                    print(f"DEBUG: First predicted item keys: {list(predicted_items[0].keys())}")
                
                # Step 5: Create minimal items with only size/alignment properties
                merged_items = merge_predicted_items(
                    original_items=desktop_items_dicts,
                    predicted_items=predicted_items
                )
                
                print(f"DEBUG: Merged items count: {len(merged_items)}")
                if merged_items:
                    print(f"DEBUG: First merged item: {merged_items[0]}")
                
                # Update the resolution with merged items
                # Update both references to ensure consistency
                items_by_resolution[resolution] = merged_items
                processed_payload['itemsByResolution'][resolution] = merged_items
                
                # Verify the update worked
                verify_items = processed_payload['itemsByResolution'].get(resolution, [])
                print(f"VERIFY: Updated {resolution} - stored {len(verify_items)} items", flush=True)
                
                resolutions_processed += 1
                print(f"Successfully processed {resolution}: {len(merged_items)} items")
        
        print(f"Processed {resolutions_processed} resolutions")
        
        # Debug: Check final output
        print(f"DEBUG: Final processed_payload['itemsByResolution'] keys: {list(processed_payload['itemsByResolution'].keys())}")
        for res, items in processed_payload['itemsByResolution'].items():
            if res != desktop_resolution:
                print(f"DEBUG: {res} has {len(items) if isinstance(items, list) else 'N/A'} items")
                if isinstance(items, list) and len(items) > 0:
                    print(f"DEBUG: First item in {res}: {items[0]}")
        
        if resolutions_processed == 0:
            print("Warning: No resolutions were processed. Check if resolutions are empty arrays.")
            print(f"Debug: items_by_resolution contents: {list(items_by_resolution.keys())}")
            for res, items in items_by_resolution.items():
                if res != desktop_resolution:
                    print(f"  {res}: {type(items)} = {items}")
        
    except Exception as e:
        # Log error but return payload unchanged
        import traceback
        error_msg = f"Error in GNN processing: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        # Return error information in response for debugging
        return JSONResponse(
            status_code=500,
            content={
                "error": error_msg,
                "processedPayload": processed_payload,
                "traceback": traceback.format_exc()
            }
        )
    
    # Final verification - ensure items are actually in the response
    final_response = {"processedPayload": processed_payload}
    
    # Verify items are in response
    if 'itemsByResolution' in processed_payload:
        for res, items in processed_payload['itemsByResolution'].items():
            if res != desktop_resolution:
                item_count = len(items) if isinstance(items, list) else 0
                if item_count > 0:
                    print(f"FINAL CHECK: {res} has {item_count} items in response", flush=True)
                else:
                    print(f"FINAL CHECK: WARNING - {res} is EMPTY in response!", flush=True)
    
    return final_response

@app.get("/")
async def root():
    return {"message": "PageCraft NN Server is running! POST to /process for NN proxy."}

@app.get("/health")
async def health():
    """Health check endpoint to verify server is running."""
    return {
        "status": "healthy",
        "message": "Server is ready to process requests"
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)