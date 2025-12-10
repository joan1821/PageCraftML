# nn_server.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List, Optional
import uvicorn
from datetime import datetime
import json

app = FastAPI(title="PageCraft NN Proxy Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
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



@app.post("/process")
async def process_nn(request: Request):
    """Processes layout data using PageCraftML GNN. Accepts multiple request structures."""
    import copy
    
    # Parse JSON body
    try:
        raw_body = await request.body()
        body = json.loads(raw_body.decode('utf-8'))
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
    
    # Process the payload
    processed_payload = copy.deepcopy(payload)
    
    try:
        # Color all items in all resolutions green (placeholder ML processing)
        for resolution, items in processed_payload['itemsByResolution'].items():
            colored_items = []
            for item in items:
                # Convert Item object to dict if needed
                if hasattr(item, 'model_dump'):
                    item_dict = item.model_dump()
                elif hasattr(item, 'dict'):
                    item_dict = item.dict()
                else:
                    item_dict = dict(item) if hasattr(item, '__dict__') else item

                # Add green color and ensure basic properties
                item_dict['color'] = '#00FF00'
                item_dict['position'] = item_dict.get('position', 'absolute')
                item_dict['x'] = item_dict.get('x', 0)
                item_dict['y'] = item_dict.get('y', 0)
                item_dict['width'] = item_dict.get('width', 100)
                item_dict['height'] = item_dict.get('height', 50)

                colored_items.append(item_dict)

            processed_payload['itemsByResolution'][resolution] = colored_items

    except Exception as e:
        # Return payload unchanged on error
        pass
    
    return {"processedPayload": processed_payload}

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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)