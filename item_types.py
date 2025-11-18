"""
Shared type definitions for the GNN server.
"""
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, ForwardRef

ItemRef = ForwardRef('Item')

class Item(BaseModel):
    model_config = ConfigDict(extra="allow")  # Allow extra fields that might be in the JSON
    
    id: str
    position: Optional[str] = "absolute"  # "static" or "absolute"
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
    # Image properties (for scaling images)
    imageEnabled: Optional[bool] = None
    imageSrc: Optional[str] = None
    imageWidth: Optional[float] = None
    imageHeight: Optional[float] = None
    imageFit: Optional[str] = None  # 'cover' | 'contain'
    children: Optional[List[ItemRef]] = None

Item.model_rebuild()

class GalleryImage(BaseModel):
    id: str
    name: str
    mimeType: str
    dataBase64: str
    width: Optional[int] = None
    height: Optional[int] = None

