"""
Extract training examples from existing JSON templates.
Converts templates with desktop + target resolutions into training data.
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from data_processor import extract_target_properties


def extract_examples_from_template(template_path: str, output_dir: str = "training_data"):
    """
    Extract training examples from a template JSON file.
    
    Args:
        template_path: Path to template JSON (e.g., ChineseRestaurant.json)
        output_dir: Directory to save training examples
    """
    with open(template_path, 'r') as f:
        data = json.load(f)
    
    # Extract pages data
    pages = data.get('pages', [])
    if not pages:
        print("No pages found in template")
        return
    
    examples = []
    
    for page in pages:
        page_data = page.get('data', {})
        items_by_resolution = page_data.get('itemsByResolution', {})
        
        # Find desktop resolution
        desktop_items = None
        desktop_resolution = None
        
        for resolution, items in items_by_resolution.items():
            if 'Desktop' in resolution and items:
                desktop_items = items
                desktop_resolution = resolution
                break
        
        if not desktop_items:
            print(f"No desktop items found in page {page.get('page', {}).get('id')}")
            continue
        
        # Create examples for each target resolution
        for resolution, target_items in items_by_resolution.items():
            if resolution == desktop_resolution:
                continue
            
            # Only use resolutions that have items (not empty)
            if target_items and len(target_items) > 0:
                # Extract only size/alignment properties from target
                target_properties = []
                for item in target_items:
                    target_properties.append(extract_target_properties(item))
                
                example = {
                    'desktop': desktop_items,
                    'target_resolution': resolution,
                    'target': target_properties
                }
                examples.append(example)
    
    # Save examples
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as separate files or one combined file
    if examples:
        # Save as one file per page
        for i, example in enumerate(examples):
            filename = f"example_{i+1}.json"
            filepath = output_path / filename
            
            # Group by target resolution for easier organization
            example_data = {
                'desktop': example['desktop'],
                'targets': {
                    example['target_resolution']: example['target']
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(example_data, f, indent=2)
        
        print(f"Extracted {len(examples)} training examples to {output_dir}")
    else:
        print("No training examples found")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_training_examples.py <template_json_path> [output_dir]")
        print("Example: python extract_training_examples.py ../DragNDropTest/src/templates/ChineseRestaurant.json")
        sys.exit(1)
    
    template_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "training_data"
    
    extract_examples_from_template(template_path, output_dir)

