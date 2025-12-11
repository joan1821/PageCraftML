# Training the LayoutGNN Model

The GNN model learns to predict layout sizes and alignments from examples rather than using hardcoded rules.

## Training Data Format

Create training examples in JSON format. Each file should contain:

```json
{
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
          }
        ]
      }
    ],
    "Tablet (768x1024)": [
      {
        "id": "box-1",
        "width": -1,
        "height": 75,
        "alignItems": "flex-start",
        "children": [
          {
            "id": "box-2",
            "width": -2,
            "height": -1
          }
        ]
      }
    ]
  }
}
```

## Creating Training Data

1. **From existing templates**: Extract desktop and target resolution data from your JSON templates
2. **Manual creation**: Create examples showing how desktop layouts should adapt to different resolutions
3. **Use the example generator**: The training script will create an example file structure

## Training the Model

```bash
# Basic training
python train_model.py

# With custom parameters
python train_model.py --data_dir training_data --epochs 200 --learning_rate 0.0001

# The model will be saved to model_checkpoint.pth
```

## Using the Trained Model

Once trained, the model will automatically be loaded by `nn_server.py` when processing requests.

The server will:
1. Check for `model_checkpoint.pth`
2. Load the trained weights if available
3. Use model predictions instead of rule-based logic
4. Fall back to rules if no trained model exists

## Training Tips

1. **More examples = better predictions**: Collect examples from real projects
2. **Diverse layouts**: Include various layout patterns (rows, columns, nested structures)
3. **All resolutions**: Provide examples for all target resolutions you want to support
4. **Iterative improvement**: Start with a few examples, train, test, add more examples

## Current Status

The model architecture is ready, but the training loop needs to be completed to properly:
- Batch graphs of different sizes
- Map node-level predictions back to items
- Handle variable-length item structures

The current implementation uses rule-based predictions as a baseline. Once training is fully implemented, the model will learn these patterns from data.

