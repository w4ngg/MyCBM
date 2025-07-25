# Stanford Cars Dataset to ImageFolder Converter

This repository contains Python scripts to convert the Stanford Cars dataset from its original format to PyTorch's ImageFolder structure for easy model training.

## Prerequisites

```bash
pip install scipy numpy pathlib
```

## Dataset Structure

### Original Stanford Cars Structure:
```
stanford_cars/
├── cars_test_annos_withlabels.mat
├── cars_train/
│   └── *.jpg
├── cars_test/
│   └── *.jpg
└── devkit/
    ├── cars_meta.mat
    ├── cars_test_annos.mat
    ├── cars_train_annos.mat
    ├── eval_train.m
    ├── README.txt
    └── train_perfect_preds.txt
```

### Output ImageFolder Structure:
```
stanford_cars_imagefolder/
├── train/
│   ├── 001_AM_General_Hummer_SUV_2000/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── 002_Acura_RL_Sedan_2012/
│   │   └── ...
│   └── ...
└── test/
    ├── 001_AM_General_Hummer_SUV_2000/
    ├── 002_Acura_RL_Sedan_2012/
    └── ...
```

## Usage

### Method 1: Direct Script Execution

```bash
python convert_stanford_cars.py
```

This will use the default paths:
- Input: `/kaggle/MyCBM/stanford_cars`
- Output: `/kaggle/MyCBM/stanford_cars_imagefolder`

### Method 2: Command Line with Custom Paths

```bash
python convert_stanford_cars_usage.py --input /path/to/stanford_cars --output /path/to/output
```

### Method 3: Import as Module

```python
from convert_stanford_cars import convert_stanford_cars_to_imagefolder

# Convert dataset
convert_stanford_cars_to_imagefolder(
    dataset_path="/path/to/stanford_cars",
    output_path="/path/to/output"
)
```

## Using with PyTorch

After conversion, you can use the dataset with PyTorch's ImageFolder:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder('/path/to/output/train', transform=transform)
test_dataset = datasets.ImageFolder('/path/to/output/test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Get class information
num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
print(f"Number of classes: {num_classes}")
print(f"Classes: {class_names[:5]}...")  # Show first 5 classes
```

## Features

- **Automatic class directory creation**: Creates numbered directories with clean class names
- **Handles missing files**: Warns about missing images but continues processing
- **Test set support**: Processes test images if labels are available (`cars_test_annos_withlabels.mat`)
- **Statistics reporting**: Shows conversion statistics including image counts
- **Error handling**: Comprehensive error handling and validation
- **Flexible paths**: Easy to customize input and output paths

## Notes

- The Stanford Cars dataset has 196 classes (car makes/models)
- Class directories are named with format: `{class_id:03d}_{clean_class_name}`
- If test labels are not available, test images are copied to `test_unlabeled/` directory
- The script preserves original image files (copies, doesn't move)

## Troubleshooting

1. **Missing scipy**: Install with `pip install scipy`
2. **Missing .mat files**: Ensure the devkit directory contains all required .mat files
3. **Permission errors**: Check write permissions for the output directory
4. **Memory issues**: For large datasets, the script processes images one by one to minimize memory usage