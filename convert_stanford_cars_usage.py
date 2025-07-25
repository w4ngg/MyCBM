"""
Usage example for Stanford Cars dataset conversion to ImageFolder structure.

This script demonstrates how to use the converter with different options.
"""

from convert_stanford_cars import convert_stanford_cars_to_imagefolder
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Convert Stanford Cars dataset to ImageFolder structure')
    parser.add_argument('--input', '-i', type=str, 
                       default='/kaggle/MyCBM/stanford_cars',
                       help='Path to the stanford_cars directory')
    parser.add_argument('--output', '-o', type=str, 
                       default='/kaggle/MyCBM/stanford_cars_imagefolder',
                       help='Path where the ImageFolder structure will be created')
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input):
        print(f"Error: Input path {args.input} does not exist!")
        print("Please provide a valid path to the stanford_cars directory.")
        return
    
    # Check required files
    required_files = [
        'devkit/cars_meta.mat',
        'devkit/cars_train_annos.mat',
        'cars_train',
        'cars_test'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(args.input, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: Missing required files/directories:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return
    
    print(f"Converting Stanford Cars dataset...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("-" * 50)
    
    try:
        convert_stanford_cars_to_imagefolder(args.input, args.output)
        print("\n" + "="*50)
        print("SUCCESS: Conversion completed!")
        print(f"You can now use PyTorch's ImageFolder with: {args.output}")
        
        # Example usage code
        print("\nExample PyTorch usage:")
        print("```python")
        print("from torchvision import datasets, transforms")
        print("from torch.utils.data import DataLoader")
        print("")
        print("transform = transforms.Compose([")
        print("    transforms.Resize((224, 224)),")
        print("    transforms.ToTensor(),")
        print("    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
        print("])")
        print("")
        print(f"train_dataset = datasets.ImageFolder('{args.output}/train', transform=transform)")
        print(f"test_dataset = datasets.ImageFolder('{args.output}/test', transform=transform)")
        print("")
        print("train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)")
        print("test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)")
        print("```")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()