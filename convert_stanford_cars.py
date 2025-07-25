import os
import shutil
from scipy.io import loadmat
import numpy as np
from pathlib import Path

def convert_stanford_cars_to_imagefolder(dataset_path, output_path):
    """
    Convert Stanford Cars dataset to ImageFolder structure for PyTorch training.
    
    Args:
        dataset_path (str): Path to the stanford_cars directory
        output_path (str): Path where the ImageFolder structure will be created
    """
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Create output directories
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    devkit_path = dataset_path / "devkit"
    cars_meta = loadmat(devkit_path / "cars_meta.mat")
    cars_train_annos = loadmat(devkit_path / "cars_train_annos.mat")
    
    # Extract class names from metadata
    class_names = [item[0] for item in cars_meta['class_names'][0]]
    print(f"Found {len(class_names)} classes")
    
    # Create class directories for training set
    for i, class_name in enumerate(class_names):
        # Clean class name for directory naming (remove special characters)
        clean_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        class_dir_name = f"{i+1:03d}_{clean_name.replace(' ', '_')}"
        
        train_class_dir = train_dir / class_dir_name
        test_class_dir = test_dir / class_dir_name
        train_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)
    
    # Process training images
    print("Processing training images...")
    train_annotations = cars_train_annos['annotations'][0]
    cars_train_path = dataset_path / "cars_train"
    
    for annotation in train_annotations:
        # Extract information from annotation
        fname = annotation[5][0]  # filename
        class_id = annotation[4][0][0]  # class id (1-indexed)
        
        # Source and destination paths
        src_path = cars_train_path / fname
        class_name = class_names[class_id - 1]  # Convert to 0-indexed
        clean_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        class_dir_name = f"{class_id:03d}_{clean_name.replace(' ', '_')}"
        dst_path = train_dir / class_dir_name / fname
        
        # Copy file if source exists
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: Training image {fname} not found")
    
    # Process test images (if labels are available)
    print("Processing test images...")
    
    # Check if test annotations with labels exist
    test_annos_with_labels_path = dataset_path / "cars_test_annos_withlabels.mat"
    if test_annos_with_labels_path.exists():
        test_annotations = loadmat(test_annos_with_labels_path)['annotations'][0]
        cars_test_path = dataset_path / "cars_test"
        
        for annotation in test_annotations:
            # Extract information from annotation
            fname = annotation[5][0]  # filename
            class_id = annotation[4][0][0]  # class id (1-indexed)
            
            # Source and destination paths
            src_path = cars_test_path / fname
            class_name = class_names[class_id - 1]  # Convert to 0-indexed
            clean_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            class_dir_name = f"{class_id:03d}_{clean_name.replace(' ', '_')}"
            dst_path = test_dir / class_dir_name / fname
            
            # Copy file if source exists
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Test image {fname} not found")
    else:
        print("Warning: Test annotations with labels not found. Test images will not be organized by class.")
        # If no labels available, just copy all test images to a single directory
        cars_test_path = dataset_path / "cars_test"
        if cars_test_path.exists():
            unlabeled_test_dir = output_path / "test_unlabeled"
            unlabeled_test_dir.mkdir(exist_ok=True)
            for img_file in cars_test_path.glob("*.jpg"):
                shutil.copy2(img_file, unlabeled_test_dir / img_file.name)
    
    print(f"\nConversion completed!")
    print(f"ImageFolder structure created at: {output_path}")
    print(f"Training images: {train_dir}")
    print(f"Test images: {test_dir}")
    
    # Print statistics
    train_count = sum(len(list(class_dir.glob("*.jpg"))) for class_dir in train_dir.iterdir() if class_dir.is_dir())
    test_count = sum(len(list(class_dir.glob("*.jpg"))) for class_dir in test_dir.iterdir() if class_dir.is_dir())
    
    print(f"\nStatistics:")
    print(f"Total training images: {train_count}")
    print(f"Total test images: {test_count}")
    print(f"Number of classes: {len(class_names)}")

def main():
    # Default paths
    dataset_path = "/kaggle/MyCBM/stanford_cars"
    output_path = "/kaggle/MyCBM/stanford_cars_imagefolder"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist!")
        print("Please update the dataset_path variable in the script.")
        return
    
    print(f"Converting Stanford Cars dataset from: {dataset_path}")
    print(f"Output ImageFolder structure to: {output_path}")
    
    try:
        convert_stanford_cars_to_imagefolder(dataset_path, output_path)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()