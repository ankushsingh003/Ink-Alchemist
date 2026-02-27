import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse

def preprocess_fragment(fragment_id, data_dir, output_dir, z_start=16, z_end=48):
    """
    Industry-level preprocessing: 
    Converts a stack of TIFF slices into a single memory-mapped NumPy volume.
    """
    frag_path = os.path.join(data_dir, "train", str(fragment_id))
    surface_vol_path = os.path.join(frag_path, "surface_volume")
    
    # 1. Load Metadata
    print(f"--- Processing Fragment {fragment_id} ---")
    mask = cv2.imread(os.path.join(frag_path, "mask.png"), 0)
    labels = cv2.imread(os.path.join(frag_path, "inklabels.png"), 0)
    
    h, w = mask.shape
    z_count = z_end - z_start
    
    print(f"Fragment Shape: {h}x{w}, Target Slices: {z_count}")
    
    # 2. Initialize Volume (Using float16 to save space if needed, or uint8 for raw)
    # We use uint8 to mirror the input data format efficiently
    volume = np.zeros((z_count, h, w), dtype=np.uint8)
    
    # 3. Load Slices
    for i, z in enumerate(tqdm(range(z_start, z_end), desc="Loading Slices")):
        slice_path = os.path.join(surface_vol_path, f"{z:02d}.tif")
        if os.path.exists(slice_path):
            # Using cv2.IMREAD_UNCHANGED for 16-bit support if the TIFFs are 16-bit
            # Competition data is 8-bit or 16-bit depending on the slice
            img = cv2.imread(slice_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Normalize or resize if needed, but for now we keep raw
                volume[i] = img
        else:
            print(f"Warning: Slice {slice_path} not found.")

    # 4. Save as Memory-Mapped NumPy Array
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"fragment{fragment_id}_volume.npy")
    np.save(out_file, volume)
    
    print(f"Success: Saved volume to {out_file} ({os.path.getsize(out_file) / 1e9:.2f} GB)")
    
    # 5. Save Labels and Mask for quick access
    np.save(os.path.join(output_dir, f"fragment{fragment_id}_labels.npy"), labels)
    np.save(os.path.join(output_dir, f"fragment{fragment_id}_mask.npy"), mask)

class TiledDataset:
    """
    Demonstrates Industry-Level tiling.
    Splits large fragments into overlapping patches for GPU training.
    """
    def __init__(self, volume, tile_size=512, stride=256):
        self.volume = volume # (Z, H, W)
        self.tile_size = tile_size
        self.stride = stride
        
    def generate_tiles(self):
        z, h, w = self.volume.shape
        tiles = []
        for y in range(0, h - self.tile_size + 1, self.stride):
            for x in range(0, w - self.tile_size + 1, self.stride):
                tile = self.volume[:, y:y+self.tile_size, x:x+self.tile_size]
                tiles.append(tile)
        return np.array(tiles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vesuvius Phase 1 Preprocessing")
    parser.add_argument("--fragment", type=int, default=1, help="Fragment ID to process")
    parser.add_argument("--data_dir", type=str, default="d:/ink_detection_fragments", help="Root data dir")
    parser.add_argument("--out_dir", type=str, default="d:/ink_detection_fragments/processed", help="Output dir")
    
    args = parser.parse_args()
    preprocess_fragment(args.fragment, args.data_dir, args.out_dir)
