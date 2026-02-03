import torch
from torch.utils.data import Dataset
import numpy as np
from datatools import MeasurementDataReader, Measurement, Tool
from datatools import to_ts_data
from fhgutils import Segment, contextual_recarray_dtype, filter_ts_data
from fhgutils import filter_labels, one_label_per_window
from seglearn.base import TS_Data
from seglearn.pipe import Pype
import torch.nn.functional as F
from src.utils import get_mfcc

class ToolTrackingDataset(Dataset):
    def __init__(self, source_path, tool_name="electric_screwdriver", window_length=0.4, overlap=0.5, exclude_time=True):
        self.source_path = source_path
        self.exclude_time = exclude_time
        
        mdr = MeasurementDataReader(source=source_path)
        
        print(f"[INFO] Querying data for tool: {tool_name}...")
        self.data_dict = mdr.query().filter_by(Tool == tool_name).get()

        print("[INFO] Converting to TS Data...")
        Xt, Xc, y = to_ts_data(self.data_dict, contextual_recarray_dtype)
        
        print(f"[INFO] Segmenting data (Window: {window_length}s, Overlap: {overlap})...")
        X_obj = TS_Data(Xt, Xc)
                
        pipe = Pype([
            ('segment', Segment(window_length=window_length, 
                                overlap=overlap, 
                                enforce_size=True, 
                                n=len(np.unique(Xc.desc))))
        ])
        
        self.X_trans, self.y_trans = pipe.fit_transform(X_obj, y)
        # self.X_trans, self.y_trans are lists of TS_Data and labels respectively

        # here X_trans and y_trans are like this

        # acc[0]
        # gyr[0]
        # mag[0]
        # mic[0]
        # acc[1]
        # gyr[1]
        # mag[1]
        # ...
        # ...
        # ...

        # so we step by 4 to get one sample with 4 sensors (acc, gyr, mag, mic)

        # --- FILTERING ---
        # we only keep the valid datapoints where all 4 windows have the same label
        # like acc=8, gyr=8, mag=8, mic=8 -> keep
        # but acc=8, gyr=8, mag=2, mic=8 -> discard
        print(f"[INFO] Filtering ambiguous windows (Total raw groups: {len(self.y_trans)//4})...")
        self.valid = []

        for i in range(0, len(self.y_trans), 4):
            y_window = self.y_trans[i:i+4]

            if (-1 in y_window[0]) or (-1 in y_window[1]) or (-1 in y_window[2]) or (-1 in y_window[3]):
                print(f"Warning: skipping window {i}! Contains -1")
                continue

            if (not has_unique_label_for_window(y_window[0])) or (not has_unique_label_for_window(y_window[1])) or (not has_unique_label_for_window(y_window[2])) or (not has_unique_label_for_window(y_window[3])):
                print(f"Warning: skipping window {i}! One of them not have majority class")
                continue

            val_acc, counts_acc = np.unique(y_window[0], return_counts=True)
            idx = np.argmax(counts_acc)
            y_acc_window = int(val_acc[idx])

            val_gyr, counts_gyr = np.unique(y_window[1], return_counts=True)
            idx = np.argmax(counts_gyr)
            y_gyr_window = int(val_gyr[idx])

            val_mag, counts_mag = np.unique(y_window[2], return_counts=True)
            idx = np.argmax(counts_mag)
            y_mag_window = int(val_mag[idx])

            val_mic, counts_mic = np.unique(y_window[3], return_counts=True)
            idx = np.argmax(counts_mic)
            y_mic_window = int(val_mic[idx])

            # remove windows that has different labels across sensor types
            if not (y_acc_window == y_gyr_window == y_mag_window == y_mic_window):
                print(f"Warning: skipping window {i}! All the labels are not same")
                continue

            self.valid.append(i)

        # reference shapes for upsampling. We use first valid sample as the ref sample.
        # we will reshape all other samples to match this shape as shape differs with window size
        # and even in same window size, sometimes there are slight variations
        ref_data = self.X_trans[0 : 0+4].ts_data
        self.ref_acc, self.ref_gyr, self.ref_mag, self.ref_mic = ref_data[0].shape, ref_data[1].shape, ref_data[2].shape, ref_data[3].shape
        print(f"[INFO] Filtering ambiguous windows (Total raw groups: {len(self.y_trans)//4})...")
        print(f"[INFO] Filtering complete. Valid groups: {len(self.valid)}")

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        idx = self.valid[idx]
    
        x_acc, x_gyr, x_mag, x_mic = self.X_trans[idx:idx+4].ts_data

        if self.exclude_time:
            x_acc = x_acc[:, 1:] 
            x_gyr = x_gyr[:, 1:]
            x_mag = x_mag[:, 1:]
            x_mic = x_mic[:, 1:] 

        # Input: (82, 3) -> Output: (3, 82)
        x_acc = torch.tensor(x_acc, dtype=torch.float32).permute(1, 0)
        x_gyr = torch.tensor(x_gyr, dtype=torch.float32).permute(1, 0)
        x_mag = torch.tensor(x_mag, dtype=torch.float32).permute(1, 0)
        x_mic = torch.tensor(x_mic, dtype=torch.float32).permute(1, 0)

        # upsample to match refmag shape
        x_acc = fix_len(x_acc, self.ref_mag[0])
        x_gyr = fix_len(x_gyr, self.ref_mag[0])
        x_mag = fix_len(x_mag, self.ref_mag[0])

        #upsample to match ref mic shape
        x_mic = fix_len(x_mic, self.ref_mic[0])

        
        y_window = self.y_trans[idx:idx+4]
        y_label = torch.tensor(y_window[0][0], dtype=torch.long)
        
        return x_acc, x_gyr, x_mag, x_mic, y_label
    

def fix_len(tensor, target_len):
    current_len = tensor.shape[1]
    if current_len != target_len:
        return F.interpolate(tensor.unsqueeze(0), size=target_len, mode='linear', align_corners=False).squeeze(0)
    return tensor


class ToolTrackingDataset2(Dataset):
    def __init__(self, source_path, tool_name="electric_screwdriver", window_length=0.4, overlap=0.5, exclude_time=True):
        self.source_path = source_path
        self.exclude_time = exclude_time
        
        mdr = MeasurementDataReader(source=source_path)
        
        print(f"[INFO] Querying data for tool: {tool_name}...")
        self.data_dict = mdr.query().filter_by(Tool == tool_name).get()

        print("[INFO] Converting to TS Data...")
        Xt, Xc, y = to_ts_data(self.data_dict, contextual_recarray_dtype)
        
        print(f"[INFO] Segmenting data (Window: {window_length}s, Overlap: {overlap})...")
        X_obj = TS_Data(Xt, Xc)
                
        pipe = Pype([
            ('segment', Segment(window_length=window_length, 
                                overlap=overlap, 
                                enforce_size=True, 
                                n=len(np.unique(Xc.desc))))
        ])
        
        self.X_trans, self.y_trans = pipe.fit_transform(X_obj, y)

        print(f"[INFO] Filtering ambiguous windows (Total raw groups: {len(self.y_trans)//4})...")
        self.valid = []

        for i in range(0, len(self.y_trans), 4):
            y_window = self.y_trans[i:i+3]
            
            if (-1 in y_window[0]) or (-1 in y_window[1]) or (-1 in y_window[2]):
                print(f"Warning: skipping window {i}! Contains -1")
                continue

            if (not has_unique_label_for_window(y_window[0])) or (not has_unique_label_for_window(y_window[1])) or (not has_unique_label_for_window(y_window[2])):
                print(f"Warning: skipping window {i}! All the labels are not same")
                continue

            val_acc, counts_acc = np.unique(y_window[0], return_counts=True)
            idx = np.argmax(counts_acc)
            y_acc_window = int(val_acc[idx])

            val_gyr, counts_gyr = np.unique(y_window[1], return_counts=True)
            idx = np.argmax(counts_gyr)
            y_gyr_window = int(val_gyr[idx])

            val_mag, counts_mag = np.unique(y_window[2], return_counts=True)
            idx = np.argmax(counts_mag)
            y_mag_window = int(val_mag[idx])

            # remove windows that has different labels across sensor types
            if not (y_acc_window == y_gyr_window == y_mag_window):
                print(f"Warning: skipping window {i}! All the labels are not same")
                continue

            self.valid.append(i)

        # reference shapes for upsampling. We use first valid sample as the ref sample.
        # we will reshape all other samples to match this shape as shape differs with window size
        # and even in same window size, sometimes there are slight variations
        ref_data = self.X_trans[0 : 0+4].ts_data
        self.ref_acc, self.ref_gyr, self.ref_mag, self.ref_mic = ref_data[0].shape, ref_data[1].shape, ref_data[2].shape, ref_data[3].shape
        print(f"[INFO] Filtering ambiguous windows (Total raw groups: {len(self.y_trans)//4})...")
        print(f"[INFO] Filtering complete. Valid groups: {len(self.valid)}")

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        idx = self.valid[idx]
    
        x_acc, x_gyr, x_mag, x_mic = self.X_trans[idx:idx+4].ts_data

        if self.exclude_time:
            x_acc = x_acc[:, 1:] 
            x_gyr = x_gyr[:, 1:]
            x_mag = x_mag[:, 1:]
            # x_mic = x_mic[:, 1:] 

        # Input: (82, 3) -> Output: (3, 82)
        x_acc = torch.tensor(x_acc, dtype=torch.float32).permute(1, 0)
        x_gyr = torch.tensor(x_gyr, dtype=torch.float32).permute(1, 0)
        x_mag = torch.tensor(x_mag, dtype=torch.float32).permute(1, 0)
        # x_mic = torch.tensor(x_mic, dtype=torch.float32).permute(1, 0)

        # upsample to match refmag shape
        x_acc = fix_len(x_acc, self.ref_mag[0])
        x_gyr = fix_len(x_gyr, self.ref_mag[0])
        x_mag = fix_len(x_mag, self.ref_mag[0])

        #upsample to match ref mic shape
        # x_mic = fix_len(x_mic, self.ref_mic[0])

        y_window = self.y_trans[idx:idx+4]
        y_label = torch.tensor(y_window[0][0], dtype=torch.long)
        
        return (x_acc, x_gyr, x_mag), y_label
    
def has_unique_label_for_window(y):
    values, counts = np.unique(y, return_counts=True)
    # idx = np.argmax(counts)
    max_count = np.max(counts)
    if max_count > 0.5 * np.sum(counts):
        return True
    else: return False


class ToolTrackingDataset3(Dataset):
    def __init__(self, source_path, tool_name="electric_screwdriver", window_length=0.4, overlap=0.5, exclude_time=True):
        self.source_path = source_path
        self.exclude_time = exclude_time
        
        mdr = MeasurementDataReader(source=source_path)
        
        print(f"[INFO] Querying data for tool: {tool_name}...")
        self.data_dict = mdr.query().filter_by(Tool == tool_name).get()

        print("[INFO] Converting to TS Data...")
        Xt, Xc, y = to_ts_data(self.data_dict, contextual_recarray_dtype)
        
        print(f"[INFO] Segmenting data (Window: {window_length}s, Overlap: {overlap})...")
        X_obj = TS_Data(Xt, Xc)
                
        pipe = Pype([
            ('segment', Segment(window_length=window_length, 
                                overlap=overlap, 
                                enforce_size=True, 
                                n=len(np.unique(Xc.desc))))
        ])
        
        self.X_trans, self.y_trans = pipe.fit_transform(X_obj, y)

        print(f"[INFO] Filtering ambiguous windows (Total raw groups: {len(self.y_trans)//4})...")
        self.valid = []

        for i in range(0, len(self.y_trans), 4):
            y_window = self.y_trans[i:i+3]
            
            if (-1 in y_window[0]) or (-1 in y_window[1]) or (-1 in y_window[2]):
                print(f"Warning: skipping window {i}! Contains -1")
                continue

            if (not has_unique_label_for_window(y_window[0])) or (not has_unique_label_for_window(y_window[1])) or (not has_unique_label_for_window(y_window[2])):
                print(f"Warning: skipping window {i}! All the labels are not same")
                continue

            val_acc, counts_acc = np.unique(y_window[0], return_counts=True)
            idx = np.argmax(counts_acc)
            y_acc_window = int(val_acc[idx])

            val_gyr, counts_gyr = np.unique(y_window[1], return_counts=True)
            idx = np.argmax(counts_gyr)
            y_gyr_window = int(val_gyr[idx])

            val_mag, counts_mag = np.unique(y_window[2], return_counts=True)
            idx = np.argmax(counts_mag)
            y_mag_window = int(val_mag[idx])

            # remove windows that has different labels across sensor types
            if not (y_acc_window == y_gyr_window == y_mag_window):
                print(f"Warning: skipping window {i}! All the labels are not same")
                continue

            self.valid.append(i)

        # reference shapes for upsampling. We use first valid sample as the ref sample.
        # we will reshape all other samples to match this shape as shape differs with window size
        # and even in same window size, sometimes there are slight variations
        ref_data = self.X_trans[0 : 0+4].ts_data
        self.ref_acc, self.ref_gyr, self.ref_mag, self.ref_mic = ref_data[0].shape, ref_data[1].shape, ref_data[2].shape, ref_data[3].shape
        print(f"[INFO] Filtering ambiguous windows (Total raw groups: {len(self.y_trans)//4})...")
        print(f"[INFO] Filtering complete. Valid groups: {len(self.valid)}")

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        idx = self.valid[idx]
    
        x_acc, x_gyr, x_mag, x_mic = self.X_trans[idx:idx+4].ts_data

        if self.exclude_time:
            x_acc = x_acc[:, 1:] 
            x_gyr = x_gyr[:, 1:]
            x_mag = x_mag[:, 1:]
            x_mic = x_mic[:, 1:] 

        # Input: (82, 3) -> Output: (3, 82)
        x_acc = torch.tensor(x_acc, dtype=torch.float32).permute(1, 0)
        x_gyr = torch.tensor(x_gyr, dtype=torch.float32).permute(1, 0)
        x_mag = torch.tensor(x_mag, dtype=torch.float32).permute(1, 0)
        x_mic = torch.tensor(x_mic, dtype=torch.float32)

        # upsample to match refmag shape
        target_len = self.ref_acc[0]
        x_acc = fix_len(x_acc, target_len)
        x_gyr = fix_len(x_gyr, target_len)
        x_mag = fix_len(x_mag, target_len)

        #upsample to match ref mic shape
        x_mic = fix_len(x_mic, self.ref_mic[0])
        raw_audio = x_mic.flatten()
        x_min_mfcc = get_mfcc(raw_audio.numpy(), target_frames=target_len, n_mfcc=13, sr=3200)
        # print(f"MFCC Shape: {x_min_mfcc.shape}")
        x_min_mfcc = torch.tensor(x_min_mfcc, dtype=torch.float32).permute(1, 0)

        y_window = self.y_trans[idx:idx+4]
        y_label = torch.tensor(y_window[0][0], dtype=torch.long)
        
        return (x_acc, x_gyr, x_mag, x_min_mfcc), y_label
    

from tqdm import tqdm # Install with: pip install tqdm

class ToolTrackingDataset4(Dataset):
    def __init__(self, source_path, tool_name="electric_screwdriver", window_length=0.4, overlap=0.5, exclude_time=True):
        self.source_path = source_path
        self.exclude_time = exclude_time
        
        mdr = MeasurementDataReader(source=source_path)
        
        print(f"[INFO] Querying data for tool: {tool_name}...")
        self.data_dict = mdr.query().filter_by(Tool == tool_name).get()

        print("[INFO] Converting to TS Data...")
        Xt, Xc, y = to_ts_data(self.data_dict, contextual_recarray_dtype)
        
        print(f"[INFO] Segmenting data (Window: {window_length}s, Overlap: {overlap})...")
        X_obj = TS_Data(Xt, Xc)
                
        pipe = Pype([
            ('segment', Segment(window_length=window_length, 
                                overlap=overlap, 
                                enforce_size=True, 
                                n=len(np.unique(Xc.desc))))
        ])
        
        self.X_trans, self.y_trans = pipe.fit_transform(X_obj, y)

        print(f"[INFO] Filtering ambiguous windows (Total raw groups: {len(self.y_trans)//4})...")
        self.valid = []

        # --- 1. Filter Valid Windows ---
        for i in range(0, len(self.y_trans), 4):
            y_window = self.y_trans[i:i+3]
            
            # Check for invalid labels
            if any(-1 in w for w in y_window): continue
            
            # Check for mixed labels in one window
            if any(not has_unique_label_for_window(w) for w in y_window): continue

            # Check majority vote consistency
            labels = []
            for w in y_window:
                val, counts = np.unique(w, return_counts=True)
                labels.append(int(val[np.argmax(counts)]))

            if len(set(labels)) > 1: continue

            self.valid.append(i)

        # --- 2. Determine Reference Shapes ---
        # Get shapes from the first valid sample
        ref_data = self.X_trans[self.valid[0] : self.valid[0]+4].ts_data
        self.ref_acc_len = ref_data[0].shape[0] # e.g., 41
        self.ref_mic_len = ref_data[3].shape[0] # e.g., 3200
        
        print(f"[INFO] Filtering complete. Valid groups: {len(self.valid)}")
        print(f"[INFO] Target Frames for MFCC: {self.ref_acc_len}")

        # --- 3. PRE-CALCULATE MFCCs (The Fix) ---
        print("[INFO] Pre-calculating MFCCs for all valid samples... (This runs once)")
        
        # We will store processed tensors in a list so __getitem__ is instant
        self.processed_data = []

        # Use tqdm to show a progress bar
        for idx in tqdm(self.valid, desc="Processing MFCCs"):
            # Extract raw data
            x_acc, x_gyr, x_mag, x_mic = self.X_trans[idx:idx+4].ts_data

            if self.exclude_time:
                x_acc = x_acc[:, 1:] 
                x_gyr = x_gyr[:, 1:]
                x_mag = x_mag[:, 1:]
                x_mic = x_mic[:, 1:] 

            # Prepare Tensors (IMU)
            x_acc = torch.tensor(x_acc, dtype=torch.float32).permute(1, 0)
            x_gyr = torch.tensor(x_gyr, dtype=torch.float32).permute(1, 0)
            x_mag = torch.tensor(x_mag, dtype=torch.float32).permute(1, 0)

            # Resize IMU to target length (41)
            x_acc = fix_len(x_acc, self.ref_acc_len)
            x_gyr = fix_len(x_gyr, self.ref_acc_len)
            x_mag = fix_len(x_mag, self.ref_acc_len)

            # --- PROCESS MFCC NOW ---
            # 1. Fix raw audio length first
            x_mic = torch.tensor(x_mic, dtype=torch.float32)
            x_mic = fix_len(x_mic, self.ref_mic_len)
            
            # 2. Extract MFCC
            raw_audio = x_mic.flatten()
            mfcc = get_mfcc(
                raw_audio.numpy(), 
                target_frames=self.ref_acc_len, # Target = 41
                n_mfcc=13, 
                sr=3200
            )
            
            # 3. Convert to tensor (13, 41)
            x_mfcc = torch.tensor(mfcc, dtype=torch.float32).permute(1, 0)

            # Get Label
            y_window = self.y_trans[idx:idx+4]
            y_label = torch.tensor(y_window[0][0], dtype=torch.long)

            # Store everything as a tuple
            self.processed_data.append(((x_acc, x_gyr, x_mag, x_mfcc), y_label))

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        # Simply return the pre-calculated data
        return self.processed_data[idx]