import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DRAMWaveformDataset(Dataset):
    """
    PyTorch Dataset for DRAM waveform input/output pairs.
    Each sample is a dict with:
        - 'x': input waveform (np.ndarray, shape [10000])
        - 'y': output waveform (np.ndarray, shape [10000])
        - 'label': 1 for valid, 0 for invalid
        - 'file': source file name
        - 'channel': channel index
        - 'segment': segment index
    """
    def __init__(self, root_dir, split='train', valid_ratio=0.8, transform=None, files=None, channels=None, segment_limit=None):
        """
        Args:
            root_dir (str): Path to prepared_data directory.
            split (str): 'train' or 'test'.
            valid_ratio (float): Ratio for train/test split.
            transform (callable, optional): Optional transform to apply to samples.
            files (list, optional): List of subdirs (file names) to include.
            channels (list, optional): List of channel indices to include.
            segment_limit (int, optional): Max number of segments per class per channel.
        """
        self.samples = []
        self.transform = transform
        self.split = split
        self.valid_ratio = valid_ratio

        # List all file subdirs if not specified
        if files is None:
            files = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        for file_name in files:
            file_dir = os.path.join(root_dir, file_name)
            for label_str, label_val in [('valid', 1), ('invalid', 0)]:
                class_dir = os.path.join(file_dir, label_str)
                if not os.path.exists(class_dir):
                    continue
                # Group by channel for splitting
                ch_dict = {}
                for fname in os.listdir(class_dir):
                    if fname.endswith('.npz'):
                        # Parse channel and segment from filename
                        parts = fname.split('_')
                        ch = int(parts[0][2:])
                        seg = int(parts[1][3:7])
                        if (channels is not None) and (ch not in channels):
                            continue
                        ch_dict.setdefault(ch, []).append((fname, seg))
                for ch, seglist in ch_dict.items():
                    seglist.sort(key=lambda x: x[1])
                    n_total = len(seglist)
                    n_train = int(n_total * valid_ratio)
                    if split == 'train':
                        use_list = seglist[:n_train]
                    else:
                        use_list = seglist[n_train:]
                    if segment_limit is not None:
                        use_list = use_list[:segment_limit]
                    for fname, seg in use_list:
                        self.samples.append({
                            'path': os.path.join(class_dir, fname),
                            'label': label_val,
                            'file': file_name,
                            'channel': ch,
                            'segment': seg
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        data = np.load(sample_info['path'])
        x = data['x'].astype(np.float32)
        y = data['y'].astype(np.float32)
        label = sample_info['label']
        sample = {
            'x': x,
            'y': y,
            'label': label,
            'file': sample_info['file'],
            'channel': sample_info['channel'],
            'segment': sample_info['segment']
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_dataloaders(root_dir, batch_size=32, valid_ratio=0.8, shuffle=True, num_workers=0, **kwargs):
    """
    Returns PyTorch DataLoader objects for train and test splits.
    """
    train_set = DRAMWaveformDataset(root_dir, split='train', valid_ratio=valid_ratio, **kwargs)
    test_set = DRAMWaveformDataset(root_dir, split='test', valid_ratio=valid_ratio, **kwargs)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

if __name__ == "__main__":
    root = "data/prepared_data"
    batch_size = 4
    train_loader, test_loader = get_dataloaders(root, batch_size=batch_size)
    print("Train set size:", len(train_loader.dataset))
    print("Test set size:", len(test_loader.dataset))
    # Fetch one batch
    for batch in train_loader:
        print("Batch keys:", batch.keys())
        print("x shape:", batch['x'].shape)
        print("y shape:", batch['y'].shape)
        print("label:", batch['label'])
        print("file:", batch['file'])
        print("channel:", batch['channel'])
        print("segment:", batch['segment'])
        break