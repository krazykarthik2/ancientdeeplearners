import json
import urllib.request
import random
import math
import torch
from torch.utils.data import Dataset

class RealGenomicEPDataset(Dataset):
    """
    Real Human Genomic Dataset.
    Fetches real DNA sequence from Ensembl (Chromosome 22) and constructs
    loop prediction pairs based on the locations of real biological motifs (e.g., TATA and CTCF binding cores).
    """
    def __init__(self, num_samples=1000, seed=42):
        super().__init__()
        random.seed(seed)
        self.num_samples = num_samples
        self.motif_a = "TATA"  # TATA box core
        self.motif_b = "CCGC"  # CTCF binding core
        
        # Fetch real DNA sequence from Ensembl API
        self.raw_seq = self._fetch_ensembl_sequence()
        print(f"Fetched sequence of length {len(self.raw_seq)} from Ensembl (Human Chromosome 22)")
        
        # Build samples
        self.samples = self._build_samples()

    def _fetch_ensembl_sequence(self):
        # Human chromosome 22 region (50,000 bp)
        url = "https://rest.ensembl.org/sequence/region/human/22:30000000..30050000?content-type=application/json"
        try:
            req = urllib.request.Request(url, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data.get('seq', '').upper()
        except Exception as e:
            print(f"Warning: Failed to fetch from Ensembl ({e}). Falling back to synthetic genomic simulation.")
            # Fallback to realistic genomic nucleotide distribution
            return "".join(random.choices(['A', 'C', 'G', 'T'], weights=[0.26, 0.24, 0.24, 0.26], k=50000))

    def _build_samples(self):
        # Locate all occurrences of Motif A and Motif B in the raw genome
        indices_a = []
        indices_b = []
        
        for i in range(len(self.raw_seq) - 4):
            sub = self.raw_seq[i:i+4]
            if sub == self.motif_a:
                indices_a.append(i)
            elif sub == self.motif_b:
                indices_b.append(i)
                
        samples_list = []
        
        # 1. Generate Positive Samples (both motifs present within 24bp)
        pos_pairs = []
        for a in indices_a:
            for b in indices_b:
                if 4 <= abs(a - b) <= 24:
                    pos_pairs.append((a, b))
                    
        random.shuffle(pos_pairs)
        
        # Determine how many positive/negative samples to create (ensuring exactly 50% positive via oversampling)
        target_pos = self.num_samples // 2
        num_neg = self.num_samples - target_pos
        
        # Build positive samples (oversampling unique pairs if needed to balance the dataset)
        for k in range(target_pos):
            a, b = pos_pairs[k % len(pos_pairs)]
            min_idx = min(a, b)
            max_idx = max(a, b) + 4
            
            # Pad window to exactly 32 bp
            rem_len = 32 - (max_idx - min_idx)
            left_pad = random.randint(0, rem_len)
            
            start_coord = min_idx - left_pad
            end_coord = start_coord + 32
            
            # Ensure coordinates are within bounds
            if start_coord >= 0 and end_coord <= len(self.raw_seq):
                window_seq = self.raw_seq[start_coord:end_coord]
                
                # Relative indices in the 32bp window
                rel_a = a - start_coord
                rel_b = b - start_coord
                
                samples_list.append({
                    'seq': window_seq,
                    'is_loop': True,
                    'idx_a': rel_a,
                    'idx_b': rel_b
                })
                
        # 2. Generate Negative Samples (random chunks where we don't have both motifs close)
        attempts = 0
        while len(samples_list) < self.num_samples and attempts < self.num_samples * 10:
            attempts += 1
            start_coord = random.randint(0, len(self.raw_seq) - 32)
            window_seq = self.raw_seq[start_coord:start_coord+32]
            
            # Check if this window already qualifies as a positive
            has_a = self.motif_a in window_seq
            has_b = self.motif_b in window_seq
            
            # If it has only one or neither, it's a negative sample
            if not (has_a and has_b):
                samples_list.append({
                    'seq': window_seq,
                    'is_loop': False,
                    'idx_a': -1,
                    'idx_b': -1
                })
                
        return samples_list

    def __len__(self):
        return len(self.samples)

    def _one_hot(self, seq):
        mapping = {
            'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1],
            'N': [0.25, 0.25, 0.25, 0.25]
        }
        return torch.tensor([mapping.get(c, [0.25, 0.25, 0.25, 0.25]) for c in seq], dtype=torch.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        one_hot = self._one_hot(sample['seq']) # [32, 4]
        
        # Append normalized positional coordinate channel (0.0 to 1.0)
        L = len(sample['seq'])
        pos_coord = torch.arange(L, dtype=torch.float32).view(L, 1) / (L - 1) # [32, 1]
        input_tensor = torch.cat([one_hot, pos_coord], dim=-1) # [32, 5]
        
        target = torch.zeros(32, 32, dtype=torch.float32)
        if sample['is_loop']:
            a = sample['idx_a']
            b = sample['idx_b']
            # Motif A and B are 4 base pairs long.
            # Set the 4 physical interaction spots (start and end of both motifs)
            target[a, b] = 1.0
            target[b, a] = 1.0
            
            # Prevent out-of-bounds if motif is at the very edge
            if a + 3 < 32 and b + 3 < 32:
                target[a, b+3] = 1.0
                target[b+3, a] = 1.0
                
                target[a+3, b] = 1.0
                target[b, a+3] = 1.0
                
                target[a+3, b+3] = 1.0
                target[b+3, a+3] = 1.0
            
        return {
            'sequence': input_tensor,
            'target': target,
            'is_loop': torch.tensor(1.0 if sample['is_loop'] else 0.0, dtype=torch.float32)
        }
