import torch
import numpy as np

def get_1d_sincos_pos_embed_from_grid(embed_dim, positions):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    positions = positions.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", positions, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()


class SimplePool():
    def __init__(self, pool_size, version='pt', min_size=1):
        self.pool_size = pool_size
        self.version = version
        self.items = []
        self.min_size = min_size
        
        if not (version=='pt' or version=='np'):
            print('version = %s; please choose pt or np')
            assert(False) # please choose pt or np
            
    def __len__(self):
        return len(self.items)
    
    def mean(self, min_size=None):
        if min_size is None:
            pool_size_thresh = self.min_size
        elif min_size=='half':
            pool_size_thresh = self.pool_size/2
        else:
            pool_size_thresh = min_size
            
        if self.version=='np':
            if len(self.items) >= pool_size_thresh:
                return np.sum(self.items)/float(len(self.items))
            else:
                return np.nan
        if self.version=='pt':
            if len(self.items) >= pool_size_thresh:
                return torch.sum(self.items)/float(len(self.items))
            else:
                return torch.from_numpy(np.nan)
    
    def sample(self, with_replacement=True):
        idx = np.random.randint(len(self.items))
        if with_replacement:
            return self.items[idx]
        else:
            return self.items.pop(idx)
    
    def fetch(self, num=None):
        if self.version=='pt':
            item_array = torch.stack(self.items)
        elif self.version=='np':
            item_array = np.stack(self.items)
        if num is not None:
            # there better be some items
            assert(len(self.items) >= num)
                
            # if there are not that many elements just return however many there are
            if len(self.items) < num:
                return item_array
            else:
                idxs = np.random.randint(len(self.items), size=num)
                return item_array[idxs]
        else:
            return item_array
            
    def is_full(self):
        full = len(self.items)==self.pool_size
        return full
    
    def empty(self):
        self.items = []

    def have_min_size(self):
        return len(self.items) >= self.min_size
        
            
    def update(self, items):
        for item in items:
            if len(self.items) < self.pool_size:
                # the pool is not full, so let's add this in
                self.items.append(item)
            else:
                # the pool is full
                # pop from the front
                self.items.pop(0)
                # add to the back
                self.items.append(item)
        return self.items
