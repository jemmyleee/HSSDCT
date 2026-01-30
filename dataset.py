import glob
import numpy as np
import torch, os, pdb
import random as rn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torch.multiprocessing
import cv2
from PIL import Image
# from scipy.io import loadmat

def getAffine(src, y, pmax = 0.08, pmin=0.92, angle_max = 180, scale_min=0.9, scale_max=1.1):
   
        
        srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]] ).astype(np.float32)
        dstTri = np.array( [[0, src.shape[1]*rn.uniform(0, pmax)], 
                            [src.shape[1]*rn.uniform(pmin, 1), src.shape[0]*rn.uniform(0, pmax)], 
                            [src.shape[1]*rn.uniform(pmin, 1), src.shape[0]*rn.uniform(0, pmax)]] ).astype(np.float32)
        # Rotating the image after Warp
        angle = rn.randint(0, angle_max)
        scale = rn.uniform(scale_min, scale_max)
        
        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
        center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)

        rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
        warp_rotate_dst = cv2.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
        
        warp_dst = cv2.warpAffine(y, warp_mat, (src.shape[1], src.shape[0]))
        warp_rotate_dst2 = cv2.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
        return warp_rotate_dst, warp_rotate_dst2

class dataset_h5(torch.utils.data.Dataset):
    
    def __init__(self, X, args, mode='train'):

        super(dataset_h5, self).__init__()

        self.root       = args.root
        self.fns        = X
        self.n_images   = len(self.fns)
        self.indices    = np.array(range(self.n_images))
        self.offset     = (args.image_size- args.crop_size)
        self.offsetB    = (args.bands- args.crop_size)
        self.mis_pix    = args.mis_pix
        self.msi        = args.msi_bands
        self.mixed      = args.mixed_align_opt==1

        self.mode       = mode.lower()
        self.crop_size  = args.crop_size
        self.img_size   = args.image_size
        
        self.offsets = []
        print(f'Found {len(X)} image in {mode} mode!')
        
    def __getitem__(self, index):
        data, label = {}, {}
        
        fn = os.path.join(self.root, self.fns[index])
        y = np.load(fn+'GT.npz.npy').astype(np.float32)
        y = y[:256,:256, :]
        
        x = np.load(fn+f'co{self.msi}{self.mis_pix+1}.npz.npy').astype(np.float32)
        assert x.shape==(self.img_size,self.img_size,172)     
        assert y.shape==(self.img_size,self.img_size,172)
        maxv = y.max()
        minv = y.min()
        y = (y-minv) / (maxv-minv)         
        y = (y-0.5) * 2
        x = (x-0.5) * 2
        
        if self.mode=='train':
            
            # Random crop
            xim, yim = rn.randint(0, self.offset), rn.randint(0, self.offset)
                
            h = yim + self.crop_size
            w = xim + self.crop_size
            
            x = x[yim:h, xim:w,:]
            y = y[yim:h, xim:w,:]
            
            if rn.random()>=0.5:
                x = np.flip(x, 1).copy()
                y = np.flip(y, 1).copy()
            if rn.random()>=0.5:
                x = np.flip(x, 0).copy()
                y = np.flip(y, 0).copy()
            
            if rn.random()>=0.5:
                times = rn.randint(1,3)
                x = np.rot90(x, times).copy()
                y = np.rot90(y, times).copy()
                
        
        x=np.transpose(x, (2,0,1))
        y=np.transpose(y, (2,0,1))
        
        return x, y, fn, maxv, minv

    def __len__(self):
        return self.n_images
    
class dataset_joint_(torch.utils.data.Dataset):
    
    hrmsi = ['hrmsi4', 'hrmsi6']
    lrhsi = ['lrhsi', 'lrhsi1', 'lrhsi2', 'lrhsi3']
    
    def __init__(self, X, args, mode='train'):

        super(dataset_joint_, self).__init__()

        self.root       = args.root
        self.fns        = X
        self.n_images   = len(self.fns)
        self.indices    = np.array(range(self.n_images))
        self.offset     = (args.image_size- args.crop_size)
        self.offsetB    = (args.bands- args.crop_size)
        self.mis_pix    = args.mis_pix
        self.msi        = args.msi_bands
        #self.hsi        = args.hsi_bands
        self.mixed      = args.mixed_align_opt==1

        self.mode       = mode.lower()
        self.crop_size  = args.crop_size
        self.img_size   = args.image_size
        
        self.offsets = []
        print(f'Found {len(X)} image in {mode} mode!')
        for i in range(0,self.offset, 4):
            self.offsets.append(i)
    
        
    def __getitem__(self, index):
        data, label = {}, {}
        
        fn = os.path.join(self.root, self.fns[index])
        
        x1key, x2key = f'hrmsi{self.msi}', 'lrhsi'
        if self.mixed and self.mode=='train':
            x2key = rn.choice(self.lrhsi)
        
        elif self.mis_pix > 0:
            x2key = eval(f"'lrhsi{self.mis_pix}'") if self.mis_pix>0 else "lrhsi"
            
        x = np.load(fn+'hrmsi.npz')[x1key].astype(np.float32)
        x2= np.load(fn+'lrhsi.npz')[x2key].astype(np.float32)
        y   = np.load(fn+'GT.npz.npy').astype(np.float32)
        y = y[:256,:256, :]
        assert y.shape==(self.img_size,self.img_size,172)
        maxv = y.max()
        minv = y.min()
        y = (y-minv) / (maxv-minv)         
        y = (y-0.5) * 2
        x = (x-0.5) * 2
        x2 = (x2-0.5) *2
        if self.mode=='train':
                        
            # Random crop
            xim, yim = rn.choice(self.offsets), rn.choice(self.offsets)
            
            h = yim + self.crop_size
            w = xim + self.crop_size
            h2 = yim//4 + self.crop_size//4
            w2 = xim//4 + self.crop_size//4
            
            
            x = x[yim:h, xim:w,:]
            x2 = x2[yim//4:h2, xim//4:w2,:]
            y = y[yim:h, xim:w,:]
            
            if rn.random()>=0.5:
                x = np.flip(x, 1).copy()
                x2 = np.flip(x2, 1).copy()
                y = np.flip(y, 1).copy()
            if rn.random()>=0.5:
                x = np.flip(x, 0).copy()
                x2 = np.flip(x2, 0).copy()
                y = np.flip(y, 0).copy()
                
            if rn.random()>=0.5:
                times = rn.randint(1,3)
                x =  np.rot90(x,  times).copy()
                x2 = np.rot90(x2, times).copy()
                y = np.rot90(y, times).copy()
                
        
        x=np.transpose(x, (2,0,1))
        x2=np.transpose(x2, (2,0,1))
        y=np.transpose(y, (2,0,1))
        return x, x2, y, fn, maxv, minv

    def __len__(self):
        return self.n_images


class dataset_joint(torch.utils.data.Dataset):

    lrhsi_keys = ['lrhsi', 'lrhsi1', 'lrhsi2', 'lrhsi3']

    def __init__(self, X, args, mode='train'):
        """
        Initializes the dataset.

        Args:
            X (list): List of file prefixes.
            args (Namespace): Arguments containing configuration like root path, image size, etc.
                               Expected attributes: root, image_size, crop_size, bands (?),
                                                  mis_pix, msi_bands, hsi_bands, mixed_align_opt.
            mode (str): Dataset mode ('train' or other like 'test'/'val').
        """
        super(dataset_joint, self).__init__()

        self.root         = args.root
        self.fns          = X
        self.n_images     = len(self.fns)
        self.indices      = np.arange(self.n_images) # Use np.arange for clarity
        self.mode         = mode.lower()
        self.crop_size    = args.crop_size
        self.img_size     = args.image_size

        # Check if image_size and crop_size are valid
        if args.image_size < args.crop_size:
             raise ValueError("image_size must be greater than or equal to crop_size")
        self.offset       = (args.image_size - args.crop_size)

        self.mis_pix      = args.mis_pix
        self.msi    = args.msi_bands # Number of bands for HRMSI (e.g., 4 or 6)
        self.hsi    = args.hsi_bands # DESIRED number of bands for LRHSI/HRHSI (e.g., 30 or 172)
        self.mixed        = args.mixed_align_opt == 1

        self.offsets = []
        # Generate possible top-left corner starting points for cropping
        if self.offset >= 0:
            for i in range(0, self.offset + 1, 4):
                 self.offsets.append(i)
        else:
             self.offsets.append(0)

        if not self.offsets:
            self.offsets = [0]


        print(f'Found {len(X)} images in {self.mode} mode!')
        print(f'Target LRHSI bands: {self.hsi}') # Info message
        print(f'Target HRMSI bands: {self.msi}') # Info message

    def __getitem__(self, index):
        """
        Retrieves a single data sample.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (hrmsi_crop, lrhsi_crop, hrhsi_crop, filename_prefix, max_val, min_val)
                   Where crops are numpy arrays with shape (C, H, W).
        """
        fn_prefix = os.path.join(self.root, self.fns[index])

        # Determine keys for loading data
        x1key, x2key = f'hrmsi{self.msi}', 'lrhsi'
        if self.mixed and self.mode=='train':
            x2key = rn.choice(self.lrhsi)
        elif self.mis_pix > 0:
            x2key = eval(f"'lrhsi{self.mis_pix}'") if self.mis_pix>0 else "lrhsi"

        # Load original data
        x = np.load(fn_prefix+'hrmsi.npz')[x1key].astype(np.float32)
        x2_full = np.load(fn_prefix+'lrhsi.npz')[x2key].astype(np.float32)
        y = np.load(fn_prefix+'GT.npz.npy').astype(np.float32)

        # LRHSI (x2) band selection
        original_hsi_bands = x2_full.shape[2]
        desired_hsi_bands = self.hsi

        if desired_hsi_bands < original_hsi_bands:
            # Calculate indices for equidistant sampling
            indices = np.linspace(0, original_hsi_bands - 1, num=desired_hsi_bands)
            indices = np.round(indices).astype(int)
            indices = np.unique(indices)
            x2 = x2_full[:, :, indices]
        elif desired_hsi_bands == original_hsi_bands:
             x2 = x2_full
        else:
             print(f"Warning: Requested {desired_hsi_bands} LRHSI bands, but only {original_hsi_bands} available. Using all bands.")
             x2 = x2_full

        # GT processing and normalization
        y = y[:self.img_size,:self.img_size, :]
        assert y.shape==(self.img_size,self.img_size,172), f"GT shape assertion failed: expected {(self.img_size,self.img_size,172)}, got {y.shape}"

        # Normalization for y based on its own min/max
        maxv = y.max()
        minv = y.min()
        if (maxv - minv) > 1e-6:
            y = (y - minv) / (maxv - minv)
        else:
            y = np.zeros_like(y)
        y = (y - 0.5) * 2  # Scale to [-1, 1]

        # Fixed [-1, 1] normalization for x and x2
        x = (x - 0.5) * 2
        x2 = (x2 - 0.5) * 2

        # Data augmentation in training mode
        if self.mode=='train':

            # Random crop
            if not self.offsets:
                 xim, yim = 0, 0
            else:
                 xim, yim = rn.choice(self.offsets), rn.choice(self.offsets)

            h = yim + self.crop_size
            w = xim + self.crop_size
            h2 = yim // 4 + self.crop_size // 4
            w2 = xim // 4 + self.crop_size // 4

            x = x[yim:h, xim:w, :]
            x2 = x2[yim//4:h2, xim//4:w2, :]
            y = y[yim:h, xim:w, :]

            # Random horizontal flip
            if rn.random() >= 0.5:
                x = np.flip(x, 1).copy()
                x2 = np.flip(x2, 1).copy()
                y = np.flip(y, 1).copy()

            # Random vertical flip
            if rn.random() >= 0.5:
                x = np.flip(x, 0).copy()
                x2 = np.flip(x2, 0).copy()
                y = np.flip(y, 0).copy()

            # Random rotation (90, 180, or 270 degrees)
            if rn.random() >= 0.5:
                times = rn.randint(1, 3)
                x =  np.rot90(x,  times, axes=(0, 1)).copy()
                x2 = np.rot90(x2, times, axes=(0, 1)).copy()
                y = np.rot90(y, times, axes=(0, 1)).copy()

        # Transpose from (H, W, C) to PyTorch's (C, H, W) format
        x = np.transpose(x, (2, 0, 1))
        x2 = np.transpose(x2, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))

        return x.astype(np.float32), x2.astype(np.float32), y.astype(np.float32), fn_prefix, maxv, minv

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.n_images
    
class dataset_joint2(torch.utils.data.Dataset):
    
    hrmsi = ['hrmsi4', 'hrmsi6']
    lrhsi = ['lrhsi', 'lrhsi1', 'lrhsi2', 'lrhsi3']
    co4 = ['co41', 'co42', 'co43', 'co44']
    co6 = ['co61', 'co62', 'co63', 'co64']
    
    def __init__(self, X, args, mode='train'):

        super(dataset_joint2, self).__init__()

        self.root       = args.root
        self.fns        = X
        self.n_images   = len(self.fns)
        self.indices    = np.array(range(self.n_images))
        self.offset     = (args.image_size- args.crop_size)
        self.offsetB    = (args.bands- args.crop_size)
        self.mis_pix    = args.mis_pix
        self.msi        = args.msi_bands
        self.mixed      = args.mixed_align_opt==1

        self.mode       = mode.lower()
        self.crop_size  = args.crop_size
        self.img_size   = args.image_size
        
        self.offsets = []
        print(f'Found {len(X)} image in {mode} mode!')
        for i in range(0,self.offset, 4):
            self.offsets.append(i)
    
        
    def __getitem__(self, index):
        data, label = {}, {}
        
        fn = os.path.join(self.root, self.fns[index])
        
        x1key, x2key, cokey = f'hrmsi{self.msi}', 'lrhsi', f'co{self.msi}1'
        if self.mixed and self.mode=='train':
            rind = rn.randint(0,3)
            x2key = self.lrhsi[rind]
            cokey = self.co4[rind] if self.msi==4 else self.co6[rind]
        
        elif self.mis_pix > 0:
            x2key = f'lrhsi{self.mis_pix}' if self.mis_pix>0 else "lrhsi"
            cokey = f'co{self.msi}{self.mis_pix+1}'
            
        x = np.load(fn+f'{cokey}.npz.npy').astype(np.float32)
        x2 = np.load(fn+'lrhsi.npz')[x2key].astype(np.float32)
        x3 = np.load(fn+'hrmsi.npz')[x1key].astype(np.float32)
        
        y   = np.load(fn+'GT.npz.npy').astype(np.float32)
        y = y[:256,:256, :]
        assert y.shape==(self.img_size,self.img_size,172)
        maxv = y.max()
        minv = y.min()
        y = (y-minv) / (maxv-minv)         
        y =  (y -0.5) * 2
        x =  (x -0.5) * 2
        x2 = (x2-0.5) * 2
        x3 = (x3-0.5) * 2
        
        if self.mode=='train':
                        
            # Random crop
            xim, yim = rn.choice(self.offsets), rn.choice(self.offsets)
            
            h = yim + self.crop_size
            w = xim + self.crop_size
            h2 = yim//4 + self.crop_size//4
            w2 = xim//4 + self.crop_size//4
            
            
            x = x[yim:h, xim:w,:]
            x3 = x3[yim:h, xim:w,:]
            x2 = x2[yim//4:h2, xim//4:w2,:]
            y = y[yim:h, xim:w,:]
            
            if rn.random()>=0.5:
                x = np.flip(x, 1).copy()
                x2 = np.flip(x2, 1).copy()
                x3 = np.flip(x3, 1).copy()
                y = np.flip(y, 1).copy()
            if rn.random()>=0.5:
                x = np.flip(x, 0).copy()
                x2 = np.flip(x2, 0).copy()
                x3 = np.flip(x3, 0).copy()
                y = np.flip(y, 0).copy()
                
            if rn.random()>=0.5:
                times = rn.randint(1,3)
                x =  np.rot90(x,  times).copy()
                x2 = np.rot90(x2, times).copy()
                x3 = np.rot90(x3, times).copy()
                y = np.rot90(y, times).copy()
                
        
        x=np.transpose(x, (2,0,1))
        x2=np.transpose(x2, (2,0,1))
        x3=np.transpose(x3, (2,0,1))
        y=np.transpose(y, (2,0,1))
        return x, x2, x3, y, fn, maxv, minv

    def __len__(self):
        return self.n_images




from perlin_noise import PerlinNoise
from scipy.ndimage import gaussian_filter

class dataset_urgent(torch.utils.data.Dataset):
    """
    A PyTorch dataset for remote sensing image fusion and cloud removal tasks.
    Generates natural cloud effects efficiently while avoiding excessive occlusion and slow computation.
    """
    lrhsi_keys = ['lrhsi', 'lrhsi1', 'lrhsi2', 'lrhsi3']

    def __init__(self, X, args, mode='train'):
        """
        Initialize the dataset.
        """
        super(dataset_urgent, self).__init__()

        self.root = args.root
        self.fns = X
        self.n_images = len(self.fns)
        self.indices = np.arange(self.n_images)
        self.mode = mode.lower()
        self.crop_size = args.crop_size
        self.img_size = args.image_size

        if args.image_size < args.crop_size:
            raise ValueError("image_size must be greater than or equal to crop_size")
        self.offset = (args.image_size - args.crop_size)

        self.mis_pix = args.mis_pix
        self.msi = args.msi_bands
        self.hsi = args.hsi_bands
        self.mixed = args.mixed_align_opt == 1

        # White cloud generation parameters - enhanced version
        self.cloud_prob = getattr(args, 'cloud_prob', 0.7)
        self.cloud_density = getattr(args, 'cloud_density', 0.4)  
        self.cloud_intensity = getattr(args, 'cloud_intensity', 0.9)  # Significantly increased intensity
        self.white_cloud_brightness = getattr(args, 'white_cloud_brightness', 0.95)  # Near-saturated whiteness
        self.cloud_contrast = getattr(args, 'cloud_contrast', 1.2)  # Enhanced contrast
        
        self.offsets = []
        if self.offset >= 0:
            for i in range(0, self.offset + 1, 4):
                self.offsets.append(i)
        
        if not self.offsets:
            self.offsets = [0]

        print(f'Found {len(X)} images in {self.mode} mode!')
        print(f'Target LRHSI bands: {self.hsi}')
        print(f'Target HRMSI bands: {self.msi}')
        if self.mode == 'train':
            print(f'Enhanced white cloud generation - probability: {self.cloud_prob}, density: {self.cloud_density}, intensity: {self.cloud_intensity}')
            print(f'White cloud brightness: {self.white_cloud_brightness}, contrast: {self.cloud_contrast}')

    def generate_fast_cloud_mask(self, height, width):
        """
        Quickly generate a natural cloud mask - optimized for fast computation with natural-looking results.
        """
        # Randomly select cloud type
        cloud_type = np.random.choice(['sparse', 'medium', 'dense'], p=[0.4, 0.4, 0.2])
        
        if cloud_type == 'sparse':
            return self._generate_sparse_clouds(height, width)
        elif cloud_type == 'medium':
            return self._generate_medium_clouds(height, width)
        else:
            return self._generate_dense_clouds(height, width)
    
    def _generate_sparse_clouds(self, height, width):
        """Generate sparse white cloud patches - fluffy cumulus cloud formation."""
        cloud_mask = np.zeros((height, width))
        num_clouds = np.random.randint(2, 5)
        
        for _ in range(num_clouds):
            # Cloud center and size
            center_x = np.random.randint(width // 6, 5 * width // 6)
            center_y = np.random.randint(height // 6, 5 * height // 6)
            radius_x = np.random.randint(width // 8, width // 3)
            radius_y = np.random.randint(height // 8, height // 3)
            
            # Generate fluffy white cloud shape
            y_grid, x_grid = np.ogrid[:height, :width]
            
            # Main cloud body - dense core
            cloud_core = ((x_grid - center_x) / radius_x) ** 2 + ((y_grid - center_y) / radius_y) ** 2
            core_mask = np.exp(-cloud_core * 1.5)  # Dense core
            
            # Outer fluffy region
            cloud_edge = ((x_grid - center_x) / (radius_x * 1.3)) ** 2 + ((y_grid - center_y) / (radius_y * 1.3)) ** 2
            edge_mask = np.exp(-cloud_edge * 0.8)  # Fluffy edges
            
            # Combine into fluffy white cloud
            cloud_shape = np.maximum(core_mask * 0.9, edge_mask * 0.5)
            cloud_shape = np.where(cloud_shape > 0.1, cloud_shape, 0)
            
            # White cloud intensity - biased toward high brightness
            intensity = np.random.uniform(0.6, 0.9)
            cloud_mask = np.maximum(cloud_mask, cloud_shape * intensity)
        
        return cloud_mask.astype(np.float32)
    
    def _generate_medium_clouds(self, height, width):
        """Generate medium white clouds - stratocumulus cloud formation."""
        # Downsampled computation
        small_h, small_w = height // 4, width // 4
        
        # Use Perlin noise to generate natural cloud shapes
        noise = PerlinNoise(octaves=4, seed=np.random.randint(0, 1000))
        small_mask = np.zeros((small_h, small_w))
        
        scale = 0.08  # Slightly larger cloud patches
        for i in range(small_h):
            for j in range(small_w):
                small_mask[i, j] = noise([i * scale, j * scale])
        
        # Normalize and adjust for white cloud characteristics
        small_mask = (small_mask + 1) / 2
        threshold = np.random.uniform(0.35, 0.55)  # Moderate threshold
        small_mask = np.where(small_mask > threshold, 
                             (small_mask - threshold) / (1 - threshold), 0)
        
        # Enhance white cloud features - higher base brightness
        small_mask = small_mask ** 0.7  # Enhance bright areas to make clouds whiter
        
        # Upsample to original size
        from scipy.ndimage import zoom
        cloud_mask = zoom(small_mask, (height / small_h, width / small_w), order=1)
        
        # White cloud intensity
        intensity = np.random.uniform(0.5, 0.8)
        cloud_mask *= intensity
        
        return cloud_mask.astype(np.float32)
    
    def _generate_dense_clouds(self, height, width):
        """Generate dense white cloud layer - stratus cloud formation while maintaining white characteristics."""
        # Use larger downsampling for faster computation
        small_h, small_w = height // 6, width // 6
        
        # Generate layered white clouds
        noise1 = PerlinNoise(octaves=3, seed=np.random.randint(0, 1000))
        noise2 = PerlinNoise(octaves=5, seed=np.random.randint(0, 1000))
        
        small_mask = np.zeros((small_h, small_w))
        
        for i in range(small_h):
            for j in range(small_w):
                base = noise1([i * 0.04, j * 0.04])
                detail = noise2([i * 0.12, j * 0.12])
                small_mask[i, j] = 0.8 * base + 0.2 * detail  # Dominated by large-scale features
        
        # Normalize and enhance white cloud features
        small_mask = (small_mask + 1) / 2
        threshold = np.random.uniform(0.25, 0.45)  # Lower threshold for larger coverage
        small_mask = np.where(small_mask > threshold, 
                             (small_mask - threshold) / (1 - threshold), 0)
        
        # White cloud feature enhancement
        small_mask = small_mask ** 0.6  # Enhance white characteristics
        
        # Upsample
        from scipy.ndimage import zoom
        cloud_mask = zoom(small_mask, (height / small_h, width / small_w), order=1)
        
        # Limit intensity while maintaining white cloud features
        intensity = np.random.uniform(0.6, 0.8)
        cloud_mask *= intensity
        
        return cloud_mask.astype(np.float32)

    def apply_cloud_effect(self, image, cloud_mask):
        """
        Apply enhanced white cloud effect - ensures clouds are highly visible and prominent.
        """
        height, width, channels = image.shape
        cloudy_image = image.copy()
        
        # Enhance cloud mask - higher intensity and contrast
        cloud_mask = np.clip(cloud_mask * self.cloud_intensity, 0, 1.0)
        
        # Enhance cloud edge contrast
        cloud_mask = cloud_mask ** (1.0 / self.cloud_contrast)  # Increase contrast
        
        for c in range(channels):
            if c < 3:  # Visible light bands - extremely bright white clouds
                # Strong brightening effect for white clouds
                cloud_base_brightness = 0.98  # Near maximum brightness
                cloud_saturation = cloud_mask * 0.95  # High saturation
                
                # Strong brightness boost
                brightness_boost = cloud_mask * cloud_base_brightness
                
                # Replace original pixels with white cloud
                cloud_pixel_value = cloud_base_brightness
                mixing_ratio = cloud_mask * 0.9  # Strong mixing ratio
                
                cloudy_image[:, :, c] = (image[:, :, c] * (1 - mixing_ratio) + 
                                       cloud_pixel_value * mixing_ratio)
                
            elif c < 6:  # Near-infrared bands - white clouds still bright
                cloud_base_brightness = 0.9
                cloud_saturation = cloud_mask * 0.8
                mixing_ratio = cloud_mask * 0.8
                
                cloudy_image[:, :, c] = (image[:, :, c] * (1 - mixing_ratio) + 
                                       cloud_base_brightness * mixing_ratio)
                
            else:  # Far-infrared bands - moderately bright white clouds
                cloud_base_brightness = 0.8
                mixing_ratio = cloud_mask * 0.7
                
                cloudy_image[:, :, c] = (image[:, :, c] * (1 - mixing_ratio) + 
                                       cloud_base_brightness * mixing_ratio)
        
        return np.clip(cloudy_image, 0, 1)

    def __getitem__(self, index):
        """
        Retrieve a single data sample.
        """
        fn_prefix = os.path.join(self.root, self.fns[index])

        # Determine keys for loading data
        x1key, x2key = f'hrmsi{self.msi}', 'lrhsi'
        if self.mixed and self.mode == 'train':
            x2key = rn.choice(self.lrhsi_keys)
        elif self.mis_pix > 0:
            x2key = f'lrhsi{self.mis_pix}'

        # Load original data
        x = np.load(fn_prefix + 'hrmsi.npz')[x1key].astype(np.float32)
        x2_full = np.load(fn_prefix + 'lrhsi.npz')[x2key].astype(np.float32)
        y = np.load(fn_prefix + 'GT.npz.npy').astype(np.float32)
        
        # Fast cloud generation (training mode only)
        if self.mode == 'train' and np.random.random() < self.cloud_prob:
            cloud_mask = self.generate_fast_cloud_mask(x.shape[0], x.shape[1])
            x = self.apply_cloud_effect(x, cloud_mask)
        
        # LRHSI (x2) band selection
        original_hsi_bands = x2_full.shape[2]
        desired_hsi_bands = self.hsi

        if desired_hsi_bands < original_hsi_bands:
            indices = np.round(np.linspace(0, original_hsi_bands - 1, num=desired_hsi_bands)).astype(int)
            indices = np.unique(indices)
            x2 = x2_full[:, :, indices]
        else:
            if desired_hsi_bands > original_hsi_bands:
                print(f"Warning: Requested {desired_hsi_bands} LRHSI bands, but only {original_hsi_bands} available. Using all bands.")
            x2 = x2_full

        # GT processing and normalization
        y = y[:self.img_size, :self.img_size, :]
        assert y.shape == (self.img_size, self.img_size, 172), f"GT shape assertion failed: expected {(self.img_size, self.img_size, 172)}, got {y.shape}"
        maxv, minv = y.max(), y.min()
        if (maxv - minv) > 1e-6:
            y = (y - minv) / (maxv - minv)
        else:
            y = np.zeros_like(y)
        y = (y - 0.5) * 2  # Normalize to [-1, 1]

        # Input image normalization (x and x2)
        x = (x - 0.5) * 2
        x2 = (x2 - 0.5) * 2

        # Data augmentation (training mode only)
        if self.mode == 'train':
            # Random crop
            xim, yim = rn.choice(self.offsets), rn.choice(self.offsets)
            h_crop, w_crop = yim + self.crop_size, xim + self.crop_size
            h2_crop, w2_crop = yim // 4 + self.crop_size // 4, xim // 4 + self.crop_size // 4
            
            x = x[yim:h_crop, xim:w_crop, :]
            x2 = x2[yim//4:h2_crop, xim//4:w2_crop, :]
            y = y[yim:h_crop, xim:w_crop, :]

            # Random flip
            if rn.random() >= 0.5:
                x, x2, y = np.flip(x, 1).copy(), np.flip(x2, 1).copy(), np.flip(y, 1).copy()
            if rn.random() >= 0.5:
                x, x2, y = np.flip(x, 0).copy(), np.flip(x2, 0).copy(), np.flip(y, 0).copy()

            # Random rotation
            if rn.random() >= 0.5:
                times = rn.randint(1, 3)
                axes = (0, 1)
                x = np.rot90(x, times, axes).copy()
                x2 = np.rot90(x2, times, axes).copy()
                y = np.rot90(y, times, axes).copy()

        # Dimension transposition (H, W, C) -> (C, H, W)
        x = np.transpose(x, (2, 0, 1))
        x2 = np.transpose(x2, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))

        return x.astype(np.float32), x2.astype(np.float32), y.astype(np.float32), fn_prefix, maxv, minv

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.n_images