import os
import numpy as np
import torch
import tifffile

from torch.utils.data import Dataset

class ToyPANDASDataset(Dataset):
    
    def __init__(self,
                 image_dir: str,
                 mask_dir: str, 
                 root_dir: str,
                 mask_suffix: str = "_mask",
                 imfile_ext: str = ".tiff",
                 maskfile_ext: str = ".tiff",
                 tile_size: int = 224,
                 channel_idx: int = 0,
                 min_thresh: float = 0.5,
                 transform = None,
                 norm_func = None):
        
        """
        Dataset class for Toy PANDA dataset of 10 images and masks.
        
        Parameters
        -----------
        image_dir: str
            The directory containing the images. 
        
        mask_dir: str
            The directory containing the masks paired with each of the images. 
        
        root_dir: str
            The root directory containing both the image and mask folders.
        
        mask_suffix: str
            The suffix associated with mask files in the folder. In this case,
            it is "_mask." Each image is labeled alphanumerically, and its corresponding
            mask file has a "_mask" suffix following the alphanumeric name. 
        
        imfile_ext: str
            The extension or file type of the image file. In this case, it is ".tiff"
            
        maskfile_ext: str
            The extension or file type of the mask file. In this case, it is ".tiff"
        
        tile_size: int
            The size of the square tiles to extract from images in a non-overlapping way.
            Each tile extarcted from the image will contain (tile_size ** 2) number of pixels.
        
        channel_idx: int
            Since the masks in this PANDA dataset contain 3 channels. This constant indicates
            the location of the official, 2 by 2 mask in the 3 channel mask tensor. Default
            value is 0. 
        
        min_thresh: float
            This constant sets a criteria for tile qualification. A tile is only qualified to
            be positively filtered for the dataset if it containts signal (% of foreground of all pixels)
            that is more than or equal to min_thresh.
        
        transform: Optional[None, function]
            This object stores None or a function. If it is a function, this function is applied
            to the data before being returned by the get item dunder method. This transform could
            contain a normalization function or it could contain a series of augmentation steps
            applied before returning an image. 
        
        norm_func: Optional[None, function]
            Function to normalize a tile depending on certain pre-defined 
            min-max statistics (for min max normalization) or z-score normalization
            depending on pre-defined statistics for the mean and standard deviation. 
        
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.root_dir = root_dir
        self.mask_suffix = mask_suffix
        self.imfile_ext = imfile_ext
        self.maskfile_ext = maskfile_ext
        self.transform = transform
        self.norm_func = norm_func
        self.tile_size = tile_size
        self.full_mask_ext = self.mask_suffix + self.maskfile_ext
        self.channel_idx = channel_idx
        self.image_files = list(filter(lambda file: file != '.DS_Store', sorted(os.listdir(self.image_dir))))
        self.mask_files = list(filter(lambda file: file != '.DS_Store', sorted(os.listdir(self.mask_dir))))
        self.n_tiles_processed = 0
        self.n_qualifying_tiles = 0
        self.min_thresh = min_thresh
        self.imtile_coords = self.get_imtile_coords()
        
    def extract_raw_regid(self,
                          filename: str,
                          ext: str) -> str:
        """
        This function receives a filename which a specific
        extension like "_mask.tiff" or ".tiff", then it processes
        and returns the part of the filename up until but excluding
        the extension. 
        
        Parameters
        -----------
        filename: str
            The filename, with the name and extension as a suffix
        
        ext: str
            The extension
        
        Returns
        ----------
        The filename up until but excluding the extension
        
        """
        i = 1
        while filename[-i:] != ext:
            i += 1
        return filename[0:len(filename)-i]
    
    def tile_qualifier(self,
                       tile_mask: np.ndarray):
        """
        A tile qualifier function. This function
        receives a mask extracted from a tile. The mask
        can be binary or it could be multi-class,
        representing presence of multiple different classes
        indicated by integer values greater than or equal to 1. 
        The background (non-signal) region should be represented
        by 0. This function checks if the mask contains a minimum
        threshold of signal, as represented by the number of non-zero
        values as a percentage of the total number of pixels in the
        tile. 
        
        Parameters
        -----------
        tile_mask: np.ndarray
            A tile (square or rectangular) as a numpy array,
            containing integer values greater than or equal to 0.
        
        Returns
        ---------
        bool
            Whether the tile qualifies or not.
            Whether its signal ratio meets the minimum
            threshold requirement for it to qualify
            to be included in the dataset. 
            
        """
        
        signal_ratio = (tile_mask > 0).sum()/tile_mask.size
        
        return signal_ratio >= self.min_thresh
    
    def extract_nonoverlapping_tile_coords(self,
                                           mask: np.ndarray):
        
        """
        This function extracts nonoverlapping tile coordinates
        from a mask according to the tile_size. It collects
        the coordinates of the top left corner of each tile and 
        then returns it. 
        
        Paramters
        ---------
        mask: np.ndarray
            A mask of arbitrary shape. Containing 3 or more
            channels, but only the 0th channel is used as the
            official, signal indicating channel for extracting 
            tile coordinates.
        
        Returns
        --------
        List[Tuple[int, int]]
            A list of coordinates stored in tuples of size 2.
            These are coordinates of tiles that qualify according
            to the tile qualifier function. 
        """
        
        mask = mask[:,:,self.channel_idx]
        assert len(mask.shape) == 2
        R, C = mask.shape
        tile_coords = []
        for i in range(0, R, self.tile_size):
            for j in range(0, C, self.tile_size):
                try:
                    tile = mask[i : i + self.tile_size, j : j + self.tile_size]
                    self.n_tiles_processed += 1
                    if tile.shape == (self.tile_size, self.tile_size) and self.tile_qualifier(tile):
                        tile_coords.append((i, j))
                        self.n_qualifying_tiles += 1
                except IndexError:
                    print(f"Skipping tile at position ({i}, {j}) because it is out of bounds.")
                        
        return tile_coords
    
    def get_imtile_coords(self):
        
        """
        Function to process the images and return a list of 
        3-sized tuples that contain the name of the image file
        along with coordinates of qualifying tiles in them.
        The image file names can be repeated in tuples, but not
        the coordinate of the tiles.
        
        Parameters
        -----------
        None, all objects used are instantiated in the constructor.
        
        Returns
        -----------
        List[Tuple[str, in, int]]
            A list of tuples. The first value in the tuple is
            the name of the image file. The second and third
            values in the tuples are coordinates of one of the
            qualifying tiles in the image.
            
        """
        
        imtile_coords = []
        
        for imfile in self.image_files:
            regid = self.extract_raw_regid(imfile, self.imfile_ext)
            mask_file = regid + self.full_mask_ext
            mask = tifffile.imread(os.path.join(self.mask_dir, mask_file))
            tile_coords = self.extract_nonoverlapping_tile_coords(mask)
            imtile_coord = list(map(lambda tup : (imfile,) + tup, tile_coords))
            imtile_coords.extend(imtile_coord)
            
        self.tile_qualifying_ratio = self.n_qualifying_tiles / self.n_tiles_processed
        print(f"Total number of tiles processed: {self.n_tiles_processed}")
        print(f"Tile qualifying ratio: {self.tile_qualifying_ratio}")
        print(f"Average number of tiles processed per image or mask: {self.n_tiles_processed/len(self.image_files)}")
        print(f"Average number of tiles qualifying per image or mask: {self.n_qualifying_tiles/len(self.image_files)}")
        
        imfiles = set([t[0] for t in imtile_coords])
        assert imfiles == set(self.image_files)
        i_vals = sorted(list(set([t[1] for t in imtile_coords])))
        j_vals = sorted(list(set([t[2] for t in imtile_coords])))
        
        for i in range(len(i_vals)-1):
            assert i_vals[i + 1] - i_vals[i] == self.tile_size
        
        for j in range(len(j_vals)-1):
            assert j_vals[j + 1] - j_vals[j] == self.tile_size
            
        return imtile_coords
        
    def __len__(self):
        
        """
        Dunder method to return the length of the
        dataset. The total length of the dataset 
        is the total number of tiles over all 
        of the images. Some images may provide more
        tiles to the dataset depending on their signal
        content, while other images may provide less
        tiles if they are weak in signal.
        """
        
        return len(self.imtile_coords)
    
    def __getitem__(self, idx):
        
        """
        Function for getting a tile from the dataset.
        Given an index, it finds a tuple in the imtile_coords
        and unpacks it to retrieve the image file and the coordinates
        of one of the qualifying tiles in the image. 
        """
        
        image_file, tile_i, tile_j = self.imtile_coords[idx]
        image = tifffile.imread(os.path.join(self.image_dir, image_file))
        tile = image[tile_i: tile_i + self.tile_size, tile_j: tile_j + self.tile_size, :]
        
        if self.norm_func:
            tile = torch.tensor(tile)
            tile = tile.unsqueeze(0).permute(0, 3, 1, 2)
            tile = self.norm_func(tile)
            tile = tile.squeeze(0).permute(1,2,0)
            
        if self.transform:
            tile = self.transform(tile)
            
        return tile