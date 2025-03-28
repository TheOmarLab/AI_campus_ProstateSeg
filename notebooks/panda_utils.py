import numpy as np
import os
import tifffile
import matplotlib.pyplot as plt

from typing import List, Dict

def verify_pair_match(image_dir: str,
                      mask_dir: str,
                      mask_ext: str = "_mask.tiff") -> bool:
    """
    Verifies if every image file in the provided image directory has a corresponding mask file 
    in the mask directory. The mask file is expected to have the same name as the image file 
    but with a specified extension (default is '_mask.tiff').

    Parameters:
    ----------
    image_dir : str
        The directory containing image files.
    
    mask_dir : str
        The directory containing mask files.
    
    mask_ext : str, optional (default is "_mask.tiff")
        The extension of the mask files. 
    
    Returns:
    -------
    bool
        Returns True if every image file has a corresponding mask file, otherwise returns False. 
        
    """
    
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    for file in image_files:
        if file != '.DS_Store':  
            if file[:file.find(".")] + mask_ext not in mask_files:
                return False  
    
    return True

def verify_dim_match(image_dir: str,
                     mask_dir: str,
                     root_folder: str,
                     image_subfolder: str,
                     mask_subfolder: str,
                     dims: List[int] = [0,1,2]) -> bool:
    """
    Verifies that each image and mask pair have the same dimensionality along the specified
    list of dimension indexes in dims. 
    
    Parameters:
    ----------
    image_dir : str
        The directory containing image files.
    
    mask_dir : str
        The directory containing mask files.
    
    root_folder: str
        The root folder containing the image and mask folders. 
    
    image_subfolder: str
        The subfolder containing the image files.
    
    mask_subfolder: str
        The subfolder containing the mask files. 
    
    dims: List[int]
        List of integers specifying dimensions and their values to check for match. 
        Default: [0,1,2]: each image and mask pair will be checked for a match at dimension indices [0,1,2]
                          
        
    Returns:
    -------
    bool
        Returns True if the dimensionality of each image and mask pair matches along all of the 
        specified dimension indices, False otherwise
        
    """
    
    image_files = list(filter(lambda file: file != '.DS_Store', sorted(os.listdir(image_dir))))
    mask_files = list(filter(lambda file: file != '.DS_Store', sorted(os.listdir(mask_dir))))
    
    assert len(image_files) == len(mask_files)
    dims = list(set(dims))
    
    n_files = len(image_files)
    
    for i in range(n_files):
        img = tifffile.imread(os.path.join(root_folder, image_subfolder, image_files[i]))
        mask = tifffile.imread(os.path.join(root_folder, mask_subfolder, mask_files[i]))
        
        img_dim = img.shape
        mask_dim = mask.shape
        
        match = all([img_dim[d] == mask_dim[d] for d in dims])
        
        if not match:
            return False
        
    return True

def validate_three_channel_mask(mask: np.ndarray) -> bool:
    
    """
    Since the PANDA dataset masks have 3 channels, this is a basic
    function to validate an input mask. Only one of the three channels
    of the mask could be effectively used as the official mask of the
    corresponding image.
    
    Parameters:
    ----------
    mask : np.ndarray
        Input three-channel mask with integer labels possibly ranging from 0-5
        across one or all channels
    
    For the Radboud study of the PANDA dataset:
        0: background (non tissue) or unknown
        1: stroma (connective tissue, non-epithelium tissue)
        2: healthy (benign) epithelium
        3: cancerous epithelium (Gleason 3)
        4: cancerous epithelium (Gleason 4)
        5: cancerous epithelium (Gleason 5)
    
    Returns:
    -------
    bool
        Returns True if the three-channel mask is valid, False otherwise.
        Validity criteria defined in the following cases. 
    """
    
    # Verify that the mask to validate contains 3 channels
    assert isinstance(mask, np.ndarray)
    assert mask.shape[2] == 3
    
    # Extract first, second, and third channels from the mask
    first = mask[:, :, 0]
    second = mask[:, :, 1]
    third = mask[:, :, 2]
    
    first_sum = first.sum()
    second_sum = second.sum()
    third_sum = third.sum()
    
    assert first_sum >= 0 and second_sum >= 0 and third_sum >= 0
        
    # Case 1: The mask indicates all background (contains only 0s)
    # If all entries in all channels are 0, all channels will sum to 0
    
    if (first_sum + second_sum + third_sum) == 0:
        return True
    
    # Case 2: The mask indicates non-zero signal and only one of the three channels
    # is designated to express the non-zero signal
    # Exactly one of the channels will sum to a number greater than 0
    # While the other two channels will each sum to 0
    
    if first_sum > 0 and (second_sum + third_sum == 0):
        return True
    if second_sum > 0 and (first_sum + third_sum == 0):
        return True
    if third_sum > 0 and (first_sum + second_sum == 0):
        return True
    
    # Case 3: The mask indicates non-zero signal and two channels are designated to
    # express that non-zero signal. So two of the three channels must sum to a number 
    # greater than 0 and both of them should be identical, while the remaining channel
    # should sum to 0
    
    if third_sum == 0 and first_sum > 0 and second_sum > 0 and np.all(first == second):
        return True
    if second_sum == 0 and first_sum > 0 and third_sum > 0 and np.all(first == third):
        return True
    if first_sum == 0 and second_sum > 0 and third_sum > 0 and np.all(second == third):
        return True
    
    # Case 4: 
    # The mask indicates non-zero signal and all of the three channels are designated
    # to express that non-zero signal. So each of the three channels must sum to a number
    # greater than 0 and all of them must be identical
    
    if (first_sum == second_sum == third_sum) and np.all(first == second) and np.all(second == third):
        return True
    
    return False

def validate_mask_dir(mask_dir: str,
                      root_folder: str,
                      mask_subfolder: str) -> bool:
    
    """
    Validate all masks in mask directory using
    validate_three_channel_mask
    
    Parameters:
    ----------
    mask_dir : str
        The directory containing mask files.
    
    root_folder : str
        The directory containing the mask folder.
    
    mask_subfolder : str
        The folder containing the masks. 
        
    Returns:
    -------
    bool
        Returns True if all masks in directory are valid
        according to validate_three_channel_mask
        
    """
    
    mask_files = list(filter(lambda file: file != '.DS_Store', sorted(os.listdir(mask_dir))))
    
    for file in mask_files:
        mask = tifffile.imread(os.path.join(root_folder, mask_subfolder, file))
        if not validate_three_channel_mask(mask):
            return False
    return True

def characterize_three_channel_mask(mask: np.ndarray) -> str:
    
    """
    Since the PANDA dataset masks have 3 channels, this is a function to
    validate and characterize an input mask. Only one of the three channels
    of the mask could be effectively used as the official mask of the corresponding
    image. 
    
    Parameters:
    ----------
    mask: np.ndarray
        Input three-channel mask with integer labels possibly ranging from 0-5
        across one or all channels
    
    For the Radboud study of the PANDA dataset:
        0: background (non tissue) or unknown
        1: stroma (connective tissue, non-epithelium tissue)
        2: healthy (benign) epithelium
        3: cancerous epithelium (Gleason 3)
        4: cancerous epithelium (Gleason 4)
        5: cancerous epithelium (Gleason 5)
    
    Returns:
    -------
    str
        Characterizes the mask according to the following possible cases
    """
    
    # Verify that the mask to validate contains 3 channels
    assert isinstance(mask, np.ndarray)
    assert mask.shape[2] == 3
    
    # Extract first, second, and third channels from the mask
    first = mask[:, :, 0]
    second = mask[:, :, 1]
    third = mask[:, :, 2]
    
    first_sum = first.sum()
    second_sum = second.sum()
    third_sum = third.sum()
    
    # Ensure that the mask only contains 0s or positive integers
    assert first_sum >= 0 and second_sum >= 0 and third_sum >= 0
    
    # Case 1: The mask indicates all background (contains only 0s)
    # If all entries in all channels are 0, all channels will sum to 0
    
    if (first_sum + second_sum + third_sum) == 0:
        return "Case 1: The mask indicates all background, any of channels 0, 1, or 2 can be used"
    
    # Case 2: The mask indicates non-zero signal and only one of the three channels
    # is designated to express the non-zero signal
    # Exactly one of the channels will sum to a number greater than 0
    # While the other two channels will each sum to 0
    
    if first_sum > 0 and (second_sum + third_sum == 0):
        return "Case 2: The mask indicates non-zero signal, only channel 0 is designated to express it"
    if second_sum > 0 and (first_sum + third_sum == 0):
        return "Case 2: The mask indicates non-zero signal, only channel 1 is designated to express it"
    if third_sum > 0 and (first_sum + second_sum == 0):
        return "Case 2: The mask indicates non-zero signal, only channel 2 is designated to express it"
    
    # Case 3: The mask indicates non-zero signal and two channels are designated to
    # express that non-zero signal. So two of the three channels must sum to a number 
    # greater than 0 and both of them should be identical, while the remaining channel
    # should sum to 0
    
    if third_sum == 0 and first_sum > 0 and second_sum > 0 and np.all(first == second):
        return "Case 3: The mask indicates non-zero signal, channels 0 and 1 are designated to express it, they are identical and can be used interchangeably"
    if second_sum == 0 and first_sum > 0 and third_sum > 0 and np.all(first == third):
        return "Case 3: The mask indicates non-zero signal, channels 0 and 2 are designated to express it, they are identical and can be used interchangeably"
    if first_sum == 0 and second_sum > 0 and third_sum > 0 and np.all(second == third):
        return "Case 3: The mask indicates non-zero signal, channels 1 and 2 are designated to express it, they are identical and can be used interchangeably"
    
    # Case 4: 
    # The mask indicates non-zero signal and all of the three channels are designated
    # to express that non-zero signal. So each of the three channels must sum to a number
    # greater than 0 and all of them must be identical
    
    if all([first_sum > 0, second_sum > 0, third_sum > 0]) and (first_sum == second_sum == third_sum) and np.all(first == second) and np.all(second == third):
        return "Case 4: The mask indicates non-zero signal, and each of channels 0, 1, and 2 express it, they are all identical and any of them can be used"
    
    return "None of the cases hold True, inspect mask"

def mask_signal_ratio(mask: np.ndarray,
                      channel_idx: int = 0):
    
    """
    Calculate the number of pixels in mask that
    are more than 0, and hence, indicate a signal
    as a percentage of the total number of pixels
    
    Parameters:
    ----------
    
    mask: np.ndarray
        Input mask
    
    channel_idx: int (default = 0)
        In case of a three-channel mask, the channel id that can be used
        as the official mask 
    
    Returns:
    ---------
    
    float
        The percentage of pixels in the mask
        that contain a signal, i.e. a value 
        greater than 0
    """
    
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = mask[:, :, channel_idx]
        
    assert len(mask.shape) == 2
    
    return (mask > 0).sum()/mask.size

def enlist_class_presence(mask: np.ndarray,
                          channel_idx: int = 0) -> np.ndarray:
    """
    Enlist the classes present in the mask, 
    i.e. the types of unique integers present
    
    Parameters:
    ----------
    mask: np.ndarray
        Input mask
    
    channel_idx: int (default = 0)
        In case of a three-channel mask, the channel id that can be used
        as the official mask
    """
    
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = mask[:, :, channel_idx]
    
    assert len(mask.shape) == 2
    
    return np.unique(mask)

def summarize_class_distribution(mask: np.ndarray,
                                 channel_idx: int = 0) -> Dict[int, float]:
    """
    Summarize the percentage distribution of 
    different classes present in the mask. 
    
    Parameters:
    ----------
    mask: np.ndarray
        Input mask
    
    channel_idx: int (default = 0)
        In case of a three-channel mask, the channel id that
        can be used as the official mask
    
    Returns:
    --------
    Dict[int, float]
        A dictionary mapping each of the present class labels
        to its percentage frequency in the mask
    """
    
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = mask[:, :, channel_idx]
    
    assert len(mask.shape) == 2
    
    dist_dict = dict()
    avail_classes = np.unique(mask)
    
    for c in avail_classes:
        dist_dict[c] = round((mask == c).sum()/mask.size, 2)
    
    return dist_dict

def dict_to_histogram(class_dict: dict,
                      whole_classes = [0, 1, 2, 3, 4, 5]):
    """
    Plot histogram of distribution of classes
    """
    for c in whole_classes:
        if c not in class_dict:
            class_dict[c] = 0.0
    
    keys = list(class_dict.keys())
    values = list(class_dict.values())
    plt.bar(keys, values)
    plt.xlabel("Class", fontsize = 20)
    plt.ylabel("Percentage Composition in Mask")
    plt.title("Distribution of Classes in Mask")
    plt.show()
    
def get_class_coordinates(mask: np.ndarray,
                          channel_idx: int = 0) -> Dict[int, List[tuple[int]]]:
    """
    For each class label (integer index) present in 
    the mask, get a list of all its coordinates in the
    format of a dictionary that maps the class label to 
    the list of coordinates in mask where it is present
    
    Parameters:
    -----------
    mask: np.ndarray
        Input mask
    
    channel_idx: int (default = 0)
        In case of a three-channel mask, the channel id
        that can be used as the official mask
    
    Returns:
    --------
    Dict:
        A dictionary mapping each class label to a list of 
        its coordinates
    """
    
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = mask[:, :, channel_idx]
    
    assert len(mask.shape) == 2
    
    avail_classes = np.unique(mask)
    coord_dict = dict()
    
    for c in avail_classes:
        coords = np.where(mask == c)
        assert len(coords) == 2
        coord_dict[c] = [(i, j) for (i, j) in zip(coords[0], coords[1])]
    
    return coord_dict

id_to_class = {0: "background (non tissue) or unknown",
               1: "stroma (connective tissue, non-epithelium tissue)",
               2: "healthy (benign) epithelium",
               3: "cancerous epithelium (Gleason 3)",
               4: "cancerous epithelium (Gleason 4)",
               5: "cancerous epithelium (Gleason 5)"}

def plot_tile_by_first_class_occurance(image: np.ndarray,
                                       mask: np.ndarray,
                                       tile_size: int = 224,
                                       pivot_class: int = 0,
                                       channel_idx: int = 0):
    
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = mask[:, :, channel_idx]
        
    assert len(mask.shape) == 2
    
    coords = np.where(mask == pivot_class)
    
    x = coords[0][0]
    y = coords[1][0]
    
    tile_image = image[x: x + tile_size, y: y + tile_size]
    tile_mask = mask[x: x + tile_size, y: y + tile_size]
    
    class_dict = summarize_class_distribution(tile_mask)
    dict_to_histogram(class_dict)
    
    plt.imshow(tile_image)
    
    
    for k in class_dict.keys():
        print(f"Percentage of {k}, {id_to_class[k]}: {class_dict[k]}")

def summarize_channel_stats(img: np.ndarray):
    
    """
    Summarize minimum and maximum value for each
    of the three channels in a three-channel image
    
    Parameters
    -----------
    img: np.ndarray (H, W, 3)
        Input image
    
    Returns
    -----------
    Prints channel-wise min and max statistics
    """
    
    assert len(img.shape) == 3 and img.shape[2] == 3
    
    print(f"Channel 1 mean: {img[:, :, 0].mean()}")
    print(f"Channel 2 mean: {img[:, :, 1].mean()}")
    print(f"Channel 3 mean: {img[:, :, 2].mean()}")
    
    print(f"Channel 1 std: {img[:, :, 0].std()}")
    print(f"Channel 2 std: {img[:, :, 1].std()}")
    print(f"Channel 3 std: {img[:, :, 2].std()}")
    
    print(f"Channel 1 min: {img[:, :, 0].min()}")
    print(f"Channel 2 min: {img[:, :, 1].min()}")
    print(f"Channel 3 min: {img[:, :, 2].min()}")
    
    print(f"Channel 1 max: {img[:, :, 0].max()}")
    print(f"Channel 2 max: {img[:, :, 1].max()}")
    print(f"Channel 3 max: {img[:, :, 2].max()}")
    

