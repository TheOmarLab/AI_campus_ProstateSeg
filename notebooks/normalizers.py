import torch

class ChannelWiseMinMaxNormalizer:
    """
    Channel-wise min max
    normalizer
    """
    def __init__(self,
                 min_values = None,
                 max_values = None):
        
        self.min_values = min_values
        self.max_values = max_values

    def fit(self, 
            dataset: torch.tensor):
        """
        Calculate dataset-wide per-channel
        minimum and maximum values 
        """
        self.min_values = torch.amin(dataset, dim=(0, 2, 3))
        self.max_values = torch.amax(dataset, dim=(0, 2, 3))

    def transform(self, 
                  to_transform: torch.tensor) -> torch.tensor:
        """
        Transform dataset by 
        (value - min)/(max - min)
        using per-channel
        min and max statistics
        """
        if self.min_values is None or self.max_values is None:
            raise ValueError("Fit min and max values pre-transformation or input external values")

        normalized_dataset = (to_transform - self.min_values[None, :, None, None]) / (self.max_values[None, :, None, None] - self.min_values[None, :, None, None])

        return normalized_dataset
        
    
    def fit_transform(self, 
                      to_transform: torch.tensor) -> torch.tensor:
        """
        Fit and Transform dataset by 
        (value - min)/(max - min)
        using per-channel
        min and max statistics
        """
        self.fit(to_transform)
        
        return self.transform(to_transform)
    
    