from torch.utils.data import Dataset

class ChexInstructDataset(Dataset):
    def __init__(self, data):
        """
        Initializes the ChexInstructDataset with the provided data.

        Args:
            data (list): A list of dictionaries containing the dataset samples.
        """
        self.data = data

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a sample from the dataset at the specified index."""
        return self.data[idx]




