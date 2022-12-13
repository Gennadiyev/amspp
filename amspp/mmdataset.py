"""amspp/mmdataset.py

Implements the MultiModalDataset class and Schema class / decorator.

@author: Yikun JI
"""

from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
from torch import Tensor as TorchTensor
from torch.nn import Module as TorchModuleOrTransform # torchvision.transforms are also nn.Modules
from numpy import ndarray as NumpyArray
from torch.utils.data import TorchDataset
from snntorch.spikevision.neuromorphic_dataset import NeuromorphicDataset # Also inherits from torch.utils.data.Dataset
from loguru import logger

class Schema(Callable):
    """A Schema is a data structure that transforms from raw data or tensors to a specific format. It works
    similarly to the C{collate_fn} argument in PyTorch's C{DataLoader} class. The main difference is that a
    schema is not a function, but a class that can be instantiated and called.

    C{Schema} class that cooks data from multiple tensors into one according to a user-defined format.
    
    A func decorated by schema takes a dictionary of inputs from multiple source datasets and returns anything
    that the user wants for downstream tasks. The input dictionary is indexed by the name of the dataset.
    
    @cvar DEFAULT_SCHEMA: the default schema. It does nothing.
    """
    DEFAULT_SCHEMA = lambda x: x # Default schema: do nothing
    
    def __init__(self, callable: Union[None, TorchModuleOrTransform, Callable]=None, /, debug_mode=False):
        """Initializes the schema.

        @param callable: the callable function that cooks data. (default: lambda x: x)
        @param debug_mode: whether to print debug messages.
        """
        self._callable = callable if callable is not None else self.DEFAULT_SCHEMA
        self._debug_mode = debug_mode

    @logger.catch
    def __call__(self, data: Dict[str, Any]) -> Any:
        """Calls the schema to cook data.

        @param data: the data to cook.
        @return: the cooked data that can be anything.
        """
        if self._debug_mode:
            # Log the data
            logger.debug(f"Schema is called with dict with the following data:")
            for k, v in data.items():
                try:
                    if isinstance(v, TorchTensor):
                        logger.debug("From {}: shape={}, dtype={}", k, v.shape, v.dtype)
                    elif isinstance(v, NumpyArray):
                        logger.debug("From {}: shape={}, dtype={}", k, v.shape, v.dtype)
                    else:
                        logger.debug("From {}: {}", k, v)
                except Exception as e:
                    logger.warning("From {}: (cannot print data due to error: {})", k, e)
        return self._callable(data)

def schema(func):
    """A decorator that converts a function into a C{Schema} object.
    """
    return Schema(func)
  
class MultiModalDataset(TorchDataset):
    """Multi-modal dataset. A multi-modal dataset takes multiple C{torch.utils.data.Dataset} (or any iterable)
    objects and create a formatted output for downstream tasks.

    @ivar _datasets: a dictionary of datasets.
    @ivar _schema: the schema of the dataset. See L{amspp.mmdataset.Schema}.
    @ivar _len: the length of the dataset. If None, the length is not set. Determined and set by the first
    registered dataset. Once set, dataset length must match the length of the first registered dataset, other-
    wise an error will be raised.
    """
    def __init__(self):
        """Initializes the multi-modal dataset with an empty dictionary of datasets and an empty (pass-through)
        schema (see L{amspp.mmdataset.Schema}).
        """
        self._datasets = {}
        self._schema = Schema()
        self._len = None
        
    # The MultiModalDataset-specific methods
    
    @logger.catch
    def register_dataset(self, name: str, dataset: Union[TorchDataset, NeuromorphicDataset, Iterable]):
        """Register a dataset.

        @param dataset: the dataset to register.
        @param name: the name of the dataset.
        """
        # Avoid dataset key collision
        if name in self._datasets:
            raise ValueError(f"Dataset {name} already exists.")
        # Add the dataset
        self._datasets[name] = dataset
        # Get the length of the dataset
        len_dataset = len(dataset)
        if self._len is None:
            logger.info("Setting the length of the MultiModalDataset to {}.", len_dataset)
            self._len = len_dataset
        elif self._len != len_dataset:
            raise ValueError(f"Length of dataset {name} ({len_dataset}) does not match the length of the MultiModalDataset ({self._len}).")
    
    def get_datasets(self) -> Dict[str, Union[TorchDataset, NeuromorphicDataset, Iterable]]:
        """Get the datasets.

        @return: the datasets.
        """
        return self._datasets

    def get_dataset(self, name: str) -> Union[TorchDataset, NeuromorphicDataset, Iterable]:
        """Get a dataset.

        @param name: the name of the dataset.
        @return: the dataset.
        """
        return self._datasets[name]

    def set_schema(self, schema: Schema):
        """Set the schema of the dataset.

        @param schema: the schema of the dataset.
        """
        self._schema = schema
        
    # The methods that are required by torch.utils.data.Dataset
    
    def __len__(self):
        return self._len if self._len is not None else 0

    @logger.catch
    def __getitem__(self, idx):
        # Gather the data from each dataset
        data = {
            dataset_name: dataset[idx]
            for dataset_name, dataset in self._datasets.items()
        }
        # Cook the data using the schema
        return self._schema(data)
    