# Data[#](data.html#data "Link to this heading")

## Generic Interfaces[#](data.html#generic-interfaces "Link to this heading")

### Dataset[#](data.html#dataset "Link to this heading")

_class _monai.data.Dataset(_data_ , _transform
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L56-L108)[#](data.html#monai.data.Dataset
"Link to this definition")

    

A generic dataset with a length property and an optional callable data
transform when fetching a data sample. If passing slicing indices, will return
a PyTorch Subset, for example: data: Subset = dataset[1:4], for more details,
please check:
<https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset>

For example, typical input data can be a list of dictionaries:

    
    
    [{                            {                            {
         'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
         'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
         'extra': 123                 'extra': 456                 'extra': 789
     },                           },                           }]
    

__getitem__(_index_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L96-L108)[#](data.html#monai.data.Dataset.__getitem__
"Link to this definition")

    

Returns a Subset if index is a slice or Sequence, a data item otherwise.

__init__(_data_ , _transform =None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L72-L84)[#](data.html#monai.data.Dataset.__init__
"Link to this definition")

    

Parameters:

    

  * **data** – input data to load and transform to generate dataset for model.

  * **transform** – a callable, sequence of callables or None. If transform is not

  * **instance** (_a Compose_)

  * **Sequences** (_it will be wrapped in a Compose instance._)

  * **passed** (_of callables are applied in order and if None is_)

  * **is.** (_the data is returned as_)

### IterableDataset[#](data.html#iterabledataset "Link to this heading")

_class _monai.data.IterableDataset(_data_ , _transform
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/iterable_dataset.py#L28-L63)[#](data.html#monai.data.IterableDataset
"Link to this definition")

    

A generic dataset for iterable data source and an optional callable data
transform when fetching a data sample. Inherit from PyTorch IterableDataset:
<https://pytorch.org/docs/stable/data.html?highlight=iterabledataset#torch.utils.data.IterableDataset>.
For example, typical input data can be web data stream which can support
multi-process access.

To accelerate the loading process, it can support multi-processing based on
PyTorch DataLoader workers, every process executes transforms on part of every
loaded data. Note that the order of output data may not match data source in
multi-processing mode. And each worker process will have a different copy of
the dataset object, need to guarantee process-safe from data source or
DataLoader.

__init__(_data_ , _transform =None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/iterable_dataset.py#L43-L51)[#](data.html#monai.data.IterableDataset.__init__
"Link to this definition")

    

Parameters:

    

  * **data** – input data source to load and transform to generate dataset for model.

  * **transform** – a callable data transform on input data.

### DatasetFunc[#](data.html#datasetfunc "Link to this heading")

_class _monai.data.DatasetFunc(_data_ , _func_ , _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L111-L158)[#](data.html#monai.data.DatasetFunc
"Link to this definition")

    

Execute function on the input dataset and leverage the output to act as a new
Dataset. It can be used to load / fetch the basic dataset items, like the list
of image, label paths. Or chain together to execute more complicated logic,
like partition_dataset, resample_datalist, etc. The data arg of Dataset will
be applied to the first arg of callable func. Usage example:

    
    
    data_list = DatasetFunc(
        data="path to file",
        func=monai.data.load_decathlon_datalist,
        data_list_key="validation",
        base_dir="path to base dir",
    )
    # partition dataset for every rank
    data_partition = DatasetFunc(
        data=data_list,
        func=lambda **kwargs: monai.data.partition_dataset(**kwargs)[torch.distributed.get_rank()],
        num_partitions=torch.distributed.get_world_size(),
    )
    dataset = Dataset(data=data_partition, transform=transforms)
    

Parameters:

    

  * **data** (`Any`) – input data for the func to process, will apply to func as the first arg.

  * **func** (`Callable`) – callable function to generate dataset items.

  * **kwargs** – other arguments for the func except for the first arg.

reset(_data =None_, _func =None_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L147-L158)[#](data.html#monai.data.DatasetFunc.reset
"Link to this definition")

    

Reset the dataset items with specified func.

Parameters:

    

  * **data** – if not None, execute func on it, default to self.src.

  * **func** – if not None, execute the func with specified kwargs, default to self.func.

  * **kwargs** – other arguments for the func except for the first arg.

### ShuffleBuffer[#](data.html#shufflebuffer "Link to this heading")

_class _monai.data.ShuffleBuffer(_data_ , _transform =None_, _buffer_size
=512_, _seed =0_, _epochs =1_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/iterable_dataset.py#L66-L135)[#](data.html#monai.data.ShuffleBuffer
"Link to this definition")

    

Extend the IterableDataset with a buffer and randomly pop items.

Parameters:

    

  * **data** – input data source to load and transform to generate dataset for model.

  * **transform** – a callable data transform on input data.

  * **buffer_size** (`int`) – size of the buffer to store items and randomly pop, default to 512.

  * **seed** (`int`) – random seed to initialize the random state of all workers, set seed += 1 in every iter() call, refer to the PyTorch idea: [pytorch/pytorch](https://github.com/pytorch/pytorch/blob/v1.10.0/torch/utils/data/distributed.py#L98).

  * **epochs** (`int`) – number of epochs to iterate over the dataset, default to 1, -1 means infinite epochs.

Note

Both `monai.data.DataLoader` and `torch.utils.data.DataLoader` do not seed
this class (as a subclass of `IterableDataset`) at run time.
`persistent_workers=True` flag (and pytorch>1.8) is therefore required for
multiple epochs of loading when `num_workers>0`. For example:

    
    
    import monai
    
    def run():
        dss = monai.data.ShuffleBuffer([1, 2, 3, 4], buffer_size=30, seed=42)
    
        dataloader = monai.data.DataLoader(
            dss, batch_size=1, num_workers=2, persistent_workers=True)
        for epoch in range(3):
            for item in dataloader:
                print(f"epoch: {epoch} item: {item}.")
    
    if __name__ == '__main__':
        run()
    

generate_item()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/iterable_dataset.py#L114-L122)[#](data.html#monai.data.ShuffleBuffer.generate_item
"Link to this definition")

    

Fill a buffer list up to self.size, then generate randomly popped items.

randomize(_size_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/iterable_dataset.py#L134-L135)[#](data.html#monai.data.ShuffleBuffer.randomize
"Link to this definition")

    

Within this method, `self.R` should be used, instead of np.random, to
introduce random factors.

all `self.R` calls happen here so that we have a better chance to identify
errors of sync the random state.

This method can generate the random factors based on properties of the input
data.

Raises:

    

**NotImplementedError** – When the subclass does not override this method.

Return type:

    

`None`

randomized_pop(_buffer_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/iterable_dataset.py#L107-L112)[#](data.html#monai.data.ShuffleBuffer.randomized_pop
"Link to this definition")

    

Return the item at a randomized location self._idx in buffer.

### CSVIterableDataset[#](data.html#csviterabledataset "Link to this heading")

_class _monai.data.CSVIterableDataset(_src_ , _chunksize =1000_, _buffer_size
=None_, _col_names =None_, _col_types =None_, _col_groups =None_, _transform
=None_, _shuffle =False_, _seed =0_, _kwargs_read_csv =None_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/iterable_dataset.py#L138-L281)[#](data.html#monai.data.CSVIterableDataset
"Link to this definition")

    

Iterable dataset to load CSV files and generate dictionary data. It is
particularly useful when data come from a stream, inherits from PyTorch
IterableDataset:
<https://pytorch.org/docs/stable/data.html?highlight=iterabledataset#torch.utils.data.IterableDataset>.

It also can be helpful when loading extremely big CSV files that can’t read
into memory directly, just treat the big CSV file as stream input, call
reset() of CSVIterableDataset for every epoch. Note that as a stream input, it
can’t get the length of dataset.

To effectively shuffle the data in the big dataset, users can set a big buffer
to continuously store the loaded data, then randomly pick data from the buffer
for following tasks.

To accelerate the loading process, it can support multi-processing based on
PyTorch DataLoader workers, every process executes transforms on part of every
loaded data. Note: the order of output data may not match data source in
multi-processing mode.

It can load data from multiple CSV files and join the tables with additional
kwargs arg. Support to only load specific columns. And it can also group
several loaded columns to generate a new column, for example, set
col_groups={“meta”: [“meta_0”, “meta_1”, “meta_2”]}, output can be:

    
    
    [
        {"image": "./image0.nii", "meta_0": 11, "meta_1": 12, "meta_2": 13, "meta": [11, 12, 13]},
        {"image": "./image1.nii", "meta_0": 21, "meta_1": 22, "meta_2": 23, "meta": [21, 22, 23]},
    ]
    

Parameters:

    

  * **src** – if provided the filename of CSV file, it can be a str, URL, path object or file-like object to load. also support to provide iter for stream input directly, will skip loading from filename. if provided a list of filenames or iters, it will join the tables.

  * **chunksize** – rows of a chunk when loading iterable data from CSV files, default to 1000. more details: <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html>.

  * **buffer_size** – size of the buffer to store the loaded chunks, if None, set to 2 x chunksize.

  * **col_names** – names of the expected columns to load. if None, load all the columns.

  * **col_types** – 

type and default value to convert the loaded columns, if None, use original
data. it should be a dictionary, every item maps to an expected column, the
key is the column name and the value is None or a dictionary to define the
default value and data type. the supported keys in dictionary are: [“type”,
“default”]. for example:

    
        col_types = {
        "subject_id": {"type": str},
        "label": {"type": int, "default": 0},
        "ehr_0": {"type": float, "default": 0.0},
        "ehr_1": {"type": float, "default": 0.0},
        "image": {"type": str, "default": None},
    }
    

  * **col_groups** – args to group the loaded columns to generate a new column, it should be a dictionary, every item maps to a group, the key will be the new column name, the value is the names of columns to combine. for example: col_groups={“ehr”: [f”ehr_{i}” for i in range(10)], “meta”: [“meta_1”, “meta_2”]}

  * **transform** – transform to apply on the loaded items of a dictionary data.

  * **shuffle** – whether to shuffle all the data in the buffer every time a new chunk loaded.

  * **seed** – random seed to initialize the random state for all the workers if shuffle is True, set seed += 1 in every iter() call, refer to the PyTorch idea: [pytorch/pytorch](https://github.com/pytorch/pytorch/blob/v1.10.0/torch/utils/data/distributed.py#L98).

  * **kwargs_read_csv** – dictionary args to pass to pandas read_csv function. Default to `{"chunksize": chunksize}`.

  * **kwargs** – additional arguments for pandas.merge() API to join tables.

close()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/iterable_dataset.py#L252-L262)[#](data.html#monai.data.CSVIterableDataset.close
"Link to this definition")

    

Close the pandas TextFileReader iterable objects. If the input src is file
path, TextFileReader was created internally, need to close it. If the input
src is iterable object, depends on users requirements whether to close it in
this function. For more details, please check:
<https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>?#iteration.

reset(_src =None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/iterable_dataset.py#L228-L250)[#](data.html#monai.data.CSVIterableDataset.reset
"Link to this definition")

    

Reset the pandas TextFileReader iterable object to read data. For more
details, please check: <https://pandas.pydata.org/pandas-
docs/stable/user_guide/io.html>?#iteration.

Parameters:

    

**src** – if not None and provided the filename of CSV file, it can be a str,
URL, path object or file-like object to load. also support to provide iter for
stream input directly, will skip loading from filename. if provided a list of
filenames or iters, it will join the tables. default to self.src.

### PersistentDataset[#](data.html#persistentdataset "Link to this heading")

_class _monai.data.PersistentDataset(_data_ , _transform_ , _cache_dir_ ,
_hash_func= <function pickle_hashing>_, _pickle_module='pickle'_ ,
_pickle_protocol=2_ , _hash_transform=None_ ,
_reset_ops_id=True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L161-L413)[#](data.html#monai.data.PersistentDataset
"Link to this definition")

    

Persistent storage of pre-computed values to efficiently manage larger than
memory dictionary format data, it can operate transforms for specific fields.
Results from the non-random transform components are computed when first used,
and stored in the cache_dir for rapid retrieval on subsequent uses. If passing
slicing indices, will return a PyTorch Subset, for example: data: Subset =
dataset[1:4], for more details, please check:
<https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset>

The transforms which are supposed to be cached must implement the
monai.transforms.Transform interface and should not be Randomizable. This
dataset will cache the outcomes before the first Randomizable Transform within
a Compose instance.

For example, typical input data can be a list of dictionaries:

    
    
    [{                            {                            {
        'image': 'image1.nii.gz',    'image': 'image2.nii.gz',    'image': 'image3.nii.gz',
        'label': 'label1.nii.gz',    'label': 'label2.nii.gz',    'label': 'label3.nii.gz',
        'extra': 123                 'extra': 456                 'extra': 789
    },                           },                           }]
    

For a composite transform like

    
    
    [ LoadImaged(keys=['image', 'label']),
    Orientationd(keys=['image', 'label'], axcodes='RAS'),
    ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(96, 96, 96),
                            pos=1, neg=1, num_samples=4, image_key='image', image_threshold=0),
    ToTensord(keys=['image', 'label'])]
    

Upon first use a filename based dataset will be processed by the transform for
the [LoadImaged, Orientationd, ScaleIntensityRanged] and the resulting tensor
written to the cache_dir before applying the remaining random dependant
transforms [RandCropByPosNegLabeld, ToTensord] elements for use in the
analysis.

Subsequent uses of a dataset directly read pre-processed results from
cache_dir followed by applying the random dependant parts of transform
processing.

During training call set_data() to update input data and recompute cache
content.

Note

The input data must be a list of file paths and will hash them as cache keys.

The filenames of the cached files also try to contain the hash of the
transforms. In this fashion, PersistentDataset should be robust to changes in
transforms. This, however, is not guaranteed, so caution should be used when
modifying transforms to avoid unexpected errors. If in doubt, it is advisable
to clear the cache directory.

Lazy Resampling:

    

If you make use of the lazy resampling feature of monai.transforms.Compose,
please refer to its documentation to familiarize yourself with the interaction
between PersistentDataset and lazy resampling.

__init__(_data_ , _transform_ , _cache_dir_ , _hash_func= <function
pickle_hashing>_, _pickle_module='pickle'_ , _pickle_protocol=2_ ,
_hash_transform=None_ ,
_reset_ops_id=True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L217-L276)[#](data.html#monai.data.PersistentDataset.__init__
"Link to this definition")

    

Parameters:

    

  * **data** – input data file paths to load and transform to generate dataset for model. PersistentDataset expects input data to be a list of serializable and hashes them as cache keys using hash_func.

  * **transform** – transforms to execute operations on input data.

  * **cache_dir** – If specified, this is the location for persistent storage of pre-computed transformed data tensors. The cache_dir is computed once, and persists on disk until explicitly removed. Different runs, programs, experiments may share a common cache dir provided that the transforms pre-processing is consistent. If cache_dir doesn’t exist, will automatically create it. If cache_dir is None, there is effectively no caching.

  * **hash_func** – a callable to compute hash from data items to be cached. defaults to monai.data.utils.pickle_hashing.

  * **pickle_module** – string representing the module used for pickling metadata and objects, default to “pickle”. due to the pickle limitation in multi-processing of Dataloader, we can’t use pickle as arg directly, so here we use a string name instead. if want to use other pickle module at runtime, just register like: >>> from monai.data import utils >>> utils.SUPPORTED_PICKLE_MOD[“test”] = other_pickle this arg is used by torch.save, for more details, please check: <https://pytorch.org/docs/stable/generated/torch.save.html#torch.save>, and `monai.data.utils.SUPPORTED_PICKLE_MOD`.

  * **pickle_protocol** – can be specified to override the default protocol, default to 2. this arg is used by torch.save, for more details, please check: <https://pytorch.org/docs/stable/generated/torch.save.html#torch.save>.

  * **hash_transform** – a callable to compute hash from the transform information when caching. This may reduce errors due to transforms changing during experiments. Default to None (no hash). Other options are pickle_hashing and json_hashing functions from monai.data.utils.

  * **reset_ops_id** – whether to set TraceKeys.ID to `Tracekys.NONE`, defaults to `True`. When this is enabled, the traced transform instance IDs will be removed from the cached MetaTensors. This is useful for skipping the transform instance checks when inverting applied operations using the cached content and with re-created transform instances.

set_data(_data_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L298-L306)[#](data.html#monai.data.PersistentDataset.set_data
"Link to this definition")

    

Set the input data and delete all the out-dated cache content.

set_transform_hash(_hash_xform_func_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L278-L296)[#](data.html#monai.data.PersistentDataset.set_transform_hash
"Link to this definition")

    

Get hashable transforms, and then hash them. Hashable transforms are
deterministic transforms that inherit from Transform. We stop at the first
non-deterministic transform, or first that does not inherit from MONAI’s
Transform class.

### GDSDataset[#](data.html#gdsdataset "Link to this heading")

_class _monai.data.GDSDataset(_data_ , _transform_ , _cache_dir_ , _device_ ,
_hash_func= <function pickle_hashing>_, _hash_transform=None_ ,
_reset_ops_id=True_ , _**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1499-L1673)[#](data.html#monai.data.GDSDataset
"Link to this definition")

    

An extension of the PersistentDataset using direct memory access(DMA) data
path between GPU memory and storage, thus avoiding a bounce buffer through the
CPU. This direct path can increase system bandwidth while decreasing latency
and utilization load on the CPU and GPU.

A tutorial is available: [Project-MONAI/tutorials](https://github.com/Project-
MONAI/tutorials/blob/main/modules/GDS_dataset.ipynb).

See also: [rapidsai/kvikio](https://github.com/rapidsai/kvikio)

__init__(_data_ , _transform_ , _cache_dir_ , _device_ , _hash_func= <function
pickle_hashing>_, _hash_transform=None_ , _reset_ops_id=True_ ,
_**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1510-L1556)[#](data.html#monai.data.GDSDataset.__init__
"Link to this definition")

    

Parameters:

    

  * **data** – input data file paths to load and transform to generate dataset for model. GDSDataset expects input data to be a list of serializable and hashes them as cache keys using hash_func.

  * **transform** – transforms to execute operations on input data.

  * **cache_dir** – If specified, this is the location for gpu direct storage of pre-computed transformed data tensors. The cache_dir is computed once, and persists on disk until explicitly removed. Different runs, programs, experiments may share a common cache dir provided that the transforms pre-processing is consistent. If cache_dir doesn’t exist, will automatically create it. If cache_dir is None, there is effectively no caching.

  * **device** – target device to put the output Tensor data. Note that only int can be used to specify the gpu to be used.

  * **hash_func** – a callable to compute hash from data items to be cached. defaults to monai.data.utils.pickle_hashing.

  * **hash_transform** – a callable to compute hash from the transform information when caching. This may reduce errors due to transforms changing during experiments. Default to None (no hash). Other options are pickle_hashing and json_hashing functions from monai.data.utils.

  * **reset_ops_id** – whether to set TraceKeys.ID to `Tracekys.NONE`, defaults to `True`. When this is enabled, the traced transform instance IDs will be removed from the cached MetaTensors. This is useful for skipping the transform instance checks when inverting applied operations using the cached content and with re-created transform instances.

### CacheNTransDataset[#](data.html#cachentransdataset "Link to this heading")

_class _monai.data.CacheNTransDataset(_data_ , _transform_ , _cache_n_trans_ ,
_cache_dir_ , _hash_func= <function pickle_hashing>_, _pickle_module='pickle'_
, _pickle_protocol=2_ , _hash_transform=None_ ,
_reset_ops_id=True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L416-L507)[#](data.html#monai.data.CacheNTransDataset
"Link to this definition")

    

Extension of PersistentDataset, it can also cache the result of first N
transforms, no matter it’s random or not.

__init__(_data_ , _transform_ , _cache_n_trans_ , _cache_dir_ , _hash_func=
<function pickle_hashing>_, _pickle_module='pickle'_ , _pickle_protocol=2_ ,
_hash_transform=None_ ,
_reset_ops_id=True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L422-L480)[#](data.html#monai.data.CacheNTransDataset.__init__
"Link to this definition")

    

Parameters:

    

  * **data** – input data file paths to load and transform to generate dataset for model. PersistentDataset expects input data to be a list of serializable and hashes them as cache keys using hash_func.

  * **transform** – transforms to execute operations on input data.

  * **cache_n_trans** – cache the result of first N transforms.

  * **cache_dir** – If specified, this is the location for persistent storage of pre-computed transformed data tensors. The cache_dir is computed once, and persists on disk until explicitly removed. Different runs, programs, experiments may share a common cache dir provided that the transforms pre-processing is consistent. If cache_dir doesn’t exist, will automatically create it. If cache_dir is None, there is effectively no caching.

  * **hash_func** – a callable to compute hash from data items to be cached. defaults to monai.data.utils.pickle_hashing.

  * **pickle_module** – string representing the module used for pickling metadata and objects, default to “pickle”. due to the pickle limitation in multi-processing of Dataloader, we can’t use pickle as arg directly, so here we use a string name instead. if want to use other pickle module at runtime, just register like: >>> from monai.data import utils >>> utils.SUPPORTED_PICKLE_MOD[“test”] = other_pickle this arg is used by torch.save, for more details, please check: <https://pytorch.org/docs/stable/generated/torch.save.html#torch.save>, and `monai.data.utils.SUPPORTED_PICKLE_MOD`.

  * **pickle_protocol** – can be specified to override the default protocol, default to 2. this arg is used by torch.save, for more details, please check: <https://pytorch.org/docs/stable/generated/torch.save.html#torch.save>.

  * **hash_transform** – a callable to compute hash from the transform information when caching. This may reduce errors due to transforms changing during experiments. Default to None (no hash). Other options are pickle_hashing and json_hashing functions from monai.data.utils.

  * **reset_ops_id** – whether to set TraceKeys.ID to `Tracekys.NONE`, defaults to `True`. When this is enabled, the traced transform instance IDs will be removed from the cached MetaTensors. This is useful for skipping the transform instance checks when inverting applied operations using the cached content and with re-created transform instances.

### LMDBDataset[#](data.html#lmdbdataset "Link to this heading")

_class _monai.data.LMDBDataset(_data_ , _transform_ , _cache_dir='cache'_ ,
_hash_func= <function pickle_hashing>_, _db_name='monai_cache'_ ,
_progress=True_ , _pickle_protocol=5_ , _hash_transform=None_ ,
_reset_ops_id=True_ ,
_lmdb_kwargs=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L510-L681)[#](data.html#monai.data.LMDBDataset
"Link to this definition")

    

Extension of PersistentDataset using LMDB as the backend.

See also

[`monai.data.PersistentDataset`](data.html#monai.data.PersistentDataset
"monai.data.PersistentDataset")

Examples

    
    
    >>> items = [{"data": i} for i in range(5)]
    # [{'data': 0}, {'data': 1}, {'data': 2}, {'data': 3}, {'data': 4}]
    >>> lmdb_ds = monai.data.LMDBDataset(items, transform=monai.transforms.SimulateDelayd("data", delay_time=1))
    >>> print(list(lmdb_ds))  # using the cached results
    

__init__(_data_ , _transform_ , _cache_dir='cache'_ , _hash_func= <function
pickle_hashing>_, _db_name='monai_cache'_ , _progress=True_ ,
_pickle_protocol=5_ , _hash_transform=None_ , _reset_ops_id=True_ ,
_lmdb_kwargs=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L526-L587)[#](data.html#monai.data.LMDBDataset.__init__
"Link to this definition")

    

Parameters:

    

  * **data** – input data file paths to load and transform to generate dataset for model. LMDBDataset expects input data to be a list of serializable and hashes them as cache keys using hash_func.

  * **transform** – transforms to execute operations on input data.

  * **cache_dir** – if specified, this is the location for persistent storage of pre-computed transformed data tensors. The cache_dir is computed once, and persists on disk until explicitly removed. Different runs, programs, experiments may share a common cache dir provided that the transforms pre-processing is consistent. If the cache_dir doesn’t exist, will automatically create it. Defaults to “./cache”.

  * **hash_func** – a callable to compute hash from data items to be cached. defaults to monai.data.utils.pickle_hashing.

  * **db_name** – lmdb database file name. Defaults to “monai_cache”.

  * **progress** – whether to display a progress bar.

  * **pickle_protocol** – pickle protocol version. Defaults to pickle.HIGHEST_PROTOCOL. <https://docs.python.org/3/library/pickle.html#pickle-protocols>

  * **hash_transform** – a callable to compute hash from the transform information when caching. This may reduce errors due to transforms changing during experiments. Default to None (no hash). Other options are pickle_hashing and json_hashing functions from monai.data.utils.

  * **reset_ops_id** – whether to set TraceKeys.ID to `Tracekeys.NONE`, defaults to `True`. When this is enabled, the traced transform instance IDs will be removed from the cached MetaTensors. This is useful for skipping the transform instance checks when inverting applied operations using the cached content and with re-created transform instances.

  * **lmdb_kwargs** – additional keyword arguments to the lmdb environment. for more details please visit: <https://lmdb.readthedocs.io/en/release/#environment-class>

info()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L671-L681)[#](data.html#monai.data.LMDBDataset.info
"Link to this definition")

    

Returns: dataset info dictionary.

set_data(_data_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L589-L595)[#](data.html#monai.data.LMDBDataset.set_data
"Link to this definition")

    

Set the input data and delete all the out-dated cache content.

### CacheDataset[#](data.html#cachedataset "Link to this heading")

_class _monai.data.CacheDataset(_data_ , _transform=None_ ,
_cache_num=9223372036854775807_ , _cache_rate=1.0_ , _num_workers=1_ ,
_progress=True_ , _copy_cache=True_ , _as_contiguous=True_ ,
_hash_as_key=False_ , _hash_func= <function pickle_hashing>_,
_runtime_cache=False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L684-L916)[#](data.html#monai.data.CacheDataset
"Link to this definition")

    

Dataset with cache mechanism that can load data and cache deterministic
transforms’ result during training.

By caching the results of non-random preprocessing transforms, it accelerates
the training data pipeline. If the requested data is not in the cache, all
transforms will run normally (see also
[`monai.data.dataset.Dataset`](data.html#monai.data.Dataset
"monai.data.dataset.Dataset")).

Users can set the cache rate or number of items to cache. It is recommended to
experiment with different cache_num or cache_rate to identify the best
training speed.

The transforms which are supposed to be cached must implement the
monai.transforms.Transform interface and should not be Randomizable. This
dataset will cache the outcomes before the first Randomizable Transform within
a Compose instance. So to improve the caching efficiency, please always put as
many as possible non-random transforms before the randomized ones when
composing the chain of transforms. If passing slicing indices, will return a
PyTorch Subset, for example: data: Subset = dataset[1:4], for more details,
please check:
<https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset>

For example, if the transform is a Compose of:

    
    
    transforms = Compose([
        LoadImaged(),
        EnsureChannelFirstd(),
        Spacingd(),
        Orientationd(),
        ScaleIntensityRanged(),
        RandCropByPosNegLabeld(),
        ToTensord()
    ])
    

when transforms is used in a multi-epoch training pipeline, before the first
training epoch, this dataset will cache the results up to
`ScaleIntensityRanged`, as all non-random transforms LoadImaged,
EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged can be
cached. During training, the dataset will load the cached results and run
`RandCropByPosNegLabeld` and `ToTensord`, as `RandCropByPosNegLabeld` is a
randomized transform and the outcome not cached.

During training call set_data() to update input data and recompute cache
content, note that it requires persistent_workers=False in the PyTorch
DataLoader.

Note

CacheDataset executes non-random transforms and prepares cache content in the
main process before the first epoch, then all the subprocesses of DataLoader
will read the same cache content in the main process during training. it may
take a long time to prepare cache content according to the size of expected
cache data. So to debug or verify the program before real training, users can
set cache_rate=0.0 or cache_num=0 to temporarily skip caching.

Lazy Resampling:

    

If you make use of the lazy resampling feature of monai.transforms.Compose,
please refer to its documentation to familiarize yourself with the interaction
between CacheDataset and lazy resampling.

__init__(_data_ , _transform=None_ , _cache_num=9223372036854775807_ ,
_cache_rate=1.0_ , _num_workers=1_ , _progress=True_ , _copy_cache=True_ ,
_as_contiguous=True_ , _hash_as_key=False_ , _hash_func= <function
pickle_hashing>_, _runtime_cache=False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L739-L809)[#](data.html#monai.data.CacheDataset.__init__
"Link to this definition")

    

Parameters:

    

  * **data** – input data to load and transform to generate dataset for model.

  * **transform** – transforms to execute operations on input data.

  * **cache_num** – number of items to be cached. Default is sys.maxsize. will take the minimum of (cache_num, data_length x cache_rate, data_length).

  * **cache_rate** – percentage of cached data in total, default is 1.0 (cache all). will take the minimum of (cache_num, data_length x cache_rate, data_length).

  * **num_workers** – the number of worker threads if computing cache in the initialization. If num_workers is None then the number returned by os.cpu_count() is used. If a value less than 1 is specified, 1 will be used instead.

  * **progress** – whether to display a progress bar.

  * **copy_cache** – whether to deepcopy the cache content before applying the random transforms, default to True. if the random transforms don’t modify the cached content (for example, randomly crop from the cached image and deepcopy the crop region) or if every cache item is only used once in a multi-processing environment, may set copy=False for better performance.

  * **as_contiguous** – whether to convert the cached NumPy array or PyTorch tensor to be contiguous. it may help improve the performance of following logic.

  * **hash_as_key** – whether to compute hash value of input data as the key to save cache, if key exists, avoid saving duplicated content. it can help save memory when the dataset has duplicated items or augmented dataset.

  * **hash_func** – if hash_as_key, a callable to compute hash from data items to be cached. defaults to monai.data.utils.pickle_hashing.

  * **runtime_cache** – 

mode of cache at the runtime. Default to False to prepare the cache content
for the entire `data` during initialization, this potentially largely increase
the time required between the constructor called and first mini-batch
generated. Three options are provided to compute the cache on the fly after
the dataset initialization:

    1. `"threads"` or `True`: use a regular `list` to store the cache items.

    2. `"processes"`: use a ListProxy to store the cache items, it can be shared among processes.

    3. A list-like object: a users-provided container to be used to store the cache items.

For thread-based caching (typically for caching cuda tensors), option 1 is
recommended. For single process workflows with multiprocessing data loading,
option 2 is recommended. For multiprocessing workflows (typically for
distributed training), where this class is initialized in subprocesses, option
3 is recommended, and the list-like object should be prepared in the main
process and passed to all subprocesses. Not following these recommendations
may lead to runtime errors or duplicated cache across processes.

set_data(_data_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L811-L846)[#](data.html#monai.data.CacheDataset.set_data
"Link to this definition")

    

Set the input data and run deterministic transforms to generate cache content.

Note: should call this func after an entire epoch and must set
persistent_workers=False in PyTorch DataLoader, because it needs to create new
worker processes based on new generated cache content.

Return type:

    

`None`

### SmartCacheDataset[#](data.html#smartcachedataset "Link to this heading")

_class _monai.data.SmartCacheDataset(_data_ , _transform =None_, _replace_rate
=0.1_, _cache_num =9223372036854775807_, _cache_rate =1.0_, _num_init_workers
=1_, _num_replace_workers =1_, _progress =True_, _shuffle =True_, _seed =0_,
_copy_cache =True_, _as_contiguous =True_, _runtime_cache
=False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L919-L1220)[#](data.html#monai.data.SmartCacheDataset
"Link to this definition")

    

Re-implementation of the SmartCache mechanism in NVIDIA Clara-train SDK. At
any time, the cache pool only keeps a subset of the whole dataset. In each
epoch, only the items in the cache are used for training. This ensures that
data needed for training is readily available, keeping GPU resources busy.
Note that cached items may still have to go through a non-deterministic
transform sequence before being fed to GPU. At the same time, another thread
is preparing replacement items by applying the transform sequence to items not
in cache. Once one epoch is completed, Smart Cache replaces the same number of
items with replacement items. Smart Cache uses a simple running window
algorithm to determine the cache content and replacement items. Let N be the
configured number of objects in cache; and R be the number of replacement
objects (R = ceil(N * r), where r is the configured replace rate). For more
details, please refer to: <https://docs.nvidia.com/clara/clara-train-
archive/3.1/nvmidl/additional_features/smart_cache.html> If passing slicing
indices, will return a PyTorch Subset, for example: data: Subset =
dataset[1:4], for more details, please check:
<https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset>

For example, if we have 5 images: [image1, image2, image3, image4, image5],
and cache_num=4, replace_rate=0.25. so the actual training images cached and
replaced for every epoch are as below:

    
    
    epoch 1: [image1, image2, image3, image4]
    epoch 2: [image2, image3, image4, image5]
    epoch 3: [image3, image4, image5, image1]
    epoch 3: [image4, image5, image1, image2]
    epoch N: [image[N % 5] ...]
    

The usage of SmartCacheDataset contains 4 steps:

>   1. Initialize SmartCacheDataset object and cache for the first epoch.
>
>   2. Call start() to run replacement thread in background.
>
>   3. Call update_cache() before every epoch to replace training items.
>
>   4. Call shutdown() when training ends.
>
>

During training call set_data() to update input data and recompute cache
content, note to call shutdown() to stop first, then update data and call
start() to restart.

Note

This replacement will not work for below cases: 1\. Set the
multiprocessing_context of DataLoader to spawn. 2\. Launch distributed data
parallel with torch.multiprocessing.spawn. 3\. Run on windows(the default
multiprocessing method is spawn) with num_workers greater than 0. 4\. Set the
persistent_workers of DataLoader to True with num_workers greater than 0.

If using MONAI workflows, please add SmartCacheHandler to the handler list of
trainer, otherwise, please make sure to call start(), update_cache(),
shutdown() during training.

Parameters:

    

  * **data** – input data to load and transform to generate dataset for model.

  * **transform** – transforms to execute operations on input data.

  * **replace_rate** – percentage of the cached items to be replaced in every epoch (default to 0.1).

  * **cache_num** – number of items to be cached. Default is sys.maxsize. will take the minimum of (cache_num, data_length x cache_rate, data_length).

  * **cache_rate** – percentage of cached data in total, default is 1.0 (cache all). will take the minimum of (cache_num, data_length x cache_rate, data_length).

  * **num_init_workers** – the number of worker threads to initialize the cache for first epoch. If num_init_workers is None then the number returned by os.cpu_count() is used. If a value less than 1 is specified, 1 will be used instead.

  * **num_replace_workers** – the number of worker threads to prepare the replacement cache for every epoch. If num_replace_workers is None then the number returned by os.cpu_count() is used. If a value less than 1 is specified, 1 will be used instead.

  * **progress** – whether to display a progress bar when caching for the first epoch.

  * **shuffle** – whether to shuffle the whole data list before preparing the cache content for first epoch. it will not modify the original input data sequence in-place.

  * **seed** – random seed if shuffle is True, default to 0.

  * **copy_cache** – whether to deepcopy the cache content before applying the random transforms, default to True. if the random transforms don’t modify the cache content or every cache item is only used once in a multi-processing environment, may set copy=False for better performance.

  * **as_contiguous** – whether to convert the cached NumPy array or PyTorch tensor to be contiguous. it may help improve the performance of following logic.

  * **runtime_cache** – Default to False, other options are not implemented yet.

is_started()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1083-L1088)[#](data.html#monai.data.SmartCacheDataset.is_started
"Link to this definition")

    

Check whether the replacement thread is already started.

manage_replacement()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1204-L1213)[#](data.html#monai.data.SmartCacheDataset.manage_replacement
"Link to this definition")

    

Background thread for replacement.

Return type:

    

`None`

randomize(_data_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1066-L1070)[#](data.html#monai.data.SmartCacheDataset.randomize
"Link to this definition")

    

Within this method, `self.R` should be used, instead of np.random, to
introduce random factors.

all `self.R` calls happen here so that we have a better chance to identify
errors of sync the random state.

This method can generate the random factors based on properties of the input
data.

Raises:

    

**NotImplementedError** – When the subclass does not override this method.

Return type:

    

`None`

set_data(_data_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1050-L1064)[#](data.html#monai.data.SmartCacheDataset.set_data
"Link to this definition")

    

Set the input data and run deterministic transforms to generate cache content.

Note: should call shutdown() before calling this func.

shutdown()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1156-L1168)[#](data.html#monai.data.SmartCacheDataset.shutdown
"Link to this definition")

    

Shut down the background thread for replacement.

start()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1090-L1096)[#](data.html#monai.data.SmartCacheDataset.start
"Link to this definition")

    

Start the background thread to replace training items for every epoch.

update_cache()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1130-L1140)[#](data.html#monai.data.SmartCacheDataset.update_cache
"Link to this definition")

    

Update cache items for current epoch, need to call this function before every
epoch. If the cache has been shutdown before, need to restart the _replace_mgr
thread.

### ZipDataset[#](data.html#zipdataset "Link to this heading")

_class _monai.data.ZipDataset(_datasets_ , _transform
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1223-L1270)[#](data.html#monai.data.ZipDataset
"Link to this definition")

    

Zip several PyTorch datasets and output data(with the same index) together in
a tuple. If the output of single dataset is already a tuple, flatten it and
extend to the result. For example: if datasetA returns (img, imgmeta),
datasetB returns (seg, segmeta), finally return (img, imgmeta, seg, segmeta).
And if the datasets don’t have same length, use the minimum length of them as
the length of ZipDataset. If passing slicing indices, will return a PyTorch
Subset, for example: data: Subset = dataset[1:4], for more details, please
check: <https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset>

Examples:

    
    
    >>> zip_data = ZipDataset([[1, 2, 3], [4, 5]])
    >>> print(len(zip_data))
    2
    >>> for item in zip_data:
    >>>    print(item)
    [1, 4]
    [2, 5]
    

__init__(_datasets_ , _transform =None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1246-L1252)[#](data.html#monai.data.ZipDataset.__init__
"Link to this definition")

    

Parameters:

    

  * **datasets** – list of datasets to zip together.

  * **transform** – a callable data transform operates on the zipped item from datasets.

### ArrayDataset[#](data.html#arraydataset "Link to this heading")

_class _monai.data.ArrayDataset(_img_ , _img_transform =None_, _seg =None_,
_seg_transform =None_, _labels =None_, _label_transform
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1273-L1369)[#](data.html#monai.data.ArrayDataset
"Link to this definition")

    

Dataset for segmentation and classification tasks based on array format input
data and transforms. It ensures the same random seeds in the randomized
transforms defined for image, segmentation and label. The transform can be
[`monai.transforms.Compose`](transforms.html#monai.transforms.Compose
"monai.transforms.Compose") or any other callable object. For example: If
train based on Nifti format images without metadata, all transforms can be
composed:

    
    
    img_transform = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            RandAdjustContrast()
        ]
    )
    ArrayDataset(img_file_list, img_transform=img_transform)
    

If training based on images and the metadata, the array transforms can not be
composed because several transforms receives multiple parameters or return
multiple values. Then Users need to define their own callable method to parse
metadata from LoadImage or set affine matrix to Spacing transform:

    
    
    class TestCompose(Compose):
        def __call__(self, input_):
            img, metadata = self.transforms[0](input_)
            img = self.transforms[1](img)
            img, _, _ = self.transforms[2](img, metadata["affine"])
            return self.transforms[3](img), metadata
    img_transform = TestCompose(
        [
            LoadImage(image_only=False),
            EnsureChannelFirst(),
            Spacing(pixdim=(1.5, 1.5, 3.0)),
            RandAdjustContrast()
        ]
    )
    ArrayDataset(img_file_list, img_transform=img_transform)
    

Examples:

    
    
    >>> ds = ArrayDataset([1, 2, 3, 4], lambda x: x + 0.1)
    >>> print(ds[0])
    1.1
    
    >>> ds = ArrayDataset(img=[1, 2, 3, 4], seg=[5, 6, 7, 8])
    >>> print(ds[0])
    [1, 5]
    

__init__(_img_ , _img_transform =None_, _seg =None_, _seg_transform =None_,
_labels =None_, _label_transform =None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1323-L1350)[#](data.html#monai.data.ArrayDataset.__init__
"Link to this definition")

    

Initializes the dataset with the filename lists. The transform img_transform
is applied to the images and seg_transform to the segmentations.

Parameters:

    

  * **img** – sequence of images.

  * **img_transform** – transform to apply to each element in img.

  * **seg** – sequence of segmentations.

  * **seg_transform** – transform to apply to each element in seg.

  * **labels** – sequence of labels.

  * **label_transform** – transform to apply to each element in labels.

randomize(_data =None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1355-L1356)[#](data.html#monai.data.ArrayDataset.randomize
"Link to this definition")

    

Within this method, `self.R` should be used, instead of np.random, to
introduce random factors.

all `self.R` calls happen here so that we have a better chance to identify
errors of sync the random state.

This method can generate the random factors based on properties of the input
data.

Raises:

    

**NotImplementedError** – When the subclass does not override this method.

### ImageDataset[#](data.html#imagedataset "Link to this heading")

_class _monai.data.ImageDataset(_image_files_ , _seg_files=None_ ,
_labels=None_ , _transform=None_ , _seg_transform=None_ ,
_label_transform=None_ , _image_only=True_ , _transform_with_metadata=False_ ,
_dtype= <class 'numpy.float32'>_, _reader=None_ , _*args_ ,
_**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_dataset.py#L26-L155)[#](data.html#monai.data.ImageDataset
"Link to this definition")

    

Loads image/segmentation pairs of files from the given filename lists.
Transformations can be specified for the image and segmentation arrays
separately. The difference between this dataset and ArrayDataset is that this
dataset can apply transform chain to images and segs and return both the
images and metadata, and no need to specify transform to load images from
files. For more information, please see the image_dataset demo in the MONAI
tutorial repo, [Project-MONAI/tutorials](https://github.com/Project-
MONAI/tutorials/blob/master/modules/image_dataset.ipynb)

__init__(_image_files_ , _seg_files=None_ , _labels=None_ , _transform=None_ ,
_seg_transform=None_ , _label_transform=None_ , _image_only=True_ ,
_transform_with_metadata=False_ , _dtype= <class 'numpy.float32'>_,
_reader=None_ , _*args_ , _**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_dataset.py#L36-L94)[#](data.html#monai.data.ImageDataset.__init__
"Link to this definition")

    

Initializes the dataset with the image and segmentation filename lists. The
transform transform is applied to the images and seg_transform to the
segmentations.

Parameters:

    

  * **image_files** – list of image filenames.

  * **seg_files** – if in segmentation task, list of segmentation filenames.

  * **labels** – if in classification task, list of classification labels.

  * **transform** – transform to apply to image arrays.

  * **seg_transform** – transform to apply to segmentation arrays.

  * **label_transform** – transform to apply to the label data.

  * **image_only** – if True return only the image volume, otherwise, return image volume and the metadata.

  * **transform_with_metadata** – if True, the metadata will be passed to the transforms whenever possible.

  * **dtype** – if not None convert the loaded image to this data type.

  * **reader** – register reader to load image file and metadata, if None, will use the default readers. If a string of reader name provided, will construct a reader object with the *args and **kwargs parameters, supported reader name: “NibabelReader”, “PILReader”, “ITKReader”, “NumpyReader”

  * **args** – additional parameters for reader if providing a reader name.

  * **kwargs** – additional parameters for reader if providing a reader name.

Raises:

    

**ValueError** – When `seg_files` length differs from `image_files`

randomize(_data =None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_dataset.py#L99-L100)[#](data.html#monai.data.ImageDataset.randomize
"Link to this definition")

    

Within this method, `self.R` should be used, instead of np.random, to
introduce random factors.

all `self.R` calls happen here so that we have a better chance to identify
errors of sync the random state.

This method can generate the random factors based on properties of the input
data.

Raises:

    

**NotImplementedError** – When the subclass does not override this method.

### NPZDictItemDataset[#](data.html#npzdictitemdataset "Link to this heading")

_class _monai.data.NPZDictItemDataset(_npzfile_ , _keys_ , _transform =None_,
_other_keys =()_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1372-L1421)[#](data.html#monai.data.NPZDictItemDataset
"Link to this definition")

    

Represents a dataset from a loaded NPZ file. The members of the file to load
are named in the keys of keys and stored under the keyed name. All loaded
arrays must have the same 0-dimension (batch) size. Items are always dicts
mapping names to an item extracted from the loaded arrays. If passing slicing
indices, will return a PyTorch Subset, for example: data: Subset =
dataset[1:4], for more details, please check:
<https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset>

Parameters:

    

  * **npzfile** – Path to .npz file or stream containing .npz file data

  * **keys** – Maps keys to load from file to name to store in dataset

  * **transform** – Transform to apply to batch dict

  * **other_keys** – secondary data to load from file and store in dict other_keys, not returned by __getitem__

### CSVDataset[#](data.html#csvdataset "Link to this heading")

_class _monai.data.CSVDataset(_src =None_, _row_indices =None_, _col_names
=None_, _col_types =None_, _col_groups =None_, _transform =None_,
_kwargs_read_csv =None_, _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset.py#L1424-L1496)[#](data.html#monai.data.CSVDataset
"Link to this definition")

    

Dataset to load data from CSV files and generate a list of dictionaries, every
dictionary maps to a row of the CSV file, and the keys of dictionary map to
the column names of the CSV file.

It can load multiple CSV files and join the tables with additional kwargs arg.
Support to only load specific rows and columns. And it can also group several
loaded columns to generate a new column, for example, set col_groups={“meta”:
[“meta_0”, “meta_1”, “meta_2”]}, output can be:

    
    
    [
        {"image": "./image0.nii", "meta_0": 11, "meta_1": 12, "meta_2": 13, "meta": [11, 12, 13]},
        {"image": "./image1.nii", "meta_0": 21, "meta_1": 22, "meta_2": 23, "meta": [21, 22, 23]},
    ]
    

Parameters:

    

  * **src** – if provided the filename of CSV file, it can be a str, URL, path object or file-like object to load. also support to provide pandas DataFrame directly, will skip loading from filename. if provided a list of filenames or pandas DataFrame, it will join the tables.

  * **row_indices** – indices of the expected rows to load. it should be a list, every item can be a int number or a range [start, end) for the indices. for example: row_indices=[[0, 100], 200, 201, 202, 300]. if None, load all the rows in the file.

  * **col_names** – names of the expected columns to load. if None, load all the columns.

  * **col_types** – 

type and default value to convert the loaded columns, if None, use original
data. it should be a dictionary, every item maps to an expected column, the
key is the column name and the value is None or a dictionary to define the
default value and data type. the supported keys in dictionary are: [“type”,
“default”]. for example:

    
        col_types = {
        "subject_id": {"type": str},
        "label": {"type": int, "default": 0},
        "ehr_0": {"type": float, "default": 0.0},
        "ehr_1": {"type": float, "default": 0.0},
        "image": {"type": str, "default": None},
    }
    

  * **col_groups** – args to group the loaded columns to generate a new column, it should be a dictionary, every item maps to a group, the key will be the new column name, the value is the names of columns to combine. for example: col_groups={“ehr”: [f”ehr_{i}” for i in range(10)], “meta”: [“meta_1”, “meta_2”]}

  * **transform** – transform to apply on the loaded items of a dictionary data.

  * **kwargs_read_csv** – dictionary args to pass to pandas read_csv function.

  * **kwargs** – additional arguments for pandas.merge() API to join tables.

## Patch-based dataset[#](data.html#patch-based-dataset "Link to this
heading")

### GridPatchDataset[#](data.html#gridpatchdataset "Link to this heading")

_class _monai.data.GridPatchDataset(_data_ , _patch_iter_ , _transform=None_ ,
_with_coordinates=True_ , _cache=False_ , _cache_num=9223372036854775807_ ,
_cache_rate=1.0_ , _num_workers=1_ , _progress=True_ , _copy_cache=True_ ,
_as_contiguous=True_ , _hash_func= <function
pickle_hashing>_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/grid_dataset.py#L160-L363)[#](data.html#monai.data.GridPatchDataset
"Link to this definition")

    

Yields patches from data read from an image dataset. Typically used with
PatchIter or PatchIterd so that the patches are chosen in a contiguous grid
sampling scheme.

>
>     import numpy as np
>  
>     from monai.data import GridPatchDataset, DataLoader, PatchIter,
> RandShiftIntensity
>  
>     # image-level dataset
>     images = [np.arange(16, dtype=float).reshape(1, 4, 4),
>               np.arange(16, dtype=float).reshape(1, 4, 4)]
>     # image-level patch generator, "grid sampling"
>     patch_iter = PatchIter(patch_size=(2, 2), start_pos=(0, 0))
>     # patch-level intensity shifts
>     patch_intensity = RandShiftIntensity(offsets=1.0, prob=1.0)
>  
>     # construct the dataset
>     ds = GridPatchDataset(data=images,
>                           patch_iter=patch_iter,
>                           transform=patch_intensity)
>     # use the grid patch dataset
>     for item in DataLoader(ds, batch_size=2, num_workers=2):
>         print("patch size:", item[0].shape)
>         print("coordinates:", item[1])
>  
>     # >>> patch size: torch.Size([2, 1, 2, 2])
>     #     coordinates: tensor([[[0, 1], [0, 2], [0, 2]],
>     #                          [[0, 1], [2, 4], [0, 2]]])
>  

Parameters:

    

  * **data** – the data source to read image data from.

  * **patch_iter** – converts an input image (item from dataset) into a iterable of image patches. patch_iter(dataset[idx]) must yield a tuple: (patches, coordinates). see also: [`monai.data.PatchIter`](data.html#monai.data.PatchIter "monai.data.PatchIter") or [`monai.data.PatchIterd`](data.html#monai.data.PatchIterd "monai.data.PatchIterd").

  * **transform** – a callable data transform operates on the patches.

  * **with_coordinates** – whether to yield the coordinates of each patch, default to True.

  * **cache** – whether to use cache mache mechanism, default to False. see also: [`monai.data.CacheDataset`](data.html#monai.data.CacheDataset "monai.data.CacheDataset").

  * **cache_num** – number of items to be cached. Default is sys.maxsize. will take the minimum of (cache_num, data_length x cache_rate, data_length).

  * **cache_rate** – percentage of cached data in total, default is 1.0 (cache all). will take the minimum of (cache_num, data_length x cache_rate, data_length).

  * **num_workers** – the number of worker threads if computing cache in the initialization. If num_workers is None then the number returned by os.cpu_count() is used. If a value less than 1 is specified, 1 will be used instead.

  * **progress** – whether to display a progress bar.

  * **copy_cache** – whether to deepcopy the cache content before applying the random transforms, default to True. if the random transforms don’t modify the cached content (for example, randomly crop from the cached image and deepcopy the crop region) or if every cache item is only used once in a multi-processing environment, may set copy=False for better performance.

  * **as_contiguous** – whether to convert the cached NumPy array or PyTorch tensor to be contiguous. it may help improve the performance of following logic.

  * **hash_func** – a callable to compute hash from data items to be cached. defaults to monai.data.utils.pickle_hashing.

set_data(_data_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/grid_dataset.py#L265-L281)[#](data.html#monai.data.GridPatchDataset.set_data
"Link to this definition")

    

Set the input data and run deterministic transforms to generate cache content.

Note: should call this func after an entire epoch and must set
persistent_workers=False in PyTorch DataLoader, because it needs to create new
worker processes based on new generated cache content.

Return type:

    

`None`

### PatchDataset[#](data.html#patchdataset "Link to this heading")

_class _monai.data.PatchDataset(_data_ , _patch_func_ , _samples_per_image
=1_, _transform =None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/grid_dataset.py#L366-L440)[#](data.html#monai.data.PatchDataset
"Link to this definition")

    

Yields patches from data read from an image dataset. The patches are generated
by a user-specified callable patch_func, and are optionally post-processed by
transform. For example, to generate random patch samples from an image
dataset:

    
    
    import numpy as np
    
    from monai.data import PatchDataset, DataLoader
    from monai.transforms import RandSpatialCropSamples, RandShiftIntensity
    
    # image dataset
    images = [np.arange(16, dtype=float).reshape(1, 4, 4),
              np.arange(16, dtype=float).reshape(1, 4, 4)]
    # image patch sampler
    n_samples = 5
    sampler = RandSpatialCropSamples(roi_size=(3, 3), num_samples=n_samples,
                                     random_center=True, random_size=False)
    # patch-level intensity shifts
    patch_intensity = RandShiftIntensity(offsets=1.0, prob=1.0)
    # construct the patch dataset
    ds = PatchDataset(dataset=images,
                      patch_func=sampler,
                      samples_per_image=n_samples,
                      transform=patch_intensity)
    
    # use the patch dataset, length: len(images) x samplers_per_image
    print(len(ds))
    
    >>> 10
    
    for item in DataLoader(ds, batch_size=2, shuffle=True, num_workers=2):
        print(item.shape)
    
    >>> torch.Size([2, 1, 3, 3])
    

__init__(_data_ , _patch_func_ , _samples_per_image =1_, _transform
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/grid_dataset.py#L407-L424)[#](data.html#monai.data.PatchDataset.__init__
"Link to this definition")

    

Parameters:

    

  * **data** – an image dataset to extract patches from.

  * **patch_func** – converts an input image (item from dataset) into a sequence of image patches. patch_func(dataset[idx]) must return a sequence of patches (length samples_per_image).

  * **samples_per_image** – patch_func should return a sequence of samples_per_image elements.

  * **transform** – transform applied to each patch.

#### PatchIter[#](data.html#patchiter "Link to this heading")

_class _monai.data.PatchIter(_patch_size_ , _start_pos =()_, _mode =wrap_, _**
pad_opts_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/grid_dataset.py#L42-L99)[#](data.html#monai.data.PatchIter
"Link to this definition")

    

Return a patch generator with predefined properties such as patch_size.
Typically used with
[`monai.data.GridPatchDataset`](data.html#monai.data.GridPatchDataset
"monai.data.GridPatchDataset").

__call__(_array_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/grid_dataset.py#L85-L99)[#](data.html#monai.data.PatchIter.__call__
"Link to this definition")

    

Parameters:

    

**array** (_~NdarrayTensor_) – the image to generate patches from.

Return type:

    

`Generator`[`tuple`[~NdarrayTensor, `ndarray`], `None`, `None`]

__init__(_patch_size_ , _start_pos =()_, _mode =wrap_, _**
pad_opts_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/grid_dataset.py#L49-L83)[#](data.html#monai.data.PatchIter.__init__
"Link to this definition")

    

Parameters:

    

  * **patch_size** – size of patches to generate slices for, 0/None selects whole dimension

  * **start_pos** – starting position in the array, default is 0 for each dimension

  * **mode** – available modes: (Numpy) {`"constant"`, `"edge"`, `"linear_ramp"`, `"maximum"`, `"mean"`, `"median"`, `"minimum"`, `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`} (PyTorch) {`"constant"`, `"reflect"`, `"replicate"`, `"circular"`}. One of the listed string values or a user supplied function. If None, no wrapping is performed. Defaults to `"wrap"`. See also: <https://numpy.org/doc/stable/reference/generated/numpy.pad.html> <https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html> requires pytorch >= 1.10 for best compatibility.

  * **pad_opts** – other arguments for the np.pad function. note that np.pad treats channel dimension as the first dimension.

Note

The patch_size is the size of the patch to sample from the input arrays. It is
assumed the arrays first dimension is the channel dimension which will be
yielded in its entirety so this should not be specified in patch_size. For
example, for an input 3D array with 1 channel of size (1, 20, 20, 20) a
regular grid sampling of eight patches (1, 10, 10, 10) would be specified by a
patch_size of (10, 10, 10).

#### PatchIterd[#](data.html#patchiterd "Link to this heading")

_class _monai.data.PatchIterd(_keys_ , _patch_size_ , _start_pos =()_, _mode
=wrap_, _** pad_opts_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/grid_dataset.py#L102-L157)[#](data.html#monai.data.PatchIterd
"Link to this definition")

    

Dictionary-based wrapper of
[`monai.data.PatchIter`](data.html#monai.data.PatchIter
"monai.data.PatchIter"). Return a patch generator for dictionary data and the
coordinate, Typically used with
[`monai.data.GridPatchDataset`](data.html#monai.data.GridPatchDataset
"monai.data.GridPatchDataset"). Suppose all the expected fields specified by
keys have same shape.

Parameters:

    

  * **keys** – keys of the corresponding items to iterate patches.

  * **patch_size** – size of patches to generate slices for, 0/None selects whole dimension

  * **start_pos** – starting position in the array, default is 0 for each dimension

  * **mode** – available modes: (Numpy) {`"constant"`, `"edge"`, `"linear_ramp"`, `"maximum"`, `"mean"`, `"median"`, `"minimum"`, `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`} (PyTorch) {`"constant"`, `"reflect"`, `"replicate"`, `"circular"`}. One of the listed string values or a user supplied function. If None, no wrapping is performed. Defaults to `"wrap"`. See also: <https://numpy.org/doc/stable/reference/generated/numpy.pad.html> <https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html> requires pytorch >= 1.10 for best compatibility.

  * **pad_opts** – other arguments for the np.pad function. note that np.pad treats channel dimension as the first dimension.

__call__(_data_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/grid_dataset.py#L141-L157)[#](data.html#monai.data.PatchIterd.__call__
"Link to this definition")

    

Call self as a function.

Return type:

    

`Generator`[`tuple`[`Mapping`[`Hashable`, ~NdarrayTensor], `ndarray`], `None`,
`None`]

## Image reader[#](data.html#image-reader "Link to this heading")

### ImageReader[#](data.html#imagereader "Link to this heading")

_class _monai.data.ImageReader[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L63-L120)[#](data.html#monai.data.ImageReader
"Link to this definition")

    

An abstract class defines APIs to load image files.

Typical usage of an implementation of this class is:

    
    
    image_reader = MyImageReader()
    img_obj = image_reader.read(path_to_image)
    img_data, meta_data = image_reader.get_data(img_obj)
    

  * The read call converts image filenames into image objects,

  * The get_data call fetches the image data, as well as metadata.

  * A reader should implement verify_suffix with the logic of checking the input filename by the filename extensions.

_abstract _get_data(_img_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L109-L120)[#](data.html#monai.data.ImageReader.get_data
"Link to this definition")

    

Extract data array and metadata from loaded image and return them. This
function must return two objects, the first is a numpy array of image data,
the second is a dictionary of metadata.

Parameters:

    

**img** – an image object loaded from an image file or a list of image
objects.

Return type:

    

`tuple`[`ndarray`, `dict`]

_abstract _read(_data_ , _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L96-L107)[#](data.html#monai.data.ImageReader.read
"Link to this definition")

    

Read image data from specified file or files. Note that it returns a data
object or a sequence of data objects.

Parameters:

    

  * **data** – file name or a list of file names to read.

  * **kwargs** – additional args for actual read API of 3rd party libs.

_abstract _verify_suffix(_filename_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L82-L94)[#](data.html#monai.data.ImageReader.verify_suffix
"Link to this definition")

    

Verify whether the specified filename is supported by the current reader. This
method should return True if the reader is able to read the format suggested
by the filename.

Parameters:

    

**filename** – file name or a list of file names to read. if a list of files,
verify all the suffixes.

### ITKReader[#](data.html#itkreader "Link to this heading")

_class _monai.data.ITKReader(_channel_dim =None_, _series_name =''_,
_reverse_indexing =False_, _series_meta =False_, _affine_lps_to_ras =True_,
_** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L161-L382)[#](data.html#monai.data.ITKReader
"Link to this definition")

    

Load medical images based on ITK library. All the supported image formats can
be found at:
[InsightSoftwareConsortium/ITK](https://github.com/InsightSoftwareConsortium/ITK/tree/master/Modules/IO)
The loaded data array will be in C order, for example, a 3D image NumPy array
index order will be CDWH.

Parameters:

    

  * **channel_dim** – 

the channel dimension of the input image, default is None. This is used to set
original_channel_dim in the metadata, EnsureChannelFirstD reads this field. If
None, original_channel_dim will be either no_channel or -1.

>     * Nifti file is usually “channel last”, so there is no need to specify
> this argument.
>
>     * PNG file usually has GetNumberOfComponentsPerPixel()==3, so there is
> no need to specify this argument.

  * **series_name** – the name of the DICOM series if there are multiple ones. used when loading DICOM series.

  * **reverse_indexing** – whether to use a reversed spatial indexing convention for the returned data array. If `False`, the spatial indexing convention is reversed to be compatible with ITK; otherwise, the spatial indexing follows the numpy convention. Default is `False`. This option does not affect the metadata.

  * **series_meta** – whether to load the metadata of the DICOM series (using the metadata from the first slice). This flag is checked only when loading DICOM series. Default is `False`.

  * **affine_lps_to_ras** – whether to convert the affine matrix from “LPS” to “RAS”. Defaults to `True`. Set to `True` to be consistent with `NibabelReader`, otherwise the affine matrix remains in the ITK convention.

  * **kwargs** – additional args for itk.imread API. more details about available args: [InsightSoftwareConsortium/ITK](https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itk/support/extras.py)

get_data(_img_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L272-L303)[#](data.html#monai.data.ITKReader.get_data
"Link to this definition")

    

Extract data array and metadata from loaded image and return them. This
function returns two objects, first is numpy array of image data, second is
dict of metadata. It constructs affine, original_affine, and spatial_shape and
stores them in meta dict. When loading a list of files, they are stacked
together at a new dimension as the first dimension, and the metadata of the
first image is used to represent the output metadata.

Parameters:

    

**img** – an ITK image object loaded from an image file or a list of ITK image
objects.

Return type:

    

`tuple`[`ndarray`, `dict`]

read(_data_ , _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L221-L270)[#](data.html#monai.data.ITKReader.read
"Link to this definition")

    

Read image data from specified file or files, it can read a list of images and
stack them together as multi-channel data in get_data(). If passing directory
path instead of file path, will treat it as DICOM images series and read. Note
that the returned object is ITK image object or list of ITK image objects.

Parameters:

    

  * **data** – file name or a list of file names to read,

  * **kwargs** – additional args for itk.imread API, will override self.kwargs for existing keys. More details about available args: [InsightSoftwareConsortium/ITK](https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itk/support/extras.py)

verify_suffix(_filename_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L210-L219)[#](data.html#monai.data.ITKReader.verify_suffix
"Link to this definition")

    

Verify whether the specified file or files format is supported by ITK reader.

Parameters:

    

**filename** – file name or a list of file names to read. if a list of files,
verify all the suffixes.

### NibabelReader[#](data.html#nibabelreader "Link to this heading")

_class _monai.data.NibabelReader(_channel_dim =None_, _as_closest_canonical
=False_, _squeeze_non_spatial_dims =False_, _to_gpu =False_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L994-L1215)[#](data.html#monai.data.NibabelReader
"Link to this definition")

    

Load NIfTI format images based on Nibabel library.

Parameters:

    

  * **channel_dim** – the channel dimension of the input image, default is None. this is used to set original_channel_dim in the metadata, EnsureChannelFirstD reads this field. if None, original_channel_dim will be either no_channel or -1. most Nifti files are usually “channel last”, no need to specify this argument for them.

  * **as_closest_canonical** – if True, load the image as closest to canonical axis format.

  * **squeeze_non_spatial_dims** – if True, non-spatial singletons will be squeezed, e.g. (256,256,1,3) -> (256,256,3)

  * **to_gpu** – If True, load the image into GPU memory using CuPy and Kvikio. This can accelerate data loading. Default is False. CuPy and Kvikio are required for this option. Note: For compressed NIfTI files, some operations may still be performed on CPU memory, and the acceleration may not be significant. In some cases, it may be slower than loading on CPU.

  * **kwargs** – additional args for nibabel.load API. more details about available args: [nipy/nibabel](https://github.com/nipy/nibabel/blob/master/nibabel/loadsave.py)

get_data(_img_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1093-L1135)[#](data.html#monai.data.NibabelReader.get_data
"Link to this definition")

    

Extract data array and metadata from loaded image and return them. This
function returns two objects, first is numpy array of image data, second is
dict of metadata. It constructs affine, original_affine, and spatial_shape and
stores them in meta dict. When loading a list of files, they are stacked
together at a new dimension as the first dimension, and the metadata of the
first image is used to present the output metadata.

Parameters:

    

**img** – a Nibabel image object loaded from an image file or a list of
Nibabel image objects.

Return type:

    

`tuple`[`ndarray`, `dict`]

read(_data_ , _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1068-L1091)[#](data.html#monai.data.NibabelReader.read
"Link to this definition")

    

Read image data from specified file or files, it can read a list of images and
stack them together as multi-channel data in get_data(). Note that the
returned object is Nibabel image object or list of Nibabel image objects.

Parameters:

    

  * **data** – file name or a list of file names to read.

  * **kwargs** – additional args for nibabel.load API, will override self.kwargs for existing keys. More details about available args: [nipy/nibabel](https://github.com/nipy/nibabel/blob/master/nibabel/loadsave.py)

verify_suffix(_filename_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1056-L1066)[#](data.html#monai.data.NibabelReader.verify_suffix
"Link to this definition")

    

Verify whether the specified file or files format is supported by Nibabel
reader.

Parameters:

    

**filename** – file name or a list of file names to read. if a list of files,
verify all the suffixes.

warmup_kvikio()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1039-L1054)[#](data.html#monai.data.NibabelReader.warmup_kvikio
"Link to this definition")

    

Warm up the Kvikio library to initialize the internal buffers, cuFile, GDS,
etc. This can accelerate the data loading process when to_gpu is set to True.

### NumpyReader[#](data.html#numpyreader "Link to this heading")

_class _monai.data.NumpyReader(_npz_keys =None_, _channel_dim =None_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1218-L1314)[#](data.html#monai.data.NumpyReader
"Link to this definition")

    

Load NPY or NPZ format data based on Numpy library, they can be arrays or
pickled objects. A typical usage is to load the mask data for classification
task. It can load part of the npz file with specified npz_keys.

Parameters:

    

  * **npz_keys** – if loading npz file, only load the specified keys, if None, load all the items. stack the loaded items together to construct a new first dimension.

  * **channel_dim** – if not None, explicitly specify the channel dim, otherwise, treat the array as no channel.

  * **kwargs** – additional args for numpy.load API except allow_pickle. more details about available args: <https://numpy.org/doc/stable/reference/generated/numpy.load.html>

get_data(_img_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1282-L1314)[#](data.html#monai.data.NumpyReader.get_data
"Link to this definition")

    

Extract data array and metadata from loaded image and return them. This
function returns two objects, first is numpy array of image data, second is
dict of metadata. It constructs affine, original_affine, and spatial_shape and
stores them in meta dict. When loading a list of files, they are stacked
together at a new dimension as the first dimension, and the metadata of the
first image is used to represent the output metadata.

Parameters:

    

**img** – a Numpy array loaded from a file or a list of Numpy arrays.

Return type:

    

`tuple`[`ndarray`, `dict`]

read(_data_ , _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1252-L1280)[#](data.html#monai.data.NumpyReader.read
"Link to this definition")

    

Read image data from specified file or files, it can read a list of data files
and stack them together as multi-channel data in get_data(). Note that the
returned object is Numpy array or list of Numpy arrays.

Parameters:

    

  * **data** – file name or a list of file names to read.

  * **kwargs** – additional args for numpy.load API except allow_pickle, will override self.kwargs for existing keys. More details about available args: <https://numpy.org/doc/stable/reference/generated/numpy.load.html>

verify_suffix(_filename_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1241-L1250)[#](data.html#monai.data.NumpyReader.verify_suffix
"Link to this definition")

    

Verify whether the specified file or files format is supported by Numpy
reader.

Parameters:

    

**filename** – file name or a list of file names to read. if a list of files,
verify all the suffixes.

### PILReader[#](data.html#pilreader "Link to this heading")

_class _monai.data.PILReader(_converter =None_, _reverse_indexing =True_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1317-L1419)[#](data.html#monai.data.PILReader
"Link to this definition")

    

Load common 2D image format (supports PNG, JPG, BMP) file or files from
provided path.

Parameters:

    

  * **converter** – additional function to convert the image data after read(). for example, use converter=lambda image: image.convert(“LA”) to convert image format.

  * **reverse_indexing** – whether to swap axis 0 and 1 after loading the array, this is enabled by default, so that output of the reader is consistent with the other readers. Set this option to `False` to use the PIL backend’s original spatial axes convention.

  * **kwargs** – additional args for Image.open API in read(), mode details about available args: <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open>

get_data(_img_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1375-L1402)[#](data.html#monai.data.PILReader.get_data
"Link to this definition")

    

Extract data array and metadata from loaded image and return them. This
function returns two objects, first is numpy array of image data, second is
dict of metadata. It computes spatial_shape and stores it in meta dict. When
loading a list of files, they are stacked together at a new dimension as the
first dimension, and the metadata of the first image is used to represent the
output metadata. Note that by default self.reverse_indexing is set to `True`,
which swaps axis 0 and 1 after loading the array because the spatial axes
definition in PIL is different from other common medical packages.

Parameters:

    

**img** – a PIL Image object loaded from a file or a list of PIL Image
objects.

Return type:

    

`tuple`[`ndarray`, `dict`]

read(_data_ , _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1349-L1373)[#](data.html#monai.data.PILReader.read
"Link to this definition")

    

Read image data from specified file or files, it can read a list of images and
stack them together as multi-channel data in get_data(). Note that the
returned object is PIL image or list of PIL image.

Parameters:

    

  * **data** – file name or a list of file names to read.

  * **kwargs** – additional args for Image.open API in read(), will override self.kwargs for existing keys. Mode details about available args: <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open>

verify_suffix(_filename_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1338-L1347)[#](data.html#monai.data.PILReader.verify_suffix
"Link to this definition")

    

Verify whether the specified file or files format is supported by PIL reader.

Parameters:

    

**filename** – file name or a list of file names to read. if a list of files,
verify all the suffixes.

### NrrdReader[#](data.html#nrrdreader "Link to this heading")

_class _monai.data.NrrdReader(_channel_dim=None_ , _dtype= <class
'numpy.float32'>_, _index_order='F'_ , _affine_lps_to_ras=True_ ,
_**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1430-L1586)[#](data.html#monai.data.NrrdReader
"Link to this definition")

    

Load NRRD format images based on pynrrd library.

Parameters:

    

  * **channel_dim** – the channel dimension of the input image, default is None. This is used to set original_channel_dim in the metadata, EnsureChannelFirstD reads this field. If None, original_channel_dim will be either no_channel or 0. NRRD files are usually “channel first”.

  * **dtype** – dtype of the data array when loading image.

  * **index_order** – Specify whether the returned data array should be in C-order (‘C’) or Fortran-order (‘F’). Numpy is usually in C-order, but default on the NRRD header is F

  * **affine_lps_to_ras** – whether to convert the affine matrix from “LPS” to “RAS”. Defaults to `True`. Set to `True` to be consistent with `NibabelReader`, otherwise the affine matrix is unmodified.

  * **kwargs** – additional args for nrrd.read API. more details about available args: [mhe/pynrrd](https://github.com/mhe/pynrrd/blob/master/nrrd/reader.py)

get_data(_img_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1496-L1534)[#](data.html#monai.data.NrrdReader.get_data
"Link to this definition")

    

Extract data array and metadata from loaded image and return them. This
function must return two objects, the first is a numpy array of image data,
the second is a dictionary of metadata.

Parameters:

    

**img** – a NrrdImage loaded from an image file or a list of image objects.

read(_data_ , _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1477-L1494)[#](data.html#monai.data.NrrdReader.read
"Link to this definition")

    

Read image data from specified file or files. Note that it returns a data
object or a sequence of data objects.

Parameters:

    

  * **data** – file name or a list of file names to read.

  * **kwargs** – additional args for actual read API of 3rd party libs.

verify_suffix(_filename_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_reader.py#L1465-L1475)[#](data.html#monai.data.NrrdReader.verify_suffix
"Link to this definition")

    

Verify whether the specified filename is supported by pynrrd reader.

Parameters:

    

**filename** – file name or a list of file names to read. if a list of files,
verify all the suffixes.

## Image writer[#](data.html#image-writer "Link to this heading")

### resolve_writer[#](data.html#resolve-writer "Link to this heading")

monai.data.resolve_writer(_ext_name_ , _error_if_not_found
=True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L92-L122)[#](data.html#monai.data.resolve_writer
"Link to this definition")

    

Resolves to a tuple of available `ImageWriter` in `SUPPORTED_WRITERS`
according to the filename extension key `ext_name`.

Parameters:

    

  * **ext_name** – the filename extension of the image. As an indexing key it will be converted to a lower case string.

  * **error_if_not_found** – whether to raise an error if no suitable image writer is found. if True , raise an `OptionalImportError`, otherwise return an empty tuple. Default is `True`.

Return type:

    

`Sequence`

### register_writer[#](data.html#register-writer "Link to this heading")

monai.data.register_writer(_ext_name_ , _*
im_writers_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L67-L89)[#](data.html#monai.data.register_writer
"Link to this definition")

    

Register `ImageWriter`, so that writing a file with filename extension
`ext_name` could be resolved to a tuple of potentially appropriate
`ImageWriter`. The customised writers could be registered by:

    
    
    from monai.data import register_writer
    # `MyWriter` must implement `ImageWriter` interface
    register_writer("nii", MyWriter)
    

Parameters:

    

  * **ext_name** – the filename extension of the image. As an indexing key, it will be converted to a lower case string.

  * **im_writers** – one or multiple ImageWriter classes with high priority ones first.

### ImageWriter[#](data.html#imagewriter "Link to this heading")

_class _monai.data.ImageWriter(_**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L125-L343)[#](data.html#monai.data.ImageWriter
"Link to this definition")

    

The class is a collection of utilities to write images to disk.

Main aspects to be considered are:

>   * dimensionality of the data array, arrangements of spatial dimensions and
> channel/time dimensions
>  
>     * `convert_to_channel_last()`
>
>   * metadata of the current affine and output affine, the data array should
> be converted accordingly
>  
>     * `get_meta_info()`
>
>     * `resample_if_needed()`
>
>   * data type handling of the output image (as part of
> `resample_if_needed()`)
>
>

Subclasses of this class should implement the backend-specific functions:

>   * `set_data_array()` to set the data array (input must be numpy array or
> torch tensor)
>  
>     * this method sets the backend object’s data part
>
>   * `set_metadata()` to set the metadata and output affine
>  
>     * this method sets the metadata including affine handling and image
> resampling
>
>   * backend-specific data object `create_backend_obj()`
>
>   * backend-specific writing function `write()`
>
>

The primary usage of subclasses of `ImageWriter` is:

    
    
    writer = MyWriter()  # subclass of ImageWriter
    writer.set_data_array(data_array)
    writer.set_metadata(meta_dict)
    writer.write(filename)
    

This creates an image writer object based on `data_array` and `meta_dict` and
write to `filename`.

It supports up to three spatial dimensions (with the resampling step supports
for both 2D and 3D). When saving multiple time steps or multiple channels
data_array, time and/or modality axes should be the at the channel_dim. For
example, the shape of a 2D eight-class and `channel_dim=0`, the segmentation
probabilities to be saved could be (8, 64, 64); in this case `data_array` will
be converted to (64, 64, 1, 8) (the third dimension is reserved as a spatial
dimension).

The `metadata` could optionally have the following keys:

>   * `'original_affine'`: for data original affine, it will be the
>  
>
> affine of the output object, defaulting to an identity matrix.
>
>   * `'affine'`: it should specify the current data affine, defaulting to an
> identity matrix.
>
>   * `'spatial_shape'`: for data output spatial shape.
>
>

When `metadata` is specified, the saver will may resample data from the space
defined by “affine” to the space defined by “original_affine”, for more
details, please refer to the `resample_if_needed` method.

__init__(_** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L178-L186)[#](data.html#monai.data.ImageWriter.__init__
"Link to this definition")

    

The constructor supports adding new instance members. The current member in
the base class is `self.data_obj`, the subclasses can add more members, so
that necessary meta information can be stored in the object and shared among
the class methods.

_classmethod _convert_to_channel_last(_data_ , _channel_dim =0_,
_squeeze_end_dims =True_, _spatial_ndim =3_, _contiguous
=False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L283-L330)[#](data.html#monai.data.ImageWriter.convert_to_channel_last
"Link to this definition")

    

Rearrange the data array axes to make the channel_dim-th dim the last
dimension and ensure there are `spatial_ndim` number of spatial dimensions.

When `squeeze_end_dims` is `True`, a postprocessing step will be applied to
remove any trailing singleton dimensions.

Parameters:

    

  * **data** – input data to be converted to “channel-last” format.

  * **channel_dim** – specifies the channel axes of the data array to move to the last. `None` indicates no channel dimension, a new axis will be appended as the channel dimension. a sequence of integers indicates multiple non-spatial dimensions.

  * **squeeze_end_dims** – if `True`, any trailing singleton dimensions will be removed (after the channel has been moved to the end). So if input is (H,W,D,C) and C==1, then it will be saved as (H,W,D). If D is also 1, it will be saved as (H,W). If `False`, image will always be saved as (H,W,D,C).

  * **spatial_ndim** – modifying the spatial dims if needed, so that output to have at least this number of spatial dims. If `None`, the output will have the same number of spatial dimensions as the input.

  * **contiguous** – if `True`, the output will be contiguous.

_classmethod _create_backend_obj(_data_array_ , _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L199-L205)[#](data.html#monai.data.ImageWriter.create_backend_obj
"Link to this definition")

    

Subclass should implement this method to return a backend-specific data
representation object. This method is used by `cls.write` and the input
`data_array` is assumed ‘channel-last’.

Return type:

    

`ndarray`

_classmethod _get_meta_info(_metadata
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L332-L343)[#](data.html#monai.data.ImageWriter.get_meta_info
"Link to this definition")

    

Extracts relevant meta information from the metadata object (using `.get`).
Optional keys are `"spatial_shape"`, `MetaKeys.AFFINE`, `"original_affine"`.

_classmethod _resample_if_needed(_data_array_ , _affine=None_ ,
_target_affine=None_ , _output_spatial_shape=None_ , _mode=bilinear_ ,
_padding_mode=border_ , _align_corners=False_ , _dtype= <class
'numpy.float64'>_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L207-L281)[#](data.html#monai.data.ImageWriter.resample_if_needed
"Link to this definition")

    

Convert the `data_array` into the coordinate system specified by
`target_affine`, from the current coordinate definition of `affine`.

If the transform between `affine` and `target_affine` could be achieved by
simply transposing and flipping `data_array`, no resampling will happen.
Otherwise, this function resamples `data_array` using the transformation
computed from `affine` and `target_affine`.

This function assumes the NIfTI dimension notations. Spatially it supports up
to three dimensions, that is, H, HW, HWD for 1D, 2D, 3D respectively. When
saving multiple time steps or multiple channels, time and/or modality axes
should be appended after the first three dimensions. For example, shape of 2D
eight-class segmentation probabilities to be saved could be (64, 64, 1, 8).
Also, data in shape (64, 64, 8) or (64, 64, 8, 1) will be considered as a
single-channel 3D image. The `convert_to_channel_last` method can be used to
convert the data to the format described here.

Note that the shape of the resampled `data_array` may subject to some rounding
errors. For example, resampling a 20x20 pixel image from pixel size (1.5,
1.5)-mm to (3.0, 3.0)-mm space will return a 10x10-pixel image. However,
resampling a 20x20-pixel image from pixel size (2.0, 2.0)-mm to (3.0, 3.0)-mm
space will output a 14x14-pixel image, where the image shape is rounded from
13.333x13.333 pixels. In this case `output_spatial_shape` could be specified
so that this function writes image data to a designated shape.

Parameters:

    

  * **data_array** – input data array to be converted.

  * **affine** – the current affine of `data_array`. Defaults to identity

  * **target_affine** – the designated affine of `data_array`. The actual output affine might be different from this value due to precision changes.

  * **output_spatial_shape** – spatial shape of the output image. This option is used when resampling is needed.

  * **mode** – available options are {`"bilinear"`, `"nearest"`, `"bicubic"`}. This option is used when resampling is needed. Interpolation mode to calculate output values. Defaults to `"bilinear"`. See also: <https://pytorch.org/docs/stable/nn.functional.html#grid-sample>

  * **padding_mode** – available options are {`"zeros"`, `"border"`, `"reflection"`}. This option is used when resampling is needed. Padding mode for outside grid values. Defaults to `"border"`. See also: <https://pytorch.org/docs/stable/nn.functional.html#grid-sample>

  * **align_corners** – boolean option of `grid_sample` to handle the corner convention. See also: <https://pytorch.org/docs/stable/nn.functional.html#grid-sample>

  * **dtype** – data type for resampling computation. Defaults to `np.float64` for best precision. If `None`, use the data type of input data. The output data type of this method is always `np.float32`.

write(_filename_ , _verbose =True_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L194-L197)[#](data.html#monai.data.ImageWriter.write
"Link to this definition")

    

subclass should implement this method to call the backend-specific writing
APIs.

### ITKWriter[#](data.html#itkwriter "Link to this heading")

_class _monai.data.ITKWriter(_output_dtype= <class 'numpy.float32'>_,
_affine_lps_to_ras=True_ , _**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L346-L530)[#](data.html#monai.data.ITKWriter
"Link to this definition")

    

Write data and metadata into files on disk using ITK-python.

    
    
    import numpy as np
    from monai.data import ITKWriter
    
    np_data = np.arange(48).reshape(3, 4, 4)
    
    # write as 3d spatial image no channel
    writer = ITKWriter(output_dtype=np.float32)
    writer.set_data_array(np_data, channel_dim=None)
    # optionally set metadata affine
    writer.set_metadata({"affine": np.eye(4), "original_affine": -1 * np.eye(4)})
    writer.write("test1.nii.gz")
    
    # write as 2d image, channel-first
    writer = ITKWriter(output_dtype=np.uint8)
    writer.set_data_array(np_data, channel_dim=0)
    writer.set_metadata({"spatial_shape": (5, 5)})
    writer.write("test1.png")
    

__init__(_output_dtype= <class 'numpy.float32'>_, _affine_lps_to_ras=True_ ,
_**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/utils/module.py#L376-L394)[#](data.html#monai.data.ITKWriter.__init__
"Link to this definition")

    

Parameters:

    

  * **output_dtype** – output data type.

  * **affine_lps_to_ras** – whether to convert the affine matrix from “LPS” to “RAS”. Defaults to `True`. Set to `True` to be consistent with `NibabelWriter`, otherwise the affine matrix is assumed already in the ITK convention. Set to `None` to use `data_array.meta[MetaKeys.SPACE]` to determine the flag.

  * **kwargs** – keyword arguments passed to `ImageWriter`.

The constructor will create `self.output_dtype` internally. `affine` and
`channel_dim` are initialized as instance members (default `None`, `0`):

>   * user-specified `affine` should be set in `set_metadata`,
>
>   * user-specified `channel_dim` should be set in `set_data_array`.
>
>

_classmethod _create_backend_obj(_data_array_ , _channel_dim=0_ ,
_affine=None_ , _dtype= <class 'numpy.float32'>_, _affine_lps_to_ras=True_ ,
_**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L478-L530)[#](data.html#monai.data.ITKWriter.create_backend_obj
"Link to this definition")

    

Create an ITK object from `data_array`. This method assumes a ‘channel-last’
`data_array`.

Parameters:

    

  * **data_array** – input data array.

  * **channel_dim** – channel dimension of the data array. This is used to create a Vector Image if it is not `None`.

  * **affine** – affine matrix of the data array. This is used to compute spacing, direction and origin.

  * **dtype** – output data type.

  * **affine_lps_to_ras** – whether to convert the affine matrix from “LPS” to “RAS”. Defaults to `True`. Set to `True` to be consistent with `NibabelWriter`, otherwise the affine matrix is assumed already in the ITK convention. Set to `None` to use `data_array.meta[MetaKeys.SPACE]` to determine the flag.

  * **kwargs** – keyword arguments. Current itk.GetImageFromArray will read `ttype` from this dictionary.

See also

  * [InsightSoftwareConsortium/ITK](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.2.1/Wrapping/Generators/Python/itk/support/extras.py#L389)

set_data_array(_data_array_ , _channel_dim =0_, _squeeze_end_dims =True_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L396-L423)[#](data.html#monai.data.ITKWriter.set_data_array
"Link to this definition")

    

Convert `data_array` into ‘channel-last’ numpy ndarray.

Parameters:

    

  * **data_array** – input data array with the channel dimension specified by `channel_dim`.

  * **channel_dim** – channel dimension of the data array. Defaults to 0. `None` indicates data without any channel dimension.

  * **squeeze_end_dims** – if `True`, any trailing singleton dimensions will be removed.

  * **kwargs** – keyword arguments passed to `self.convert_to_channel_last`, currently support `spatial_ndim` and `contiguous`, defauting to `3` and `False` respectively.

set_metadata(_meta_dict =None_, _resample =True_, _**
options_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L425-L449)[#](data.html#monai.data.ITKWriter.set_metadata
"Link to this definition")

    

Resample `self.dataobj` if needed. This method assumes `self.data_obj` is a
‘channel-last’ ndarray.

Parameters:

    

  * **meta_dict** – a metadata dictionary for affine, original affine and spatial shape information. Optional keys are `"spatial_shape"`, `"affine"`, `"original_affine"`.

  * **resample** – if `True`, the data will be resampled to the original affine (specified in `meta_dict`).

  * **options** – keyword arguments passed to `self.resample_if_needed`, currently support `mode`, `padding_mode`, `align_corners`, and `dtype`, defaulting to `bilinear`, `border`, `False`, and `np.float64` respectively.

write(_filename_ , _verbose =False_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L451-L476)[#](data.html#monai.data.ITKWriter.write
"Link to this definition")

    

Create an ITK object from `self.create_backend_obj(self.obj, ...)` and call
`itk.imwrite`.

Parameters:

    

  * **filename** (`Union`[`str`, `PathLike`]) – filename or PathLike object.

  * **verbose** (`bool`) – if `True`, log the progress.

  * **kwargs** – keyword arguments passed to `itk.imwrite`, currently support `compression` and `imageio`.

See also

  * [InsightSoftwareConsortium/ITK](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.2.1/Wrapping/Generators/Python/itk/support/extras.py#L809)

### NibabelWriter[#](data.html#nibabelwriter "Link to this heading")

_class _monai.data.NibabelWriter(_output_dtype= <class 'numpy.float32'>_,
_**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L533-L671)[#](data.html#monai.data.NibabelWriter
"Link to this definition")

    

Write data and metadata into files on disk using Nibabel.

    
    
    import numpy as np
    from monai.data import NibabelWriter
    
    np_data = np.arange(48).reshape(3, 4, 4)
    writer = NibabelWriter()
    writer.set_data_array(np_data, channel_dim=None)
    writer.set_metadata({"affine": np.eye(4), "original_affine": np.eye(4)})
    writer.write("test1.nii.gz", verbose=True)
    

__init__(_output_dtype= <class 'numpy.float32'>_,
_**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/utils/module.py#L554-L564)[#](data.html#monai.data.NibabelWriter.__init__
"Link to this definition")

    

Parameters:

    

  * **output_dtype** (`Union`[`dtype`, `type`, `str`, `None`]) – output data type.

  * **kwargs** – keyword arguments passed to `ImageWriter`.

The constructor will create `self.output_dtype` internally. `affine` is
initialized as instance members (default `None`), user-specified `affine`
should be set in `set_metadata`.

_classmethod _create_backend_obj(_data_array_ , _affine =None_, _dtype =None_,
_** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L640-L671)[#](data.html#monai.data.NibabelWriter.create_backend_obj
"Link to this definition")

    

Create an Nifti1Image object from `data_array`. This method assumes a
‘channel-last’ `data_array`.

Parameters:

    

  * **data_array** – input data array.

  * **affine** – affine matrix of the data array.

  * **dtype** – output data type.

  * **kwargs** – keyword arguments. Current `nib.nifti1.Nifti1Image` will read `header`, `extra`, `file_map` from this dictionary.

See also

  * <https://nipy.org/nibabel/reference/nibabel.nifti1.html#nibabel.nifti1.Nifti1Image>

set_data_array(_data_array_ , _channel_dim =0_, _squeeze_end_dims =True_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L566-L585)[#](data.html#monai.data.NibabelWriter.set_data_array
"Link to this definition")

    

Convert `data_array` into ‘channel-last’ numpy ndarray.

Parameters:

    

  * **data_array** – input data array with the channel dimension specified by `channel_dim`.

  * **channel_dim** – channel dimension of the data array. Defaults to 0. `None` indicates data without any channel dimension.

  * **squeeze_end_dims** – if `True`, any trailing singleton dimensions will be removed.

  * **kwargs** – keyword arguments passed to `self.convert_to_channel_last`, currently support `spatial_ndim`, defauting to `3`.

set_metadata(_meta_dict_ , _resample =True_, _**
options_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L587-L613)[#](data.html#monai.data.NibabelWriter.set_metadata
"Link to this definition")

    

Resample `self.dataobj` if needed. This method assumes `self.data_obj` is a
‘channel-last’ ndarray.

Parameters:

    

  * **meta_dict** – a metadata dictionary for affine, original affine and spatial shape information. Optional keys are `"spatial_shape"`, `"affine"`, `"original_affine"`.

  * **resample** – if `True`, the data will be resampled to the original affine (specified in `meta_dict`).

  * **options** – keyword arguments passed to `self.resample_if_needed`, currently support `mode`, `padding_mode`, `align_corners`, and `dtype`, defaulting to `bilinear`, `border`, `False`, and `np.float64` respectively.

write(_filename_ , _verbose =False_, _**
obj_kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L615-L638)[#](data.html#monai.data.NibabelWriter.write
"Link to this definition")

    

Create a Nibabel object from `self.create_backend_obj(self.obj, ...)` and call
`nib.save`.

Parameters:

    

  * **filename** (`Union`[`str`, `PathLike`]) – filename or PathLike object.

  * **verbose** (`bool`) – if `True`, log the progress.

  * **obj_kwargs** – keyword arguments passed to `self.create_backend_obj`,

See also

  * <https://nipy.org/nibabel/reference/nibabel.nifti1.html#nibabel.nifti1.save>

### PILWriter[#](data.html#pilwriter "Link to this heading")

_class _monai.data.PILWriter(_output_dtype= <class 'numpy.float32'>_,
_channel_dim=0_ , _scale=255_ ,
_**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L674-L862)[#](data.html#monai.data.PILWriter
"Link to this definition")

    

Write image data into files on disk using pillow.

It’s based on the Image module in PIL library:
<https://pillow.readthedocs.io/en/stable/reference/Image.html>

    
    
    import numpy as np
    from monai.data import PILWriter
    
    np_data = np.arange(48).reshape(3, 4, 4)
    writer = PILWriter(np.uint8)
    writer.set_data_array(np_data, channel_dim=0)
    writer.write("test1.png", verbose=True)
    

__init__(_output_dtype= <class 'numpy.float32'>_, _channel_dim=0_ ,
_scale=255_ , _**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/utils/module.py#L697-L709)[#](data.html#monai.data.PILWriter.__init__
"Link to this definition")

    

Parameters:

    

  * **output_dtype** – output data type.

  * **channel_dim** – channel dimension of the data array. Defaults to 0. `None` indicates data without any channel dimension.

  * **scale** – {`255`, `65535`} postprocess data by clipping to [0, 1] and scaling [0, 255] (uint8) or [0, 65535] (uint16). Default is None to disable scaling.

  * **kwargs** – keyword arguments passed to `ImageWriter`.

_classmethod _create_backend_obj(_data_array_ , _dtype =None_, _scale =255_,
_reverse_indexing =True_, _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L821-L862)[#](data.html#monai.data.PILWriter.create_backend_obj
"Link to this definition")

    

Create a PIL object from `data_array`.

Parameters:

    

  * **data_array** – input data array.

  * **dtype** – output data type.

  * **scale** – {`255`, `65535`} postprocess data by clipping to [0, 1] and scaling [0, 255] (uint8) or [0, 65535] (uint16). Default is None to disable scaling.

  * **reverse_indexing** – if `True`, the data array’s first two dimensions will be swapped.

  * **kwargs** – keyword arguments. Currently `PILImage.fromarray` will read `image_mode` from this dictionary, defaults to `None`.

See also

  * <https://pillow.readthedocs.io/en/stable/reference/Image.html>

_classmethod _get_meta_info(_metadata
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L784-L786)[#](data.html#monai.data.PILWriter.get_meta_info
"Link to this definition")

    

Extracts relevant meta information from the metadata object (using `.get`).
Optional keys are `"spatial_shape"`, `MetaKeys.AFFINE`, `"original_affine"`.

_classmethod _resample_and_clip(_data_array_ , _output_spatial_shape =None_,
_mode =bicubic_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L788-L819)[#](data.html#monai.data.PILWriter.resample_and_clip
"Link to this definition")

    

Resample `data_array` to `output_spatial_shape` if needed. :param data_array:
input data array. This method assumes the ‘channel-last’ format. :param
output_spatial_shape: output spatial shape. :param mode: interpolation mode,
default is `InterpolateMode.BICUBIC`.

set_data_array(_data_array_ , _channel_dim =0_, _squeeze_end_dims =True_,
_contiguous =False_, _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L711-L737)[#](data.html#monai.data.PILWriter.set_data_array
"Link to this definition")

    

Convert `data_array` into ‘channel-last’ numpy ndarray.

Parameters:

    

  * **data_array** – input data array with the channel dimension specified by `channel_dim`.

  * **channel_dim** – channel dimension of the data array. Defaults to 0. `None` indicates data without any channel dimension.

  * **squeeze_end_dims** – if `True`, any trailing singleton dimensions will be removed.

  * **contiguous** – if `True`, the data array will be converted to a contiguous array. Default is `False`.

  * **kwargs** – keyword arguments passed to `self.convert_to_channel_last`, currently support `spatial_ndim`, defauting to `2`.

set_metadata(_meta_dict =None_, _resample =True_, _**
options_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L739-L757)[#](data.html#monai.data.PILWriter.set_metadata
"Link to this definition")

    

Resample `self.dataobj` if needed. This method assumes `self.data_obj` is a
‘channel-last’ ndarray.

Parameters:

    

  * **meta_dict** – a metadata dictionary for affine, original affine and spatial shape information. Optional key is `"spatial_shape"`.

  * **resample** – if `True`, the data will be resampled to the spatial shape specified in `meta_dict`.

  * **options** – keyword arguments passed to `self.resample_if_needed`, currently support `mode`, defaulting to `bicubic`.

write(_filename_ , _verbose =False_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/image_writer.py#L759-L782)[#](data.html#monai.data.PILWriter.write
"Link to this definition")

    

Create a PIL image object from `self.create_backend_obj(self.obj, ...)` and
call `save`.

Parameters:

    

  * **filename** (`Union`[`str`, `PathLike`]) – filename or PathLike object.

  * **verbose** (`bool`) – if `True`, log the progress.

  * **kwargs** – optional keyword arguments passed to `self.create_backend_obj` currently support `reverse_indexing`, `image_mode`, defaulting to `True`, `None` respectively.

See also

  * <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save>

## Synthetic[#](data.html#module-monai.data.synthetic "Link to this heading")

monai.data.synthetic.create_test_image_2d(_height_ , _width_ , _num_objs =12_,
_rad_max =30_, _rad_min =5_, _noise_max =0.0_, _num_seg_classes =5_,
_channel_dim =None_, _random_state
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/synthetic.py#L21-L94)[#](data.html#monai.data.synthetic.create_test_image_2d
"Link to this definition")

    

Return a noisy 2D image with num_objs circles and a 2D mask image. The maximum
and minimum radii of the circles are given as rad_max and rad_min. The mask
will have num_seg_classes number of classes for segmentations labeled
sequentially from 1, plus a background class represented as 0. If noise_max is
greater than 0 then noise will be added to the image taken from the uniform
distribution on range [0,noise_max). If channel_dim is None, will create an
image without channel dimension, otherwise create an image with channel
dimension as first dim or last dim.

Parameters:

    

  * **height** – height of the image. The value should be larger than 2 * rad_max.

  * **width** – width of the image. The value should be larger than 2 * rad_max.

  * **num_objs** – number of circles to generate. Defaults to 12.

  * **rad_max** – maximum circle radius. Defaults to 30.

  * **rad_min** – minimum circle radius. Defaults to 5.

  * **noise_max** – if greater than 0 then noise will be added to the image taken from the uniform distribution on range [0,noise_max). Defaults to 0.

  * **num_seg_classes** – number of classes for segmentations. Defaults to 5.

  * **channel_dim** – if None, create an image without channel dimension, otherwise create an image with channel dimension as first dim or last dim. Defaults to None.

  * **random_state** – the random generator to use. Defaults to np.random.

Returns:

    

Randomised Numpy array with shape (height, width)

monai.data.synthetic.create_test_image_3d(_height_ , _width_ , _depth_ ,
_num_objs =12_, _rad_max =30_, _rad_min =5_, _noise_max =0.0_,
_num_seg_classes =5_, _channel_dim =None_, _random_state
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/synthetic.py#L97-L169)[#](data.html#monai.data.synthetic.create_test_image_3d
"Link to this definition")

    

Return a noisy 3D image and segmentation.

Parameters:

    

  * **height** – height of the image. The value should be larger than 2 * rad_max.

  * **width** – width of the image. The value should be larger than 2 * rad_max.

  * **depth** – depth of the image. The value should be larger than 2 * rad_max.

  * **num_objs** – number of circles to generate. Defaults to 12.

  * **rad_max** – maximum circle radius. Defaults to 30.

  * **rad_min** – minimum circle radius. Defaults to 5.

  * **noise_max** – if greater than 0 then noise will be added to the image taken from the uniform distribution on range [0,noise_max). Defaults to 0.

  * **num_seg_classes** – number of classes for segmentations. Defaults to 5.

  * **channel_dim** – if None, create an image without channel dimension, otherwise create an image with channel dimension as first dim or last dim. Defaults to None.

  * **random_state** – the random generator to use. Defaults to np.random.

Returns:

    

Randomised Numpy array with shape (height, width, depth)

See also

[`create_test_image_2d()`](data.html#monai.data.synthetic.create_test_image_2d
"monai.data.synthetic.create_test_image_2d")

## Ouput folder layout[#](data.html#module-monai.data.folder_layout "Link to
this heading")

_class _monai.data.folder_layout.FolderLayout(_output_dir_ , _postfix =''_,
_extension =''_, _parent =False_, _makedirs =False_, _data_root_dir
=''_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/folder_layout.py#L86-L168)[#](data.html#monai.data.folder_layout.FolderLayout
"Link to this definition")

    

A utility class to create organized filenames within `output_dir`. The
`filename` method could be used to create a filename following the folder
structure.

Example:

    
    
    from monai.data import FolderLayout
    
    layout = FolderLayout(
        output_dir="/test_run_1/",
        postfix="seg",
        extension="nii",
        makedirs=False)
    layout.filename(subject="Sub-A", idx="00", modality="T1")
    # return value: "/test_run_1/Sub-A_seg_00_modality-T1.nii"
    

The output filename is a string starting with a `subject` ID, and includes
additional information about a customized index and image modality. This
utility class doesn’t alter the underlying image data, but provides a
convenient way to create filenames.

__init__(_output_dir_ , _postfix =''_, _extension =''_, _parent =False_,
_makedirs =False_, _data_root_dir =''_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/folder_layout.py#L111-L135)[#](data.html#monai.data.folder_layout.FolderLayout.__init__
"Link to this definition")

    

Parameters:

    

  * **output_dir** (`Union`[`str`, `PathLike`]) – output directory.

  * **postfix** (`str`) – a postfix string for output file name appended to `subject`.

  * **extension** (`str`) – output file extension to be appended to the end of an output filename.

  * **parent** (`bool`) – whether to add a level of parent folder to contain each image to the output filename.

  * **makedirs** (`bool`) – whether to create the output parent directories if they do not exist.

  * **data_root_dir** (`Union`[`str`, `PathLike`]) – an optional PathLike object to preserve the folder structure of the input subject. Please see [`monai.data.utils.create_file_basename()`](data.html#monai.data.utils.create_file_basename "monai.data.utils.create_file_basename") for more details.

filename(_subject ='subject'_, _idx =None_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/folder_layout.py#L137-L168)[#](data.html#monai.data.folder_layout.FolderLayout.filename
"Link to this definition")

    

Create a filename based on the input `subject` and `idx`.

The output filename is formed as:

> `output_dir/[subject/]subject[_postfix][_idx][_key-value][ext]`

Parameters:

    

  * **subject** (`Union`[`str`, `PathLike`]) – subject name, used as the primary id of the output filename. When a PathLike object is provided, the base filename will be used as the subject name, the extension name of subject will be ignored, in favor of `extension` from this class’s constructor.

  * **idx** – additional index name of the image.

  * **kwargs** – additional keyword arguments to be used to form the output filename. The key-value pairs will be appended to the output filename as `f"_{k}-{v}"`.

Return type:

    

`Union`[`str`, `PathLike`]

_class
_monai.data.folder_layout.FolderLayoutBase[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/folder_layout.py#L35-L83)[#](data.html#monai.data.folder_layout.FolderLayoutBase
"Link to this definition")

    

Abstract base class to define a common interface for FolderLayout and derived
classes Mainly, defines the `filename(**kwargs) -> PathLike` function, which
must be defined by the deriving class.

Example:

    
    
    from monai.data import FolderLayoutBase
    
    class MyFolderLayout(FolderLayoutBase):
        def __init__(
            self,
            basepath: Path,
            extension: str = "",
            makedirs: bool = False
        ):
            self.basepath = basepath
            if not extension:
                self.extension = ""
            elif extension.startswith("."):
                self.extension = extension:
            else:
                self.extension = f".{extension}"
            self.makedirs = makedirs
    
        def filename(self, patient_no: int, image_name: str, **kwargs) -> Path:
            sub_path = self.basepath / patient_no
            if not sub_path.exists():
                sub_path.mkdir(parents=True)
    
            file = image_name
            for k, v in kwargs.items():
                file += f"_{k}-{v}"
    
            file +=  self.extension
            return sub_path / file
    

_abstract _filename(_** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/folder_layout.py#L77-L83)[#](data.html#monai.data.folder_layout.FolderLayoutBase.filename
"Link to this definition")

    

Create a filename with path based on the input kwargs. Abstract method,
implement your own.

Return type:

    

`Union`[`str`, `PathLike`]

monai.data.folder_layout.default_name_formatter(_metadict_ ,
_saver_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/folder_layout.py#L23-L32)[#](data.html#monai.data.folder_layout.default_name_formatter
"Link to this definition")

    

Returns a kwargs dict for
[`FolderLayout.filename()`](data.html#monai.data.folder_layout.FolderLayout.filename
"monai.data.folder_layout.FolderLayout.filename"), according to the input
metadata and SaveImage transform.

Return type:

    

`dict`

## Utilities[#](data.html#module-monai.data.utils "Link to this heading")

monai.data.utils.affine_to_spacing(_affine_ , _r=3_ , _dtype= <class
'float'>_, _suppress_zeros=True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L737-L760)[#](data.html#monai.data.utils.affine_to_spacing
"Link to this definition")

    

Computing the current spacing from the affine matrix.

Parameters:

    

  * **affine** (_~NdarrayTensor_) – a d x d affine matrix.

  * **r** (`int`) – indexing based on the spatial rank, spacing is computed from affine[:r, :r].

  * **dtype** – data type of the output.

  * **suppress_zeros** (`bool`) – whether to suppress the zeros with ones.

Return type:

    

~NdarrayTensor

Returns:

    

an r dimensional vector of spacing.

monai.data.utils.compute_importance_map(_patch_size_ , _mode =constant_,
_sigma_scale =0.125_, _device ='cpu'_, _dtype
=torch.float32_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1084-L1134)[#](data.html#monai.data.utils.compute_importance_map
"Link to this definition")

    

Get importance map for different weight modes.

Parameters:

    

  * **patch_size** – Size of the required importance map. This should be either H, W [,D].

  * **mode** – 

{`"constant"`, `"gaussian"`} How to blend output of overlapping windows.
Defaults to `"constant"`.

    * `"constant`”: gives equal weight to all predictions.

    * `"gaussian`”: gives less weight to predictions on edges of windows.

  * **sigma_scale** – Sigma_scale to calculate sigma for each dimension (sigma = sigma_scale * dim_size). Used for gaussian mode only.

  * **device** – Device to put importance map on.

  * **dtype** – Data type of the output importance map.

Raises:

    

**ValueError** – When `mode` is not one of [“constant”, “gaussian”].

Returns:

    

Tensor of size patch_size.

monai.data.utils.compute_shape_offset(_spatial_shape_ , _in_affine_ ,
_out_affine_ , _scale_extent =False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L875-L935)[#](data.html#monai.data.utils.compute_shape_offset
"Link to this definition")

    

Given input and output affine, compute appropriate shapes in the output space
based on the input array’s shape. This function also returns the offset to put
the shape in a good position with respect to the world coordinate system.

Parameters:

    

  * **spatial_shape** – input array’s shape

  * **in_affine** (_matrix_) – 2D affine matrix

  * **out_affine** (_matrix_) – 2D affine matrix

  * **scale_extent** – 

whether the scale is computed based on the spacing or the full extent of
voxels, for example, for a factor of 0.5 scaling:

option 1, “o” represents a voxel, scaling the distance between voxels:

    
        o--o--o
    o-----o
    

option 2, each voxel has a physical extent, scaling the full voxel extent:

    
        | voxel 1 | voxel 2 | voxel 3 | voxel 4 |
    |      voxel 1      |      voxel 2      |
    

Option 1 may reduce the number of locations that requiring interpolation.
Option 2 is more resolution agnostic, that is, resampling coordinates depend
on the scaling factor, not on the number of voxels. Default is False, using
option 1 to compute the shape and offset.

monai.data.utils.convert_tables_to_dicts(_dfs_ , _row_indices =None_,
_col_names =None_, _col_types =None_, _col_groups =None_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1433-L1507)[#](data.html#monai.data.utils.convert_tables_to_dicts
"Link to this definition")

    

Utility to join pandas tables, select rows, columns and generate groups. Will
return a list of dictionaries, every dictionary maps to a row of data in
tables.

Parameters:

    

  * **dfs** – data table in pandas Dataframe format. if providing a list of tables, will join them.

  * **row_indices** – indices of the expected rows to load. it should be a list, every item can be a int number or a range [start, end) for the indices. for example: row_indices=[[0, 100], 200, 201, 202, 300]. if None, load all the rows in the file.

  * **col_names** – names of the expected columns to load. if None, load all the columns.

  * **col_types** – 

type and default value to convert the loaded columns, if None, use original
data. it should be a dictionary, every item maps to an expected column, the
key is the column name and the value is None or a dictionary to define the
default value and data type. the supported keys in dictionary are: [“type”,
“default”], and note that the value of default should not be None. for
example:

    
        col_types = {
        "subject_id": {"type": str},
        "label": {"type": int, "default": 0},
        "ehr_0": {"type": float, "default": 0.0},
        "ehr_1": {"type": float, "default": 0.0},
    }
    

  * **col_groups** – args to group the loaded columns to generate a new column, it should be a dictionary, every item maps to a group, the key will be the new column name, the value is the names of columns to combine. for example: col_groups={“ehr”: [f”ehr_{i}” for i in range(10)], “meta”: [“meta_1”, “meta_2”]}

  * **kwargs** – additional arguments for pandas.merge() API to join tables.

monai.data.utils.correct_nifti_header_if_necessary(_img_nii_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L763-L783)[#](data.html#monai.data.utils.correct_nifti_header_if_necessary
"Link to this definition")

    

Check nifti object header’s format, update the header if needed. In the
updated image pixdim matches the affine.

Parameters:

    

**img_nii** – nifti image object

monai.data.utils.create_file_basename(_postfix_ , _input_file_name_ ,
_folder_path_ , _data_root_dir =''_, _separate_folder =True_, _patch_index
=None_, _makedirs =True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1009-L1081)[#](data.html#monai.data.utils.create_file_basename
"Link to this definition")

    

Utility function to create the path to the output file based on the input
filename (file name extension is not added by this function). When
`data_root_dir` is not specified, the output file name is:

> folder_path/input_file_name (no ext.) /input_file_name (no
> ext.)[_postfix][_patch_index]

otherwise the relative path with respect to `data_root_dir` will be inserted,
for example:

    
    
    from monai.data import create_file_basename
    create_file_basename(
        postfix="seg",
        input_file_name="/foo/bar/test1/image.png",
        folder_path="/output",
        data_root_dir="/foo/bar",
        separate_folder=True,
        makedirs=False)
    # output: /output/test1/image/image_seg
    

Parameters:

    

  * **postfix** (`str`) – output name’s postfix

  * **input_file_name** (`Union`[`str`, `PathLike`]) – path to the input image file.

  * **folder_path** (`Union`[`str`, `PathLike`]) – path for the output file

  * **data_root_dir** (`Union`[`str`, `PathLike`]) – if not empty, it specifies the beginning parts of the input file’s absolute path. This is used to compute input_file_rel_path, the relative path to the file from data_root_dir to preserve folder structure when saving in case there are files in different folders with the same file names.

  * **separate_folder** (`bool`) – whether to save every file in a separate folder, for example: if input filename is image.nii, postfix is seg and folder_path is output, if True, save as: output/image/image_seg.nii, if False, save as output/image_seg.nii. default to True.

  * **patch_index** – if not None, append the patch index to filename.

  * **makedirs** (`bool`) – whether to create the folder if it does not exist.

Return type:

    

`str`

monai.data.utils.decollate_batch(_batch_ , _detach =True_, _pad =True_,
_fill_value =None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L566-L666)[#](data.html#monai.data.utils.decollate_batch
"Link to this definition")

    

De-collate a batch of data (for example, as produced by a DataLoader).

Returns a list of structures with the original tensor’s 0-th dimension sliced
into elements using torch.unbind.

Images originally stored as (B,C,H,W,[D]) will be returned as (C,H,W,[D]).
Other information, such as metadata, may have been stored in a list (or a list
inside nested dictionaries). In this case we return the element of the list
corresponding to the batch idx.

Return types aren’t guaranteed to be the same as the original, since numpy
arrays will have been converted to torch.Tensor, sequences may be converted to
lists of tensors, mappings may be converted into dictionaries.

For example:

    
    
    batch_data = {
        "image": torch.rand((2,1,10,10)),
        DictPostFix.meta("image"): {"scl_slope": torch.Tensor([0.0, 0.0])}
    }
    out = decollate_batch(batch_data)
    print(len(out))
    >>> 2
    
    print(out[0])
    >>> {'image': tensor([[[4.3549e-01...43e-01]]]), DictPostFix.meta("image"): {'scl_slope': 0.0}}
    
    batch_data = [torch.rand((2,1,10,10)), torch.rand((2,3,5,5))]
    out = decollate_batch(batch_data)
    print(out[0])
    >>> [tensor([[[4.3549e-01...43e-01]]], tensor([[[5.3435e-01...45e-01]]])]
    
    batch_data = torch.rand((2,1,10,10))
    out = decollate_batch(batch_data)
    print(out[0])
    >>> tensor([[[4.3549e-01...43e-01]]])
    
    batch_data = {
        "image": [1, 2, 3], "meta": [4, 5],  # undetermined batch size
    }
    out = decollate_batch(batch_data, pad=True, fill_value=0)
    print(out)
    >>> [{'image': 1, 'meta': 4}, {'image': 2, 'meta': 5}, {'image': 3, 'meta': 0}]
    out = decollate_batch(batch_data, pad=False)
    print(out)
    >>> [{'image': 1, 'meta': 4}, {'image': 2, 'meta': 5}]
    

Parameters:

    

  * **batch** – data to be de-collated.

  * **detach** (`bool`) – whether to detach the tensors. Scalars tensors will be detached into number types instead of torch tensors.

  * **pad** – when the items in a batch indicate different batch size, whether to pad all the sequences to the longest. If False, the batch size will be the length of the shortest sequence.

  * **fill_value** – when pad is True, the fillvalue to use when padding, defaults to None.

monai.data.utils.dense_patch_slices(_image_size_ , _patch_size_ ,
_scan_interval_ , _return_slice =True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L166-L206)[#](data.html#monai.data.utils.dense_patch_slices
"Link to this definition")

    

Enumerate all slices defining ND patches of size patch_size from an image_size
input image.

Parameters:

    

  * **image_size** (`Sequence`[`int`]) – dimensions of image to iterate over

  * **patch_size** (`Sequence`[`int`]) – size of patches to generate slices

  * **scan_interval** (`Sequence`[`int`]) – dense patch sampling interval

  * **return_slice** (`bool`) – whether to return a list of slices (or tuples of indices), defaults to True

Return type:

    

`list`[`tuple`[`slice`, …]]

Returns:

    

a list of slice objects defining each patch

monai.data.utils.get_extra_metadata_keys()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1555-L1586)[#](data.html#monai.data.utils.get_extra_metadata_keys
"Link to this definition")

    

Get a list of unnecessary keys for metadata that can be removed.

Return type:

    

`list`[`str`]

Returns:

    

List of keys to be removed.

monai.data.utils.get_random_patch(_dims_ , _patch_size_ , _rand_state
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L107-L129)[#](data.html#monai.data.utils.get_random_patch
"Link to this definition")

    

Returns a tuple of slices to define a random patch in an array of shape dims
with size patch_size or the as close to it as possible within the given
dimension. It is expected that patch_size is a valid patch for a source of
shape dims as returned by get_valid_patch_size.

Parameters:

    

  * **dims** – shape of source array

  * **patch_size** – shape of patch size to generate

  * **rand_state** – a random state object to generate random numbers from

Returns:

    

a tuple of slice objects defining the patch

Return type:

    

(tuple of slice)

monai.data.utils.get_valid_patch_size(_image_size_ ,
_patch_size_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L343-L354)[#](data.html#monai.data.utils.get_valid_patch_size
"Link to this definition")

    

Given an image of dimensions image_size, return a patch size tuple taking the
dimension from patch_size if this is not 0/None. Otherwise, or if patch_size
is shorter than image_size, the dimension from image_size is taken. This
ensures the returned patch size is within the bounds of image_size. If
patch_size is a single number this is interpreted as a patch of the same
dimensionality of image_size with that size in each dimension.

monai.data.utils.is_no_channel(_val_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1589-L1597)[#](data.html#monai.data.utils.is_no_channel
"Link to this definition")

    

Returns whether val indicates “no_channel”, for MetaKeys.ORIGINAL_CHANNEL_DIM.

Return type:

    

`bool`

monai.data.utils.is_supported_format(_filename_ ,
_suffixes_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1137-L1154)[#](data.html#monai.data.utils.is_supported_format
"Link to this definition")

    

Verify whether the specified file or files format match supported suffixes. If
supported suffixes is None, skip the verification and return True.

Parameters:

    

  * **filename** – file name or a list of file names to read. if a list of files, verify all the suffixes.

  * **suffixes** – all the supported image suffixes of current reader, must be a list of lower case suffixes.

monai.data.utils.iter_patch(_arr_ , _patch_size =0_, _start_pos =()_, _overlap
=0.0_, _copy_back =True_, _mode =wrap_, _**
pad_opts_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L257-L340)[#](data.html#monai.data.utils.iter_patch
"Link to this definition")

    

Yield successive patches from arr of size patch_size. The iteration can start
from position start_pos in arr but drawing from a padded array extended by the
patch_size in each dimension (so these coordinates can be negative to start in
the padded region). If copy_back is True the values from each patch are
written back to arr.

Parameters:

    

  * **arr** – array to iterate over

  * **patch_size** – size of patches to generate slices for, 0 or None selects whole dimension. For 0 or None, padding and overlap ratio of the corresponding dimension will be 0.

  * **start_pos** – starting position in the array, default is 0 for each dimension

  * **overlap** – the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0). If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.

  * **copy_back** – if True data from the yielded patches is copied back to arr once the generator completes

  * **mode** – available modes: (Numpy) {`"constant"`, `"edge"`, `"linear_ramp"`, `"maximum"`, `"mean"`, `"median"`, `"minimum"`, `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`} (PyTorch) {`"constant"`, `"reflect"`, `"replicate"`, `"circular"`}. One of the listed string values or a user supplied function. If None, no wrapping is performed. Defaults to `"wrap"`. See also: <https://numpy.org/doc/stable/reference/generated/numpy.pad.html> <https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html> requires pytorch >= 1.10 for best compatibility.

  * **pad_opts** – other arguments for the np.pad or torch.pad function. note that np.pad treats channel dimension as the first dimension.

Yields:

    

Patches of array data from arr which are views into a padded array which can
be modified, if copy_back is True these changes will be reflected in arr once
the iteration completes.

Note

coordinate format is:

> [1st_dim_start, 1st_dim_end,
>  
>
> 2nd_dim_start, 2nd_dim_end, …, Nth_dim_start, Nth_dim_end]]

monai.data.utils.iter_patch_position(_image_size_ , _patch_size_ , _start_pos
=()_, _overlap =0.0_, _padded =False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L209-L254)[#](data.html#monai.data.utils.iter_patch_position
"Link to this definition")

    

Yield successive tuples of upper left corner of patches of size patch_size
from an array of dimensions image_size. The iteration starts from position
start_pos in the array, or starting at the origin if this isn’t provided. Each
patch is chosen in a contiguous grid using a rwo-major ordering.

Parameters:

    

  * **image_size** – dimensions of array to iterate over

  * **patch_size** – size of patches to generate slices for, 0 or None selects whole dimension

  * **start_pos** – starting position in the array, default is 0 for each dimension

  * **overlap** – the amount of overlap of neighboring patches in each dimension. Either a float or list of floats between 0.0 and 1.0 to define relative overlap to patch size, or an int or list of ints to define number of pixels for overlap. If only one float/int number is given, it will be applied to all dimensions. Defaults to 0.0.

  * **padded** – if the image is padded so the patches can go beyond the borders. Defaults to False.

Yields:

    

Tuples of positions defining the upper left corner of each patch

monai.data.utils.iter_patch_slices(_image_size_ , _patch_size_ , _start_pos
=()_, _overlap =0.0_, _padded =True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L132-L163)[#](data.html#monai.data.utils.iter_patch_slices
"Link to this definition")

    

Yield successive tuples of slices defining patches of size patch_size from an
array of dimensions image_size. The iteration starts from position start_pos
in the array, or starting at the origin if this isn’t provided. Each patch is
chosen in a contiguous grid using a rwo-major ordering.

Parameters:

    

  * **image_size** – dimensions of array to iterate over

  * **patch_size** – size of patches to generate slices for, 0 or None selects whole dimension

  * **start_pos** – starting position in the array, default is 0 for each dimension

  * **overlap** – the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0). If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.

  * **padded** – if the image is padded so the patches can go beyond the borders. Defaults to False.

Yields:

    

Tuples of slice objects defining each patch

monai.data.utils.json_hashing(_item_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1385-L1402)[#](data.html#monai.data.utils.json_hashing
"Link to this definition")

    

Parameters:

    

**item** – data item to be hashed

Returns: the corresponding hash key

Return type:

    

`bytes`

monai.data.utils.list_data_collate(_batch_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L483-L537)[#](data.html#monai.data.utils.list_data_collate
"Link to this definition")

    

Enhancement for PyTorch DataLoader default collate. If dataset already returns
a list of batch data that generated in transforms, need to merge all data to 1
list. Then it’s same as the default collate behavior.

Note

Need to use this collate if apply some transforms that can generate batch
data.

monai.data.utils.no_collation(_x_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L695-L699)[#](data.html#monai.data.utils.no_collation
"Link to this definition")

    

No any collation operation.

monai.data.utils.orientation_ras_lps(_affine_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1510-L1523)[#](data.html#monai.data.utils.orientation_ras_lps
"Link to this definition")

    

Convert the `affine` between the RAS and LPS orientation by flipping the first
two spatial dimensions.

Parameters:

    

**affine** (_~NdarrayTensor_) – a 2D affine matrix.

Return type:

    

~NdarrayTensor

monai.data.utils.pad_list_data_collate(_batch_ , _method =symmetric_, _mode
=constant_, _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L669-L692)[#](data.html#monai.data.utils.pad_list_data_collate
"Link to this definition")

    

Function version of
[`monai.transforms.croppad.batch.PadListDataCollate`](transforms.html#monai.transforms.PadListDataCollate
"monai.transforms.croppad.batch.PadListDataCollate").

Same as MONAI’s `list_data_collate`, except any tensors are centrally padded
to match the shape of the biggest tensor in each dimension. This transform is
useful if some of the applied transforms generate batch data of different
sizes.

This can be used on both list and dictionary data. Note that in the case of
the dictionary data, this decollate function may add the transform information
of PadListDataCollate to the list of invertible transforms if input batch have
different spatial shape, so need to call static method:
monai.transforms.croppad.batch.PadListDataCollate.inverse before inverting
other transforms.

Parameters:

    

  * **batch** (`Sequence`) – batch of data to pad-collate

  * **method** (`str`) – padding method (see [`monai.transforms.SpatialPad`](transforms.html#monai.transforms.SpatialPad "monai.transforms.SpatialPad"))

  * **mode** (`str`) – padding mode (see [`monai.transforms.SpatialPad`](transforms.html#monai.transforms.SpatialPad "monai.transforms.SpatialPad"))

  * **kwargs** – other arguments for the np.pad or torch.pad function. note that np.pad treats channel dimension as the first dimension.

monai.data.utils.partition_dataset(_data_ , _ratios =None_, _num_partitions
=None_, _shuffle =False_, _seed =0_, _drop_last =False_, _even_divisible
=False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1157-L1264)[#](data.html#monai.data.utils.partition_dataset
"Link to this definition")

    

Split the dataset into N partitions. It can support shuffle based on specified
random seed. Will return a set of datasets, every dataset contains 1 partition
of original dataset. And it can split the dataset based on specified ratios or
evenly split into num_partitions. Refer to:
<https://pytorch.org/docs/stable/distributed.html#module-
torch.distributed.launch>.

Note

It also can be used to partition dataset for ranks in distributed training.
For example, partition dataset before training and use CacheDataset, every
rank trains with its own data. It can avoid duplicated caching content in each
rank, but will not do global shuffle before every epoch:

    
    
    data_partition = partition_dataset(
        data=train_files,
        num_partitions=dist.get_world_size(),
        shuffle=True,
        even_divisible=True,
    )[dist.get_rank()]
    
    train_ds = SmartCacheDataset(
        data=data_partition,
        transform=train_transforms,
        replace_rate=0.2,
        cache_num=15,
    )
    

Parameters:

    

  * **data** – input dataset to split, expect a list of data.

  * **ratios** – a list of ratio number to split the dataset, like [8, 1, 1].

  * **num_partitions** – expected number of the partitions to evenly split, only works when ratios not specified.

  * **shuffle** – whether to shuffle the original dataset before splitting.

  * **seed** – random seed to shuffle the dataset, only works when shuffle is True.

  * **drop_last** – only works when even_divisible is False and no ratios specified. if True, will drop the tail of the data to make it evenly divisible across partitions. if False, will add extra indices to make the data evenly divisible across partitions.

  * **even_divisible** – if True, guarantee every partition has same length.

Examples:

    
    
    >>> data = [1, 2, 3, 4, 5]
    >>> partition_dataset(data, ratios=[0.6, 0.2, 0.2], shuffle=False)
    [[1, 2, 3], [4], [5]]
    >>> partition_dataset(data, num_partitions=2, shuffle=False)
    [[1, 3, 5], [2, 4]]
    >>> partition_dataset(data, num_partitions=2, shuffle=False, even_divisible=True, drop_last=True)
    [[1, 3], [2, 4]]
    >>> partition_dataset(data, num_partitions=2, shuffle=False, even_divisible=True, drop_last=False)
    [[1, 3, 5], [2, 4, 1]]
    >>> partition_dataset(data, num_partitions=2, shuffle=False, even_divisible=False, drop_last=False)
    [[1, 3, 5], [2, 4]]
    

monai.data.utils.partition_dataset_classes(_data_ , _classes_ , _ratios
=None_, _num_partitions =None_, _shuffle =False_, _seed =0_, _drop_last
=False_, _even_divisible =False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1267-L1332)[#](data.html#monai.data.utils.partition_dataset_classes
"Link to this definition")

    

Split the dataset into N partitions based on the given class labels. It can
make sure the same ratio of classes in every partition. Others are same as
[`monai.data.partition_dataset`](data.html#monai.data.partition_dataset
"monai.data.partition_dataset").

Parameters:

    

  * **data** – input dataset to split, expect a list of data.

  * **classes** – a list of labels to help split the data, the length must match the length of data.

  * **ratios** – a list of ratio number to split the dataset, like [8, 1, 1].

  * **num_partitions** – expected number of the partitions to evenly split, only works when no ratios.

  * **shuffle** – whether to shuffle the original dataset before splitting.

  * **seed** – random seed to shuffle the dataset, only works when shuffle is True.

  * **drop_last** – only works when even_divisible is False and no ratios specified. if True, will drop the tail of the data to make it evenly divisible across partitions. if False, will add extra indices to make the data evenly divisible across partitions.

  * **even_divisible** – if True, guarantee every partition has same length.

Examples:

    
    
    >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    >>> classes = [2, 0, 2, 1, 3, 2, 2, 0, 2, 0, 3, 3, 1, 3]
    >>> partition_dataset_classes(data, classes, shuffle=False, ratios=[2, 1])
    [[2, 8, 4, 1, 3, 6, 5, 11, 12], [10, 13, 7, 9, 14]]
    

monai.data.utils.pickle_hashing(_item_ , _protocol
=5_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1405-L1423)[#](data.html#monai.data.utils.pickle_hashing
"Link to this definition")

    

Parameters:

    

  * **item** – data item to be hashed

  * **protocol** – protocol version used for pickling, defaults to pickle.HIGHEST_PROTOCOL.

Returns: the corresponding hash key

Return type:

    

`bytes`

monai.data.utils.rectify_header_sform_qform(_img_nii_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L786-L820)[#](data.html#monai.data.utils.rectify_header_sform_qform
"Link to this definition")

    

Look at the sform and qform of the nifti object and correct it if any
incompatibilities with pixel dimensions

Adapted from
[NifTK/NiftyNet](https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/io/misc_io.py)

Parameters:

    

**img_nii** – nifti image object

monai.data.utils.remove_extra_metadata(_meta_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1541-L1552)[#](data.html#monai.data.utils.remove_extra_metadata
"Link to this definition")

    

Remove extra metadata from the dictionary. Operates in-place so nothing is
returned.

Parameters:

    

**meta** (`dict`) – dictionary containing metadata to be modified.

Return type:

    

`None`

Returns:

    

None

monai.data.utils.remove_keys(_data_ ,
_keys_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1526-L1538)[#](data.html#monai.data.utils.remove_keys
"Link to this definition")

    

Remove keys from a dictionary. Operates in-place so nothing is returned.

Parameters:

    

  * **data** (`dict`) – dictionary to be modified.

  * **keys** (`list`[`str`]) – keys to be deleted from dictionary.

Return type:

    

`None`

Returns:

    

None

monai.data.utils.reorient_spatial_axes(_data_shape_ , _init_affine_ ,
_target_affine_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L985-L1006)[#](data.html#monai.data.utils.reorient_spatial_axes
"Link to this definition")

    

Given the input `init_affine`, compute the orientation transform between it
and `target_affine` by rearranging/flipping the axes.

Returns the orientation transform and the updated affine (tensor or ndarray
depends on the input `affine` data type). Note that this function requires
external module `nibabel.orientations`.

Return type:

    

`tuple`[`ndarray`, `Union`[`ndarray`, `Tensor`]]

monai.data.utils.resample_datalist(_data_ , _factor_ , _random_pick =False_,
_seed =0_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1335-L1357)[#](data.html#monai.data.utils.resample_datalist
"Link to this definition")

    

Utility function to resample the loaded datalist for training, for example: If
factor < 1.0, randomly pick part of the datalist and set to Dataset, useful to
quickly test the program. If factor > 1.0, repeat the datalist to enhance the
Dataset.

Parameters:

    

  * **data** (`Sequence`) – original datalist to scale.

  * **factor** (`float`) – scale factor for the datalist, for example, factor=4.5, repeat the datalist 4 times and plus 50% of the original datalist.

  * **random_pick** (`bool`) – whether to randomly pick data if scale factor has decimal part.

  * **seed** (`int`) – random seed to randomly pick data.

monai.data.utils.select_cross_validation_folds(_partitions_ ,
_folds_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1360-L1382)[#](data.html#monai.data.utils.select_cross_validation_folds
"Link to this definition")

    

Select cross validation data based on data partitions and specified fold
index. if a list of fold indices is provided, concatenate the partitions of
these folds.

Parameters:

    

  * **partitions** – a sequence of datasets, each item is a iterable

  * **folds** – the indices of the partitions to be combined.

Returns:

    

A list of combined datasets.

Example:

    
    
    >>> partitions = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    >>> select_cross_validation_folds(partitions, 2)
    [5, 6]
    >>> select_cross_validation_folds(partitions, [1, 2])
    [3, 4, 5, 6]
    >>> select_cross_validation_folds(partitions, [-1, 2])
    [9, 10, 5, 6]
    

monai.data.utils.set_rnd(_obj_ , _seed_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L712-L734)[#](data.html#monai.data.utils.set_rnd
"Link to this definition")

    

Set seed or random state for all randomizable properties of obj.

Parameters:

    

  * **obj** – object to set seed or random state for.

  * **seed** (`int`) – set the random state with an integer seed.

Return type:

    

`int`

monai.data.utils.sorted_dict(_item_ , _key =None_, _reverse
=False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1426-L1430)[#](data.html#monai.data.utils.sorted_dict
"Link to this definition")

    

Return a new sorted dictionary from the item.

monai.data.utils.to_affine_nd(_r_ , _affine_ , _dtype= <class
'numpy.float64'>_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L938-L982)[#](data.html#monai.data.utils.to_affine_nd
"Link to this definition")

    

Using elements from affine, to create a new affine matrix by assigning the
rotation/zoom/scaling matrix and the translation vector.

When `r` is an integer, output is an (r+1)x(r+1) matrix, where the top left
kxk elements are copied from `affine`, the last column of the output affine is
copied from `affine`’s last column. k is determined by min(r, len(affine) -
1).

When `r` is an affine matrix, the output has the same shape as `r`, and the
top left kxk elements are copied from `affine`, the last column of the output
affine is copied from `affine`’s last column. k is determined by min(len(r) -
1, len(affine) - 1).

Parameters:

    

  * **r** (_int_ _or_ _matrix_) – number of spatial dimensions or an output affine to be filled.

  * **affine** (_matrix_) – 2D affine matrix

  * **dtype** – data type of the output array.

Raises:

    

  * **ValueError** – When `affine` dimensions is not 2.

  * **ValueError** – When `r` is nonpositive.

Returns:

    

an (r+1) x (r+1) matrix (tensor or ndarray depends on the input `affine` data
type)

monai.data.utils.worker_init_fn(_worker_id_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L702-L709)[#](data.html#monai.data.utils.worker_init_fn
"Link to this definition")

    

Callback function for PyTorch DataLoader worker_init_fn. It can set different
random seed for the transforms in different workers.

Return type:

    

`None`

monai.data.utils.zoom_affine(_affine_ , _scale_ , _diagonal
=True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L823-L872)[#](data.html#monai.data.utils.zoom_affine
"Link to this definition")

    

To make column norm of affine the same as scale. If diagonal is False, returns
an affine that combines orthogonal rotation and the new scale. This is done by
first decomposing affine, then setting the zoom factors to scale, and
composing a new affine; the shearing factors are removed. If diagonal is True,
returns a diagonal matrix, the scaling factors are set to the diagonal
elements. This function always return an affine with zero translations.

Parameters:

    

  * **affine** (_nxn matrix_) – a square matrix.

  * **scale** – new scaling factor along each dimension. if the components of the scale are non-positive values, will use the corresponding components of the original pixdim, which is computed from the affine.

  * **diagonal** – whether to return a diagonal scaling matrix. Defaults to True.

Raises:

    

  * **ValueError** – When `affine` is not a square matrix.

  * **ValueError** – When `scale` contains a nonpositive scalar.

Returns:

    

the updated n x n affine.

### Partition Dataset[#](data.html#partition-dataset "Link to this heading")

monai.data.partition_dataset(_data_ , _ratios =None_, _num_partitions =None_,
_shuffle =False_, _seed =0_, _drop_last =False_, _even_divisible
=False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1157-L1264)[#](data.html#monai.data.partition_dataset
"Link to this definition")

    

Split the dataset into N partitions. It can support shuffle based on specified
random seed. Will return a set of datasets, every dataset contains 1 partition
of original dataset. And it can split the dataset based on specified ratios or
evenly split into num_partitions. Refer to:
<https://pytorch.org/docs/stable/distributed.html#module-
torch.distributed.launch>.

Note

It also can be used to partition dataset for ranks in distributed training.
For example, partition dataset before training and use CacheDataset, every
rank trains with its own data. It can avoid duplicated caching content in each
rank, but will not do global shuffle before every epoch:

    
    
    data_partition = partition_dataset(
        data=train_files,
        num_partitions=dist.get_world_size(),
        shuffle=True,
        even_divisible=True,
    )[dist.get_rank()]
    
    train_ds = SmartCacheDataset(
        data=data_partition,
        transform=train_transforms,
        replace_rate=0.2,
        cache_num=15,
    )
    

Parameters:

    

  * **data** – input dataset to split, expect a list of data.

  * **ratios** – a list of ratio number to split the dataset, like [8, 1, 1].

  * **num_partitions** – expected number of the partitions to evenly split, only works when ratios not specified.

  * **shuffle** – whether to shuffle the original dataset before splitting.

  * **seed** – random seed to shuffle the dataset, only works when shuffle is True.

  * **drop_last** – only works when even_divisible is False and no ratios specified. if True, will drop the tail of the data to make it evenly divisible across partitions. if False, will add extra indices to make the data evenly divisible across partitions.

  * **even_divisible** – if True, guarantee every partition has same length.

Examples:

    
    
    >>> data = [1, 2, 3, 4, 5]
    >>> partition_dataset(data, ratios=[0.6, 0.2, 0.2], shuffle=False)
    [[1, 2, 3], [4], [5]]
    >>> partition_dataset(data, num_partitions=2, shuffle=False)
    [[1, 3, 5], [2, 4]]
    >>> partition_dataset(data, num_partitions=2, shuffle=False, even_divisible=True, drop_last=True)
    [[1, 3], [2, 4]]
    >>> partition_dataset(data, num_partitions=2, shuffle=False, even_divisible=True, drop_last=False)
    [[1, 3, 5], [2, 4, 1]]
    >>> partition_dataset(data, num_partitions=2, shuffle=False, even_divisible=False, drop_last=False)
    [[1, 3, 5], [2, 4]]
    

### Partition Dataset based on classes[#](data.html#partition-dataset-based-
on-classes "Link to this heading")

monai.data.partition_dataset_classes(_data_ , _classes_ , _ratios =None_,
_num_partitions =None_, _shuffle =False_, _seed =0_, _drop_last =False_,
_even_divisible =False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/utils.py#L1267-L1332)[#](data.html#monai.data.partition_dataset_classes
"Link to this definition")

    

Split the dataset into N partitions based on the given class labels. It can
make sure the same ratio of classes in every partition. Others are same as
[`monai.data.partition_dataset`](data.html#monai.data.partition_dataset
"monai.data.partition_dataset").

Parameters:

    

  * **data** – input dataset to split, expect a list of data.

  * **classes** – a list of labels to help split the data, the length must match the length of data.

  * **ratios** – a list of ratio number to split the dataset, like [8, 1, 1].

  * **num_partitions** – expected number of the partitions to evenly split, only works when no ratios.

  * **shuffle** – whether to shuffle the original dataset before splitting.

  * **seed** – random seed to shuffle the dataset, only works when shuffle is True.

  * **drop_last** – only works when even_divisible is False and no ratios specified. if True, will drop the tail of the data to make it evenly divisible across partitions. if False, will add extra indices to make the data evenly divisible across partitions.

  * **even_divisible** – if True, guarantee every partition has same length.

Examples:

    
    
    >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    >>> classes = [2, 0, 2, 1, 3, 2, 2, 0, 2, 0, 3, 3, 1, 3]
    >>> partition_dataset_classes(data, classes, shuffle=False, ratios=[2, 1])
    [[2, 8, 4, 1, 3, 6, 5, 11, 12], [10, 13, 7, 9, 14]]
    

### DistributedSampler[#](data.html#distributedsampler "Link to this heading")

_class _monai.data.DistributedSampler(_dataset_ , _even_divisible =True_,
_num_replicas =None_, _rank =None_, _shuffle =True_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/samplers.py#L23-L61)[#](data.html#monai.data.DistributedSampler
"Link to this definition")

    

Enhance PyTorch DistributedSampler to support non-evenly divisible sampling.

Parameters:

    

  * **dataset** – Dataset used for sampling.

  * **even_divisible** – if False, different ranks can have different data length. for example, input data: [1, 2, 3, 4, 5], rank 0: [1, 3, 5], rank 1: [2, 4].

  * **num_replicas** – number of processes participating in distributed training. by default, world_size is retrieved from the current distributed group.

  * **rank** – rank of the current process within num_replicas. by default, rank is retrieved from the current distributed group.

  * **shuffle** – if True, sampler will shuffle the indices, default to True.

  * **kwargs** – additional arguments for DistributedSampler super class, can be seed and drop_last.

More information about DistributedSampler, please check:
<https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler>.

###
DistributedWeightedRandomSampler[#](data.html#distributedweightedrandomsampler
"Link to this heading")

_class _monai.data.DistributedWeightedRandomSampler(_dataset_ , _weights_ ,
_num_samples_per_rank =None_, _generator =None_, _even_divisible =True_,
_num_replicas =None_, _rank =None_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/samplers.py#L64-L115)[#](data.html#monai.data.DistributedWeightedRandomSampler
"Link to this definition")

    

Extend the DistributedSampler to support weighted sampling. Refer to
torch.utils.data.WeightedRandomSampler, for more details please check:
<https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler>.

Parameters:

    

  * **dataset** – Dataset used for sampling.

  * **weights** – a sequence of weights, not necessary summing up to one, length should exactly match the full dataset.

  * **num_samples_per_rank** – number of samples to draw for every rank, sample from the distributed subset of dataset. if None, default to the length of dataset split by DistributedSampler.

  * **generator** – PyTorch Generator used in sampling.

  * **even_divisible** – if False, different ranks can have different data length. for example, input data: [1, 2, 3, 4, 5], rank 0: [1, 3, 5], rank 1: [2, 4].’

  * **num_replicas** – number of processes participating in distributed training. by default, world_size is retrieved from the current distributed group.

  * **rank** – rank of the current process within num_replicas. by default, rank is retrieved from the current distributed group.

  * **kwargs** – additional arguments for DistributedSampler super class, can be seed and drop_last.

### DatasetSummary[#](data.html#datasetsummary "Link to this heading")

_class _monai.data.DatasetSummary(_dataset_ , _image_key ='image'_, _label_key
='label'_, _meta_key =None_, _meta_key_postfix ='meta_dict'_, _num_workers
=0_, _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataset_summary.py#L31-L215)[#](data.html#monai.data.DatasetSummary
"Link to this definition")

    

This class provides a way to calculate a reasonable output voxel spacing
according to the input dataset. The achieved values can used to resample the
input in 3d segmentation tasks (like using as the pixdim parameter in
monai.transforms.Spacingd). In addition, it also supports to compute the mean,
std, min and max intensities of the input, and these statistics are helpful
for image normalization (as parameters of
monai.transforms.ScaleIntensityRanged and
monai.transforms.NormalizeIntensityd).

The algorithm for calculation refers to: [Automated Design of Deep Learning
Methods for Biomedical Image Segmentation](https://arxiv.org/abs/1904.08128).

### Decathlon Datalist[#](data.html#decathlon-datalist "Link to this heading")

monai.data.load_decathlon_datalist(_data_list_file_path_ , _is_segmentation
=True_, _data_list_key ='training'_, _base_dir
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/decathlon_datalist.py#L87-L133)[#](data.html#monai.data.load_decathlon_datalist
"Link to this definition")

    

Load image/label paths of decathlon challenge from JSON file

Json file is similar to what you get from <http://medicaldecathlon.com/> Those
dataset.json files

Parameters:

    

  * **data_list_file_path** – the path to the json file of datalist.

  * **is_segmentation** – whether the datalist is for segmentation task, default is True.

  * **data_list_key** – the key to get a list of dictionary to be used, default is “training”.

  * **base_dir** – the base directory of the dataset, if None, use the datalist directory.

Raises:

    

  * **ValueError** – When `data_list_file_path` does not point to a file.

  * **ValueError** – When `data_list_key` is not specified in the data list file.

Returns a list of data items, each of which is a dict keyed by element names,
for example:

    
    
    [
        {'image': '/workspace/data/chest_19.nii.gz',  'label': 0},
        {'image': '/workspace/data/chest_31.nii.gz',  'label': 1}
    ]
    

monai.data.load_decathlon_properties(_data_property_file_path_ ,
_property_keys_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/decathlon_datalist.py#L136-L158)[#](data.html#monai.data.load_decathlon_properties
"Link to this definition")

    

Load the properties from the JSON file contains data property with specified
property_keys.

Parameters:

    

  * **data_property_file_path** – the path to the JSON file of data properties.

  * **property_keys** – expected keys to load from the JSON file, for example, we have these keys in the decathlon challenge: name, description, reference, licence, tensorImageSize, modality, labels, numTraining, numTest, etc.

monai.data.check_missing_files(_datalist_ , _keys_ , _root_dir =None_,
_allow_missing_keys =False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/decathlon_datalist.py#L161-L196)[#](data.html#monai.data.check_missing_files
"Link to this definition")

    

Checks whether some files in the Decathlon datalist are missing. It would be
helpful to check missing files before a heavy training run.

Parameters:

    

  * **datalist** – a list of data items, every item is a dictionary. usually generated by load_decathlon_datalist API.

  * **keys** – expected keys to check in the datalist.

  * **root_dir** – if not None, provides the root dir for the relative file paths in datalist.

  * **allow_missing_keys** – whether allow missing keys in the datalist items. if False, raise exception if missing. default to False.

Returns:

    

A list of missing filenames.

monai.data.create_cross_validation_datalist(_datalist_ , _nfolds_ ,
_train_folds_ , _val_folds_ , _train_key ='training'_, _val_key
='validation'_, _filename =None_, _shuffle =True_, _seed =0_, _check_missing
=False_, _keys =None_, _root_dir =None_, _allow_missing_keys =False_,
_raise_error =True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/decathlon_datalist.py#L199-L252)[#](data.html#monai.data.create_cross_validation_datalist
"Link to this definition")

    

Utility to create new Decathlon style datalist based on cross validation
partition.

Parameters:

    

  * **datalist** – loaded list of dictionaries for all the items to partition.

  * **nfolds** – number of the kfold split.

  * **train_folds** – indices of folds for training part.

  * **val_folds** – indices of folds for validation part.

  * **train_key** – the key of train part in the new datalist, defaults to “training”.

  * **val_key** – the key of validation part in the new datalist, defaults to “validation”.

  * **filename** – if not None and ends with “.json”, save the new datalist into JSON file.

  * **shuffle** – whether to shuffle the datalist before partition, defaults to True.

  * **seed** – if shuffle is True, set the random seed, defaults to 0.

  * **check_missing** – whether to check all the files specified by keys are existing.

  * **keys** – if not None and check_missing_files is True, the expected keys to check in the datalist.

  * **root_dir** – if not None, provides the root dir for the relative file paths in datalist.

  * **allow_missing_keys** – if check_missing_files is True, whether allow missing keys in the datalist items. if False, raise exception if missing. default to False.

  * **raise_error** – when found missing files, if True, raise exception and stop, if False, print warning.

### DataLoader[#](data.html#dataloader "Link to this heading")

_class _monai.data.DataLoader(_dataset_ , _num_workers =0_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/dataloader.py#L26-L106)[#](data.html#monai.data.DataLoader
"Link to this definition")

    

Provides an iterable over the given dataset. It inherits the PyTorch
DataLoader and adds enhanced collate_fn and worker_fn by default.

Although this class could be configured to be the same as
torch.utils.data.DataLoader, its default configuration is recommended, mainly
for the following extra features:

>   * It handles MONAI randomizable objects with appropriate random state
> managements for deterministic behaviour.
>
>   * It is aware of the patch-based transform (such as
> `monai.transforms.RandSpatialCropSamplesDict`) samples for preprocessing
> with enhanced data collating behaviour. See:
> [`monai.transforms.Compose`](transforms.html#monai.transforms.Compose
> "monai.transforms.Compose").
>
>

For more details about `torch.utils.data.DataLoader`, please see:
<https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>.

For example, to construct a randomized dataset and iterate with the data
loader:

    
    
    import torch
    
    from monai.data import DataLoader
    from monai.transforms import Randomizable
    
    
    class RandomDataset(torch.utils.data.Dataset, Randomizable):
        def __getitem__(self, index):
            return self.R.randint(0, 1000, (1,))
    
        def __len__(self):
            return 16
    
    
    dataset = RandomDataset()
    dataloader = DataLoader(dataset, batch_size=2, num_workers=4)
    for epoch in range(2):
        for i, batch in enumerate(dataloader):
            print(epoch, i, batch.data.numpy().flatten().tolist())
    

Parameters:

    

  * **dataset** (`Dataset`) – dataset from which to load the data.

  * **num_workers** (`int`) – how many subprocesses to use for data loading. `0` means that the data will be loaded in the main process. (default: `0`)

  * **collate_fn** – default to [`monai.data.utils.list_data_collate()`](data.html#monai.data.utils.list_data_collate "monai.data.utils.list_data_collate").

  * **worker_init_fn** – default to [`monai.data.utils.worker_init_fn()`](data.html#monai.data.utils.worker_init_fn "monai.data.utils.worker_init_fn").

  * **kwargs** – other parameters for PyTorch DataLoader.

### ThreadBuffer[#](data.html#threadbuffer "Link to this heading")

_class _monai.data.ThreadBuffer(_src_ , _buffer_size =1_, _timeout
=0.01_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/thread_buffer.py#L23-L80)[#](data.html#monai.data.ThreadBuffer
"Link to this definition")

    

Iterates over values from self.src in a separate thread but yielding them in
the current thread. This allows values to be queued up asynchronously. The
internal thread will continue running so long as the source has values or
until the stop() method is called.

One issue raised by using a thread in this way is that during the lifetime of
the thread the source object is being iterated over, so if the thread hasn’t
finished another attempt to iterate over it will raise an exception or yield
unexpected results. To ensure the thread releases the iteration and proper
cleanup is done the stop() method must be called which will join with the
thread.

Parameters:

    

  * **src** – Source data iterable

  * **buffer_size** (`int`) – Number of items to buffer from the source

  * **timeout** (`float`) – Time to wait for an item from the buffer, or to wait while the buffer is full when adding items

### ThreadDataLoader[#](data.html#threaddataloader "Link to this heading")

_class _monai.data.ThreadDataLoader(_dataset_ , _buffer_size =1_,
_buffer_timeout =0.01_, _repeats =1_, _use_thread_workers =False_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/thread_buffer.py#L137-L199)[#](data.html#monai.data.ThreadDataLoader
"Link to this definition")

    

Subclass of DataLoader using a ThreadBuffer object to implement __iter__
method asynchronously. This will iterate over data from the loader as expected
however the data is generated on a separate thread. Use this class where a
DataLoader instance is required and not just an iterable object.

The default behaviour with repeats set to 1 is to yield each batch as it is
generated, however with a higher value the generated batch is yielded that
many times while underlying dataset asynchronously generates the next.
Typically not all relevant information is learned from a batch in a single
iteration so training multiple times on the same batch will still produce good
training with minimal short-term overfitting while allowing a slow batch
generation process more time to produce a result. This duplication is done by
simply yielding the same object many times and not by regenerating the data.

Another typical usage is to accelerate light-weight preprocessing (usually
cached all the deterministic transforms and no IO operations), because it
leverages the separate thread to execute preprocessing to avoid unnecessary
IPC between multiple workers of DataLoader. And as CUDA may not work well with
the multi-processing of DataLoader, ThreadDataLoader can be useful for GPU
transforms. For more details: [Project-
MONAI/tutorials](https://github.com/Project-
MONAI/tutorials/blob/master/acceleration/fast_model_training_guide.md).

The use_thread_workers will cause workers to be created as threads rather than
processes although everything else in terms of how the class works is
unchanged. This allows multiple workers to be used in Windows for example, or
in any other situation where thread semantics is desired. Please note that
some MONAI components like several datasets and random transforms are not
thread-safe and can’t work as expected with thread workers, need to check all
the preprocessing components carefully before enabling use_thread_workers.

See:

    

  * Fischetti et al. “Faster SGD training by minibatch persistency.” ArXiv (2018) <https://arxiv.org/abs/1806.07353>

  * Dami et al., “Faster Neural Network Training with Data Echoing” ArXiv (2020) <https://arxiv.org/abs/1907.05550>

  * Ramezani et al. “GCN meets GPU: Decoupling “When to Sample” from “How to Sample”.” NeurIPS (2020). <https://proceedings.neurips.cc/paper/2020/file/d714d2c5a796d5814c565d78dd16188d-Paper.pdf>

Parameters:

    

  * **dataset** ([`Dataset`](data.html#monai.data.Dataset "monai.data.dataset.Dataset")) – input dataset.

  * **buffer_size** (`int`) – number of items to buffer from the data source.

  * **buffer_timeout** (`float`) – time to wait for an item from the buffer, or to wait while the buffer is full when adding items.

  * **repeats** (`int`) – number of times to yield the same batch.

  * **use_thread_workers** (`bool`) – if True and num_workers > 0 the workers are created as threads instead of processes

  * **kwargs** – other arguments for DataLoader except for dataset.

### TestTimeAugmentation[#](data.html#testtimeaugmentation "Link to this
heading")

_class _monai.data.TestTimeAugmentation(_transform_ , _batch_size_ ,
_num_workers=0_ , _inferrer_fn= <function _identity>_, _device='cpu'_ ,
_image_key=image_ , _orig_key=label_ , _nearest_interp=True_ ,
_orig_meta_keys=None_ , _meta_key_postfix='meta_dict'_ , _to_tensor=True_ ,
_output_device='cpu'_ , _post_func= <function _identity>_,
_return_full_data=False_ ,
_progress=True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/test_time_augmentation.py#L50-L215)[#](data.html#monai.data.TestTimeAugmentation
"Link to this definition")

    

Class for performing test time augmentations. This will pass the same image
through the network multiple times.

The user passes transform(s) to be applied to each realization, and provided
that at least one of those transforms is random, the network’s output will
vary. Provided that inverse transformations exist for all supplied spatial
transforms, the inverse can be applied to each realization of the network’s
output. Once in the same spatial reference, the results can then be combined
and metrics computed.

Test time augmentations are a useful feature for computing network
uncertainty, as well as observing the network’s dependency on the applied
random transforms.

Reference:

    

Wang et al., Aleatoric uncertainty estimation with test-time augmentation for
medical image segmentation with convolutional neural networks,
<https://doi.org/10.1016/j.neucom.2019.01.103>

Parameters:

    

  * **transform** – transform (or composed) to be applied to each realization. At least one transform must be of type

  * **RandomizableTrait** (i.e. Randomizable, RandomizableTransform, or RandomizableTrait) – . All random transforms must be of type InvertibleTransform.

  * **batch_size** – number of realizations to infer at once.

  * **num_workers** – how many subprocesses to use for data.

  * **inferrer_fn** – function to use to perform inference.

  * **device** – device on which to perform inference.

  * **image_key** – key used to extract image from input dictionary.

  * **orig_key** – the key of the original input data in the dict. will get the applied transform information for this input data, then invert them for the expected data with image_key.

  * **orig_meta_keys** – the key of the metadata of original input data, will get the affine, data_shape, etc. the metadata is a dictionary object which contains: filename, original_shape, etc. if None, will try to construct meta_keys by {orig_key}_{meta_key_postfix}.

  * **meta_key_postfix** – use key_{postfix} to fetch the metadata according to the key data, default is meta_dict, the metadata is a dictionary object. For example, to handle key image, read/write affine matrices from the metadata image_meta_dict dictionary’s affine field. this arg only works when meta_keys=None.

  * **to_tensor** – whether to convert the inverted data into PyTorch Tensor first, default to True.

  * **output_device** – if converted the inverted data to Tensor, move the inverted results to target device before post_func, default to “cpu”.

  * **post_func** – post processing for the inverted data, should be a callable function.

  * **return_full_data** – normally, metrics are returned (mode, mean, std, vvc). Setting this flag to True will return the full data. Dimensions will be same size as when passing a single image through inferrer_fn, with a dimension appended equal in size to num_examples (N), i.e., [N,C,H,W,[D]].

  * **progress** – whether to display a progress bar.

Example

    
    
    model = UNet(...).to(device)
    transform = Compose([RandAffined(keys, ...), ...])
    transform.set_random_state(seed=123)  # ensure deterministic evaluation
    
    tt_aug = TestTimeAugmentation(
        transform, batch_size=5, num_workers=0, inferrer_fn=model, device=device
    )
    mode, mean, std, vvc = tt_aug(test_data)
    

### N-Dim Fourier Transform[#](data.html#module-monai.data.fft_utils "Link to
this heading")

monai.data.fft_utils.fftn_centered(_im_ , _spatial_dims_ , _is_complex
=True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/fft_utils.py#L60-L96)[#](data.html#monai.data.fft_utils.fftn_centered
"Link to this definition")

    

Pytorch-based fft for spatial_dims-dim signals. “centered” means this function
automatically takes care of the required ifft and fft shifts. This function
calls monai.networks.blocks.fft_utils_t.fftn_centered_t. This is equivalent to
do ifft in numpy based on numpy.fft.fftn, numpy.fft.fftshift, and
numpy.fft.ifftshift

Parameters:

    

  * **im** (`Union`[`ndarray`, `Tensor`]) – image that can be 1) real-valued: the shape is (C,H,W) for 2D spatial inputs and (C,H,W,D) for 3D, or 2) complex-valued: the shape is (C,H,W,2) for 2D spatial data and (C,H,W,D,2) for 3D. C is the number of channels.

  * **spatial_dims** (`int`) – number of spatial dimensions (e.g., is 2 for an image, and is 3 for a volume)

  * **is_complex** (`bool`) – if True, then the last dimension of the input im is expected to be 2 (representing real and imaginary channels)

Return type:

    

`Union`[`ndarray`, `Tensor`]

Returns:

    

“out” which is the output kspace (fourier of im)

Example

    
    
    import torch
    im = torch.ones(1,3,3,2) # the last dim belongs to real/imaginary parts
    # output1 and output2 will be identical
    output1 = torch.fft.fftn(torch.view_as_complex(torch.fft.ifftshift(im,dim=(-3,-2))), dim=(-2,-1), norm="ortho")
    output1 = torch.fft.fftshift( torch.view_as_real(output1), dim=(-3,-2) )
    
    output2 = fftn_centered(im, spatial_dims=2, is_complex=True)
    

monai.data.fft_utils.ifftn_centered(_ksp_ , _spatial_dims_ , _is_complex
=True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/fft_utils.py#L21-L57)[#](data.html#monai.data.fft_utils.ifftn_centered
"Link to this definition")

    

Pytorch-based ifft for spatial_dims-dim signals. “centered” means this
function automatically takes care of the required ifft and fft shifts. This
function calls monai.networks.blocks.fft_utils_t.ifftn_centered_t. This is
equivalent to do fft in numpy based on numpy.fft.ifftn, numpy.fft.fftshift,
and numpy.fft.ifftshift

Parameters:

    

  * **ksp** (`Union`[`ndarray`, `Tensor`]) – k-space data that can be 1) real-valued: the shape is (C,H,W) for 2D spatial inputs and (C,H,W,D) for 3D, or 2) complex-valued: the shape is (C,H,W,2) for 2D spatial data and (C,H,W,D,2) for 3D. C is the number of channels.

  * **spatial_dims** (`int`) – number of spatial dimensions (e.g., is 2 for an image, and is 3 for a volume)

  * **is_complex** (`bool`) – if True, then the last dimension of the input ksp is expected to be 2 (representing real and imaginary channels)

Return type:

    

`Union`[`ndarray`, `Tensor`]

Returns:

    

“out” which is the output image (inverse fourier of ksp)

Example

    
    
    import torch
    ksp = torch.ones(1,3,3,2) # the last dim belongs to real/imaginary parts
    # output1 and output2 will be identical
    output1 = torch.fft.ifftn(torch.view_as_complex(torch.fft.ifftshift(ksp,dim=(-3,-2))), dim=(-2,-1), norm="ortho")
    output1 = torch.fft.fftshift( torch.view_as_real(output1), dim=(-3,-2) )
    
    output2 = ifftn_centered(ksp, spatial_dims=2, is_complex=True)
    

### ITK Torch Bridge[#](data.html#module-monai.data.itk_torch_bridge "Link to
this heading")

monai.data.itk_torch_bridge.get_itk_image_center(_image_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/itk_torch_bridge.py#L194-L210)[#](data.html#monai.data.itk_torch_bridge.get_itk_image_center
"Link to this definition")

    

Calculates the center of the ITK image based on its origin, size, and spacing.
This center is equivalent to the implicit image center that MONAI uses.

Parameters:

    

**image** – The ITK image.

Returns:

    

The center of the image as a list of coordinates.

monai.data.itk_torch_bridge.itk_image_to_metatensor(_image_ ,
_channel_dim=None_ , _dtype= <class
'float'>_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/itk_torch_bridge.py#L43-L66)[#](data.html#monai.data.itk_torch_bridge.itk_image_to_metatensor
"Link to this definition")

    

Converts an ITK image to a MetaTensor object.

Parameters:

    

  * **image** – The ITK image to be converted.

  * **channel_dim** – the channel dimension of the input image, default is None. This is used to set original_channel_dim in the metadata, EnsureChannelFirst reads this field. If None, the channel_dim is inferred automatically. If the input array doesn’t have a channel dim, this value should be `'no_channel'`.

  * **dtype** – output dtype, defaults to the Python built-in float.

Returns:

    

A MetaTensor object containing the array data and metadata in ChannelFirst
format.

monai.data.itk_torch_bridge.itk_to_monai_affine(_image_ , _matrix_ ,
_translation_ , _center_of_rotation =None_, _reference_image
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/itk_torch_bridge.py#L105-L154)[#](data.html#monai.data.itk_torch_bridge.itk_to_monai_affine
"Link to this definition")

    

Converts an ITK affine matrix (2x2 for 2D or 3x3 for 3D matrix and translation
vector) to a MONAI affine matrix.

Parameters:

    

  * **image** – The ITK image object. This is used to extract the spacing and direction information.

  * **matrix** – The 2x2 or 3x3 ITK affine matrix.

  * **translation** – The 2-element or 3-element ITK affine translation vector.

  * **center_of_rotation** – The center of rotation. If provided, the affine matrix will be adjusted to account for the difference between the center of the image and the center of rotation.

  * **reference_image** – The coordinate space that matrix and translation were defined in respect to. If not supplied, the coordinate space of image is used.

Return type:

    

`Tensor`

Returns:

    

A 4x4 MONAI affine matrix.

monai.data.itk_torch_bridge.metatensor_to_itk_image(_meta_tensor_ ,
_channel_dim=0_ , _dtype= <class 'numpy.float32'>_,
_**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/itk_torch_bridge.py#L69-L102)[#](data.html#monai.data.itk_torch_bridge.metatensor_to_itk_image
"Link to this definition")

    

Converts a MetaTensor object to an ITK image. Expects the MetaTensor to be in
ChannelFirst format.

Parameters:

    

  * **meta_tensor** – The MetaTensor to be converted.

  * **channel_dim** – channel dimension of the data array, defaults to `0` (Channel-first). `None` indicates no channel dimension. This is used to create a Vector Image if it is not `None`.

  * **dtype** – output data type, defaults to np.float32.

  * **kwargs** – additional keyword arguments. Currently itk.GetImageFromArray will get `ttype` from this dictionary.

Returns:

    

The ITK image.

See also: `ITKWriter.create_backend_obj()`

monai.data.itk_torch_bridge.monai_to_itk_affine(_image_ , _affine_matrix_ ,
_center_of_rotation =None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/itk_torch_bridge.py#L157-L191)[#](data.html#monai.data.itk_torch_bridge.monai_to_itk_affine
"Link to this definition")

    

Converts a MONAI affine matrix to an ITK affine matrix (2x2 for 2D or 3x3 for
3D matrix and translation vector). See also ‘itk_to_monai_affine’.

Parameters:

    

  * **image** – The ITK image object. This is used to extract the spacing and direction information.

  * **affine_matrix** – The 3x3 for 2D or 4x4 for 3D MONAI affine matrix.

  * **center_of_rotation** – The center of rotation. If provided, the affine matrix will be adjusted to account for the difference between the center of the image and the center of rotation.

Returns:

    

The ITK matrix and the translation vector.

monai.data.itk_torch_bridge.monai_to_itk_ddf(_image_ ,
_ddf_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/itk_torch_bridge.py#L310-L345)[#](data.html#monai.data.itk_torch_bridge.monai_to_itk_ddf
"Link to this definition")

    

converting the dense displacement field from the MONAI space to the ITK :param
image: itk image of array shape 2D: (H, W) or 3D: (D, H, W) :param ddf: numpy
array of shape 2D: (2, H, W) or 3D: (3, D, H, W)

Returns:

    

itk image of the corresponding displacement field

Return type:

    

displacement_field

## Meta Object[#](data.html#module-monai.data.meta_obj "Link to this heading")

_class _monai.data.meta_obj.MetaObj[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_obj.py#L63-L244)[#](data.html#monai.data.meta_obj.MetaObj
"Link to this definition")

    

Abstract base class that stores data as well as any extra metadata.

This allows for subclassing torch.Tensor and np.ndarray through multiple
inheritance.

Metadata is stored in the form of a dictionary.

Behavior should be the same as extended class (e.g., torch.Tensor or
np.ndarray) aside from the extended meta functionality.

Copying of information:

>   * For c = a + b, then auxiliary data (e.g., metadata) will be copied from
> the first instance of MetaObj if a.is_batch is False (For batched data, the
> metadata will be shallow copied for efficiency purposes).
>
>

_property _applied_operations _:
list[dict]_[#](data.html#monai.data.meta_obj.MetaObj.applied_operations "Link
to this definition")

    

Get the applied operations. Defaults to `[]`.

Return type:

    

`list`[`dict`]

_static _copy_items(_data_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_obj.py#L107-L116)[#](data.html#monai.data.meta_obj.MetaObj.copy_items
"Link to this definition")

    

returns a copy of the data. list and dict are shallow copied for efficiency
purposes.

copy_meta_from(_input_objs_ , _copy_attr =True_, _keys
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_obj.py#L118-L138)[#](data.html#monai.data.meta_obj.MetaObj.copy_meta_from
"Link to this definition")

    

Copy metadata from a MetaObj or an iterable of MetaObj instances.

Parameters:

    

  * **input_objs** – list of MetaObj to copy data from.

  * **copy_attr** – whether to copy each attribute with MetaObj.copy_item. note that if the attribute is a nested list or dict, only a shallow copy will be done.

  * **keys** – the keys of attributes to copy from the `input_objs`. If None, all keys from the input_objs will be copied.

_static _flatten_meta_objs(_* args_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_obj.py#L88-L105)[#](data.html#monai.data.meta_obj.MetaObj.flatten_meta_objs
"Link to this definition")

    

Recursively flatten input and yield all instances of MetaObj. This means that
for both torch.add(a, b), torch.stack([a, b]) (and their numpy equivalents),
we return [a, b] if both a and b are of type MetaObj.

Parameters:

    

**args** (`Iterable`) – Iterables of inputs to be flattened.

Returns:

    

list of nested MetaObj from input.

_static
_get_default_applied_operations()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_obj.py#L149-L156)[#](data.html#monai.data.meta_obj.MetaObj.get_default_applied_operations
"Link to this definition")

    

Get the default applied operations.

Return type:

    

`list`

Returns:

    

default applied operations.

_static _get_default_meta()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_obj.py#L140-L147)[#](data.html#monai.data.meta_obj.MetaObj.get_default_meta
"Link to this definition")

    

Get the default meta.

Return type:

    

`dict`

Returns:

    

default metadata.

_property _has_pending_operations _:
bool_[#](data.html#monai.data.meta_obj.MetaObj.has_pending_operations "Link to
this definition")

    

Determine whether there are pending operations. :rtype: `bool` :returns: True
if there are pending operations; False if not

_property _is_batch _: bool_[#](data.html#monai.data.meta_obj.MetaObj.is_batch
"Link to this definition")

    

Return whether object is part of batch or not.

Return type:

    

`bool`

_property _meta _: dict_[#](data.html#monai.data.meta_obj.MetaObj.meta "Link
to this definition")

    

Get the meta. Defaults to `{}`.

Return type:

    

`dict`

_property _pending_operations _:
list[dict]_[#](data.html#monai.data.meta_obj.MetaObj.pending_operations "Link
to this definition")

    

Get the pending operations. Defaults to `[]`.

Return type:

    

`list`[`dict`]

monai.data.meta_obj.get_track_meta()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_obj.py#L47-L60)[#](data.html#monai.data.meta_obj.get_track_meta
"Link to this definition")

    

Return the boolean as to whether metadata is tracked. If True, metadata will
be associated its data by using subclasses of MetaObj. If False, then data
will be returned with empty metadata.

If set_track_meta is False, then standard data objects will be returned (e.g.,
torch.Tensor and np.ndarray) as opposed to MONAI’s enhanced objects.

By default, this is True, and most users will want to leave it this way.
However, if you are experiencing any problems regarding metadata, and aren’t
interested in preserving metadata, then you can disable it.

Return type:

    

`bool`

monai.data.meta_obj.set_track_meta(_val_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_obj.py#L30-L44)[#](data.html#monai.data.meta_obj.set_track_meta
"Link to this definition")

    

Boolean to set whether metadata is tracked. If True, metadata will be
associated its data by using subclasses of MetaObj. If False, then data will
be returned with empty metadata.

If set_track_meta is False, then standard data objects will be returned (e.g.,
torch.Tensor and np.ndarray) as opposed to MONAI’s enhanced objects.

By default, this is True, and most users will want to leave it this way.
However, if you are experiencing any problems regarding metadata, and aren’t
interested in preserving metadata, then you can disable it.

Return type:

    

`None`

## MetaTensor[#](data.html#metatensor "Link to this heading")

_class _monai.data.MetaTensor(_x_ , _affine =None_, _meta =None_,
_applied_operations =None_, _* _args_, _**
_kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L52-L609)[#](data.html#monai.data.MetaTensor
"Link to this definition")

    

Bases: [`MetaObj`](data.html#monai.data.meta_obj.MetaObj
"monai.data.meta_obj.MetaObj"), `Tensor`

Class that inherits from both torch.Tensor and MetaObj, adding support for
metadata.

Metadata is stored in the form of a dictionary. Nested, an affine matrix will
be stored. This should be in the form of torch.Tensor.

Behavior should be the same as torch.Tensor aside from the extended meta
functionality.

Copying of information:

>   * For c = a + b, then auxiliary data (e.g., metadata) will be copied from
> the first instance of MetaTensor if a.is_batch is False (For batched data,
> the metadata will be shallow copied for efficiency purposes).
>
>

Example

    
    
    import torch
    from monai.data import MetaTensor
    
    t = torch.tensor([1,2,3])
    affine = torch.as_tensor([[2,0,0,0],
                              [0,2,0,0],
                              [0,0,2,0],
                              [0,0,0,1]], dtype=torch.float64)
    meta = {"some": "info"}
    m = MetaTensor(t, affine=affine, meta=meta)
    m2 = m + m
    assert isinstance(m2, MetaTensor)
    assert m2.meta["some"] == "info"
    assert torch.all(m2.affine == affine)
    

Notes

  * Requires pytorch 1.9 or newer for full compatibility.

  * Older versions of pytorch (<=1.8), torch.jit.trace(net, im) may not work if im is of type MetaTensor. This can be resolved with torch.jit.trace(net, im.as_tensor()).

  * For pytorch < 1.8, sharing MetaTensor instances across processes may not be supported.

  * For pytorch < 1.9, next(iter(meta_tensor)) returns a torch.Tensor. see: [pytorch/pytorch#54457](https://github.com/pytorch/pytorch/issues/54457)

  * A warning will be raised if in the constructor affine is not None and meta already contains the key affine.

  * You can query whether the MetaTensor is a batch with the is_batch attribute.

  * With a batch of data, batch[0] will return the 0th image with the 0th metadata. When the batch dimension is non-singleton, e.g., batch[:, 0], batch[…, -1] and batch[1:3], then all (or a subset in the last example) of the metadata will be returned, and is_batch will return True.

  * When creating a batch with this class, use monai.data.DataLoader as opposed to torch.utils.data.DataLoader, as this will take care of collating the metadata properly.

__init__(_x_ , _affine =None_, _meta =None_, _applied_operations =None_, _*
_args_, _** _kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L119-L172)[#](data.html#monai.data.MetaTensor.__init__
"Link to this definition")

    

Parameters:

    

  * **x** – initial array for the MetaTensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.

  * **affine** – optional 4x4 array.

  * **meta** – dictionary of metadata.

  * **applied_operations** – list of previously applied operations on the MetaTensor, the list is typically maintained by monai.transforms.TraceableTransform. See also: [`monai.transforms.TraceableTransform`](transforms.html#monai.transforms.TraceableTransform "monai.transforms.TraceableTransform")

  * **_args** – additional args (currently not in use in this constructor).

  * **_kwargs** – additional kwargs (currently not in use in this constructor).

Note

If a meta dictionary is given, use it. Else, if meta exists in the input
tensor x, use it. Else, use the default value. Similar for the affine, except
this could come from four places, priority: affine, meta[“affine”], x.affine,
get_default_affine.

_property _affine _: Tensor_[#](data.html#monai.data.MetaTensor.affine "Link
to this definition")

    

Get the affine. Defaults to `torch.eye(4, dtype=torch.float64)`

Return type:

    

`Tensor`

_property _array[#](data.html#monai.data.MetaTensor.array "Link to this
definition")

    

Returns a numpy array of `self`. The array and `self` shares the same
underlying storage if self is on cpu. Changes to `self` (it’s a subclass of
torch.Tensor) will be reflected in the ndarray and vice versa. If `self` is
not on cpu, the call will move the array to cpu and then the storage is not
shared.

Getter:

    

see also: [`MetaTensor.get_array()`](data.html#monai.data.MetaTensor.get_array
"monai.data.MetaTensor.get_array")

Setter:

    

see also: [`MetaTensor.set_array()`](data.html#monai.data.MetaTensor.set_array
"monai.data.MetaTensor.set_array")

as_dict(_key_ , _output_type= <class 'torch.Tensor'>_,
_dtype=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L411-L432)[#](data.html#monai.data.MetaTensor.as_dict
"Link to this definition")

    

Get the object as a dictionary for backwards compatibility. This method does
not make a deep copy of the objects.

Parameters:

    

  * **key** (`str`) – Base key to store main data. The key for the metadata will be determined using PostFix.

  * **output_type** – torch.Tensor or np.ndarray for the main data.

  * **dtype** – dtype of output data. Converted to correct library type (e.g., np.float32 is converted to torch.float32 if output type is torch.Tensor). If left blank, it remains unchanged.

Return type:

    

`dict`

Returns:

    

A dictionary consisting of three keys, the main data (stored under key) and
the metadata.

as_tensor()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L349-L354)[#](data.html#monai.data.MetaTensor.as_tensor
"Link to this definition")

    

Return the MetaTensor as a torch.Tensor. It is OS dependent as to whether this
will be a deep copy or not.

Return type:

    

`Tensor`

astype(_dtype_ , _device =None_, _* _args_, _**
_kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L434-L461)[#](data.html#monai.data.MetaTensor.astype
"Link to this definition")

    

Cast to `dtype`, sharing data whenever possible.

Parameters:

    

  * **dtype** – dtypes such as np.float32, torch.float, “np.float32”, float.

  * **device** – the device if dtype is a torch data type.

  * **_args** – additional args (currently unused).

  * **_kwargs** – additional kwargs (currently unused).

Returns:

    

data array instance

clone(_** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L520-L531)[#](data.html#monai.data.MetaTensor.clone
"Link to this definition")

    

Returns a copy of the MetaTensor instance.

Parameters:

    

**kwargs** – additional keyword arguments to torch.clone.

See also: <https://pytorch.org/docs/stable/generated/torch.clone.html>

_static _ensure_torch_and_prune_meta(_im_ , _meta_ , _simple_keys =False_,
_pattern =None_, _sep ='.'_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L533-L581)[#](data.html#monai.data.MetaTensor.ensure_torch_and_prune_meta
"Link to this definition")

    

Convert the image to MetaTensor (when meta is not None). If affine is in the
meta dictionary, convert that to torch.Tensor, too. Remove any superfluous
metadata.

Parameters:

    

  * **im** – Input image (np.ndarray or torch.Tensor)

  * **meta** – Metadata dictionary. When it’s None, the metadata is not tracked, this method returns a torch.Tensor.

  * **simple_keys** – whether to keep only a simple subset of metadata keys.

  * **pattern** – combined with sep, a regular expression used to match and prune keys in the metadata (nested dictionary), default to None, no key deletion.

  * **sep** – combined with pattern, used to match and delete keys in the metadata (nested dictionary). default is “.”, see also [`monai.transforms.DeleteItemsd`](transforms.html#monai.transforms.DeleteItemsd "monai.transforms.DeleteItemsd"). e.g. `pattern=".*_code$", sep=" "` removes any meta keys that ends with `"_code"`.

Returns:

    

By default, a MetaTensor is returned. However, if get_track_meta() is False or
meta=None, a torch.Tensor is returned.

get_array(_output_type= <class 'numpy.ndarray'>_, _dtype=None_ , _device=None_
, _*_args_ , _**_kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L356-L370)[#](data.html#monai.data.MetaTensor.get_array
"Link to this definition")

    

Returns a new array in output_type, the array shares the same underlying
storage when the output is a numpy array. Changes to self tensor will be
reflected in the ndarray and vice versa.

Parameters:

    

  * **output_type** – output type, see also: `monai.utils.convert_data_type()`.

  * **dtype** – dtype of output data. Converted to correct library type (e.g., np.float32 is converted to torch.float32 if output type is torch.Tensor). If left blank, it remains unchanged.

  * **device** – if the output is a torch.Tensor, select device (if None, unchanged).

  * **_args** – currently unused parameters.

  * **_kwargs** – currently unused parameters.

new_empty(_size_ , _dtype =None_, _device =None_, _requires_grad
=False_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L509-L518)[#](data.html#monai.data.MetaTensor.new_empty
"Link to this definition")

    

must be defined for deepcopy to work

See:

    

  * <https://pytorch.org/docs/stable/generated/torch.Tensor.new_empty.html#torch-tensor-new-empty>

peek_pending_shape()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L480-L489)[#](data.html#monai.data.MetaTensor.peek_pending_shape
"Link to this definition")

    

Get the currently expected spatial shape as if all the pending operations are
executed. For tensors that have more than 3 spatial dimensions, only the
shapes of the top 3 dimensions will be returned.

_property _pixdim[#](data.html#monai.data.MetaTensor.pixdim "Link to this
definition")

    

Get the spacing

print_verbose()[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L605-L609)[#](data.html#monai.data.MetaTensor.print_verbose
"Link to this definition")

    

Verbose print with meta data.

Return type:

    

`None`

set_array(_src_ , _non_blocking =False_, _* _args_, _**
_kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L372-L392)[#](data.html#monai.data.MetaTensor.set_array
"Link to this definition")

    

Copies the elements from src into self tensor and returns self. The src tensor
must be broadcastable with the self tensor. It may be of a different data type
or reside on a different device.

See also: https://pytorch.org/docs/stable/generated/torch.Tensor.copy_.html

Parameters:

    

  * **src** – the source tensor to copy from.

  * **non_blocking** (`bool`) – if True and this copy is between CPU and GPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect.

  * **_args** – currently unused parameters.

  * **_kwargs** – currently unused parameters.

_static _update_meta(_rets_ , _func_ , _args_ ,
_kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/meta_tensor.py#L174-L227)[#](data.html#monai.data.MetaTensor.update_meta
"Link to this definition")

    

Update the metadata from the output of MetaTensor.__torch_function__.

The output of torch.Tensor.__torch_function__ could be a single object or a
sequence of them. Hence, in MetaTensor.__torch_function__ we convert them to a
list of not already, and then we loop across each element, processing metadata
as necessary. For each element, if not of type MetaTensor, then nothing to do.

Parameters:

    

  * **rets** (`Sequence`) – the output from torch.Tensor.__torch_function__, which has been converted to a list in MetaTensor.__torch_function__ if it wasn’t already a Sequence.

  * **func** – the torch function that was applied. Examples might be torch.squeeze or torch.Tensor.__add__. We need this since the metadata need to be treated differently if a batch of data is considered. For example, slicing (torch.Tensor.__getitem__) the ith element of the 0th dimension of a batch of data should return a ith tensor with the ith metadata.

  * **args** – positional arguments that were passed to func.

  * **kwargs** – keyword arguments that were passed to func.

Return type:

    

`Sequence`

Returns:

    

A sequence with the same number of elements as rets. For each element, if the
input type was not MetaTensor, then no modifications will have been made. If
global parameters have been set to false (e.g., not get_track_meta()), then
any MetaTensor will be converted to torch.Tensor. Else, metadata will be
propagated as necessary (see `MetaTensor._copy_meta()`).

## Whole slide image reader[#](data.html#whole-slide-image-reader "Link to
this heading")

### BaseWSIReader[#](data.html#basewsireader "Link to this heading")

_class _monai.data.BaseWSIReader(_level=None_ , _mpp=None_ , _mpp_rtol=0.05_ ,
_mpp_atol=0.0_ , _power=None_ , _power_rtol=0.05_ , _power_atol=0.0_ ,
_channel_dim=0_ , _dtype= <class 'numpy.uint8'>_, _device=None_ , _mode='RGB'_
, _**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L43-L444)[#](data.html#monai.data.BaseWSIReader
"Link to this definition")

    

An abstract class that defines APIs to load patches from whole slide image
files.

Parameters:

    

  * **level** – the whole slide image level at which the patches are extracted.

  * **mpp** – the resolution in micron per pixel at which the patches are extracted.

  * **mpp_rtol** – the acceptable relative tolerance for resolution in micro per pixel.

  * **mpp_atol** – the acceptable absolute tolerance for resolution in micro per pixel.

  * **power** – the objective power at which the patches are extracted.

  * **power_rtol** – the acceptable relative tolerance for objective power.

  * **power_atol** – the acceptable absolute tolerance for objective power.

  * **channel_dim** – the desired dimension for color channel.

  * **dtype** – the data type of output image.

  * **device** – target device to put the extracted patch. Note that if device is “cuda””, the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.

  * **mode** – the output image color mode, e.g., “RGB” or “RGBA”.

  * **kwargs** – additional args for the reader

  * **Notes** – Only one of resolution parameters, level, mpp, or power, should be provided. If such parameters are provided in get_data method, those will override the values provided here. If none of them are provided here or in get_data, level=0 will be used.

Typical usage of a concrete implementation of this class is:

    
    
    image_reader = MyWSIReader()
    wsi = image_reader.read(filepath, **kwargs)
    img_data, meta_data = image_reader.get_data(wsi)
    

  * The read call converts an image filename into whole slide image object,

  * The get_data call fetches the image data, as well as metadata.

The following methods needs to be implemented for any concrete implementation
of this class:

  * read reads a whole slide image object from a given file

  * get_size returns the size of the whole slide image of a given wsi object at a given level.

  * get_level_count returns the number of levels in the whole slide image

  * _get_patch extracts and returns a patch image form the whole slide image

  * _get_metadata extracts and returns metadata for a whole slide image and a specific patch.

get_data(_wsi_ , _location =(0, 0)_, _size =None_, _level =None_, _mpp =None_,
_power =None_, _mode =None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L320-L432)[#](data.html#monai.data.BaseWSIReader.get_data
"Link to this definition")

    

Verifies inputs, extracts patches from WSI image and generates metadata.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file or a list of such objects.

  * **location** – (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).

  * **size** – (height, width) tuple giving the patch size at the given level (level). If not provided or None, it is set to the full image size at the given level.

  * **level** – the whole slide image level at which the patches are extracted.

  * **mpp** – the resolution in micron per pixel at which the patches are extracted.

  * **power** – the objective power at which the patches are extracted.

  * **dtype** – the data type of output image.

  * **mode** – the output image mode, ‘RGB’ or ‘RGBA’.

Returns:

    

a tuples, where the first element is an image patch [CxHxW] or stack of
patches,

    

and second element is a dictionary of metadata.

Notes

Only one of resolution parameters, level, mpp, or power, should be provided.
If none of them are provided, it uses the defaults that are set during class
instantiation. If none of them are set here or during class instantiation,
level=0 will be used.

_abstract _get_downsample_ratio(_wsi_ ,
_level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L228-L238)[#](data.html#monai.data.BaseWSIReader.get_downsample_ratio
"Link to this definition")

    

Returns the down-sampling ratio of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the downsample ratio is calculated.

Return type:

    

`float`

_abstract _get_file_path(_wsi_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L240-L243)[#](data.html#monai.data.BaseWSIReader.get_file_path
"Link to this definition")

    

Return the file path for the WSI object

Return type:

    

`str`

_abstract _get_level_count(_wsi_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L217-L226)[#](data.html#monai.data.BaseWSIReader.get_level_count
"Link to this definition")

    

Returns the number of levels in the whole slide image.

Parameters:

    

**wsi** – a whole slide image object loaded from a file.

Return type:

    

`int`

_abstract _get_mpp(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L245-L255)[#](data.html#monai.data.BaseWSIReader.get_mpp
"Link to this definition")

    

Returns the micro-per-pixel resolution of the whole slide image at a given
level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the mpp is calculated.

Return type:

    

`tuple`[`float`, `float`]

_abstract _get_power(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L257-L267)[#](data.html#monai.data.BaseWSIReader.get_power
"Link to this definition")

    

Returns the objective power of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the objective power is calculated.

Return type:

    

`float`

_abstract _get_size(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L135-L145)[#](data.html#monai.data.BaseWSIReader.get_size
"Link to this definition")

    

Returns the size (height, width) of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the size is calculated.

Return type:

    

`tuple`[`int`, `int`]

get_valid_level(_wsi_ , _level_ , _mpp_ ,
_power_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L173-L215)[#](data.html#monai.data.BaseWSIReader.get_valid_level
"Link to this definition")

    

Returns the level associated to the resolution parameters in the whole slide
image.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** – the level number.

  * **mpp** – the micron-per-pixel resolution.

  * **power** – the objective power.

verify_suffix(_filename_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L434-L444)[#](data.html#monai.data.BaseWSIReader.verify_suffix
"Link to this definition")

    

Verify whether the specified file or files format is supported by WSI reader.

The list of supported suffixes are read from self.supported_suffixes.

Parameters:

    

**filename** – filename or a list of filenames to read.

### WSIReader[#](data.html#wsireader "Link to this heading")

_class _monai.data.WSIReader(_backend='cucim'_ , _level=None_ , _mpp=None_ ,
_mpp_rtol=0.05_ , _mpp_atol=0.0_ , _power=None_ , _power_rtol=0.05_ ,
_power_atol=0.0_ , _channel_dim=0_ , _dtype= <class 'numpy.uint8'>_,
_device=None_ , _mode='RGB'_ ,
_**kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L447-L647)[#](data.html#monai.data.WSIReader
"Link to this definition")

    

Read whole slide images and extract patches using different backend libraries

Parameters:

    

  * **backend** – the name of backend whole slide image reader library, the default is cuCIM.

  * **level** – the whole slide image level at which the patches are extracted.

  * **mpp** – the resolution in micron per pixel at which the patches are extracted.

  * **mpp_rtol** – the acceptable relative tolerance for resolution in micro per pixel.

  * **mpp_atol** – the acceptable absolute tolerance for resolution in micro per pixel.

  * **power** – the objective power at which the patches are extracted.

  * **power_rtol** – the acceptable relative tolerance for objective power.

  * **power_atol** – the acceptable absolute tolerance for objective power.

  * **channel_dim** – the desired dimension for color channel. Default to 0 (channel first).

  * **dtype** – the data type of output image. Defaults to np.uint8.

  * **device** – target device to put the extracted patch. Note that if device is “cuda””, the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.

  * **mode** – the output image color mode, “RGB” or “RGBA”. Defaults to “RGB”.

  * **num_workers** – number of workers for multi-thread image loading (cucim backend only).

  * **kwargs** – additional arguments to be passed to the backend library

  * **Notes** – Only one of resolution parameters, level, mpp, or power, should be provided. If such parameters are provided in get_data method, those will override the values provided here. If none of them are provided here or in get_data, level=0 will be used.

get_downsample_ratio(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L580-L589)[#](data.html#monai.data.WSIReader.get_downsample_ratio
"Link to this definition")

    

Returns the down-sampling ratio of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the downsample ratio is calculated.

Return type:

    

`float`

get_file_path(_wsi_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L591-L593)[#](data.html#monai.data.WSIReader.get_file_path
"Link to this definition")

    

Return the file path for the WSI object

Return type:

    

`str`

get_level_count(_wsi_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L559-L567)[#](data.html#monai.data.WSIReader.get_level_count
"Link to this definition")

    

Returns the number of levels in the whole slide image.

Parameters:

    

**wsi** – a whole slide image object loaded from a file.

Return type:

    

`int`

get_mpp(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L595-L604)[#](data.html#monai.data.WSIReader.get_mpp
"Link to this definition")

    

Returns the micro-per-pixel resolution of the whole slide image at a given
level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the mpp is calculated.

Return type:

    

`tuple`[`float`, `float`]

get_power(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L606-L615)[#](data.html#monai.data.WSIReader.get_power
"Link to this definition")

    

Returns the micro-per-pixel resolution of the whole slide image at a given
level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the objective power is calculated.

Return type:

    

`float`

get_size(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L569-L578)[#](data.html#monai.data.WSIReader.get_size
"Link to this definition")

    

Returns the size (height, width) of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the size is calculated.

Return type:

    

`tuple`[`int`, `int`]

read(_data_ , _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L635-L647)[#](data.html#monai.data.WSIReader.read
"Link to this definition")

    

Read whole slide image objects from given file or list of files.

Parameters:

    

  * **data** – file name or a list of file names to read.

  * **kwargs** – additional args for the reader module (overrides self.kwargs for existing keys).

Returns:

    

whole slide image object or list of such objects.

### CuCIMWSIReader[#](data.html#cucimwsireader "Link to this heading")

_class _monai.data.CuCIMWSIReader(_num_workers =0_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L650-L829)[#](data.html#monai.data.CuCIMWSIReader
"Link to this definition")

    

Read whole slide images and extract patches using cuCIM library.

Parameters:

    

  * **level** – the whole slide image level at which the patches are extracted.

  * **mpp** – the resolution in micron per pixel at which the patches are extracted.

  * **mpp_rtol** – the acceptable relative tolerance for resolution in micro per pixel.

  * **mpp_atol** – the acceptable absolute tolerance for resolution in micro per pixel.

  * **power** – the objective power at which the patches are extracted.

  * **power_rtol** – the acceptable relative tolerance for objective power.

  * **power_atol** – the acceptable absolute tolerance for objective power.

  * **channel_dim** – the desired dimension for color channel. Default to 0 (channel first).

  * **dtype** – the data type of output image. Defaults to np.uint8.

  * **device** – target device to put the extracted patch. Note that if device is “cuda””, the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.

  * **mode** – the output image color mode, “RGB” or “RGBA”. Defaults to “RGB”.

  * **num_workers** (`int`) – number of workers for multi-thread image loading.

  * **kwargs** – additional args for cucim.CuImage module: [rapidsai/cucim](https://github.com/rapidsai/cucim/blob/main/cpp/include/cucim/cuimage.h)

  * **Notes** – Only one of resolution parameters, level, mpp, or power, should be provided. If such parameters are provided in get_data method, those will override the values provided here. If none of them are provided here or in get_data, level=0 will be used.

get_downsample_ratio(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L708-L717)[#](data.html#monai.data.CuCIMWSIReader.get_downsample_ratio
"Link to this definition")

    

Returns the down-sampling ratio of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the downsample ratio is calculated.

Return type:

    

`float`

_static _get_file_path(_wsi_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L719-L722)[#](data.html#monai.data.CuCIMWSIReader.get_file_path
"Link to this definition")

    

Return the file path for the WSI object

Return type:

    

`str`

_static _get_level_count(_wsi_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L686-L695)[#](data.html#monai.data.CuCIMWSIReader.get_level_count
"Link to this definition")

    

Returns the number of levels in the whole slide image.

Parameters:

    

**wsi** – a whole slide image object loaded from a file.

Return type:

    

`int`

get_mpp(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L724-L745)[#](data.html#monai.data.CuCIMWSIReader.get_mpp
"Link to this definition")

    

Returns the micro-per-pixel resolution of the whole slide image at a given
level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the mpp is calculated.

Return type:

    

`tuple`[`float`, `float`]

get_power(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L747-L765)[#](data.html#monai.data.CuCIMWSIReader.get_power
"Link to this definition")

    

Returns the objective power of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the objective power is calculated.

Return type:

    

`float`

get_size(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L697-L706)[#](data.html#monai.data.CuCIMWSIReader.get_size
"Link to this definition")

    

Returns the size (height, width) of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the size is calculated.

Return type:

    

`tuple`[`int`, `int`]

read(_data_ , _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L767-L790)[#](data.html#monai.data.CuCIMWSIReader.read
"Link to this definition")

    

Read whole slide image objects from given file or list of files.

Parameters:

    

  * **data** – file name or a list of file names to read.

  * **kwargs** – additional args that overrides self.kwargs for existing keys. For more details look at [rapidsai/cucim](https://github.com/rapidsai/cucim/blob/main/cpp/include/cucim/cuimage.h)

Returns:

    

whole slide image object or list of such objects.

### OpenSlideWSIReader[#](data.html#openslidewsireader "Link to this heading")

_class _monai.data.OpenSlideWSIReader(_**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L832-L1011)[#](data.html#monai.data.OpenSlideWSIReader
"Link to this definition")

    

Read whole slide images and extract patches using OpenSlide library.

Parameters:

    

  * **level** – the whole slide image level at which the patches are extracted.

  * **mpp** – the resolution in micron per pixel at which the patches are extracted.

  * **mpp_rtol** – the acceptable relative tolerance for resolution in micro per pixel.

  * **mpp_atol** – the acceptable absolute tolerance for resolution in micro per pixel.

  * **power** – the objective power at which the patches are extracted.

  * **power_rtol** – the acceptable relative tolerance for objective power.

  * **power_atol** – the acceptable absolute tolerance for objective power.

  * **channel_dim** – the desired dimension for color channel. Default to 0 (channel first).

  * **dtype** – the data type of output image. Defaults to np.uint8.

  * **device** – target device to put the extracted patch. Note that if device is “cuda””, the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.

  * **mode** – the output image color mode, “RGB” or “RGBA”. Defaults to “RGB”.

  * **kwargs** – additional args for openslide.OpenSlide module.

  * **Notes** – Only one of resolution parameters, level, mpp, or power, should be provided. If such parameters are provided in get_data method, those will override the values provided here. If none of them are provided here or in get_data, level=0 will be used.

get_downsample_ratio(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L887-L896)[#](data.html#monai.data.OpenSlideWSIReader.get_downsample_ratio
"Link to this definition")

    

Returns the down-sampling ratio of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the downsample ratio is calculated.

Return type:

    

`float`

_static _get_file_path(_wsi_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L898-L901)[#](data.html#monai.data.OpenSlideWSIReader.get_file_path
"Link to this definition")

    

Return the file path for the WSI object

Return type:

    

`str`

_static _get_level_count(_wsi_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L865-L874)[#](data.html#monai.data.OpenSlideWSIReader.get_level_count
"Link to this definition")

    

Returns the number of levels in the whole slide image.

Parameters:

    

**wsi** – a whole slide image object loaded from a file.

Return type:

    

`int`

get_mpp(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L903-L941)[#](data.html#monai.data.OpenSlideWSIReader.get_mpp
"Link to this definition")

    

Returns the micro-per-pixel resolution of the whole slide image at a given
level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the mpp is calculated.

Return type:

    

`tuple`[`float`, `float`]

get_power(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L943-L957)[#](data.html#monai.data.OpenSlideWSIReader.get_power
"Link to this definition")

    

Returns the objective power of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the objective power is calculated.

Return type:

    

`float`

get_size(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L876-L885)[#](data.html#monai.data.OpenSlideWSIReader.get_size
"Link to this definition")

    

Returns the size (height, width) of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the size is calculated.

Return type:

    

`tuple`[`int`, `int`]

read(_data_ , _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L959-L980)[#](data.html#monai.data.OpenSlideWSIReader.read
"Link to this definition")

    

Read whole slide image objects from given file or list of files.

Parameters:

    

  * **data** – file name or a list of file names to read.

  * **kwargs** – additional args that overrides self.kwargs for existing keys.

Returns:

    

whole slide image object or list of such objects.

### TiffFileWSIReader[#](data.html#tifffilewsireader "Link to this heading")

_class _monai.data.TiffFileWSIReader(_**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L1014-L1188)[#](data.html#monai.data.TiffFileWSIReader
"Link to this definition")

    

Read whole slide images and extract patches using TiffFile library.

Parameters:

    

  * **level** – the whole slide image level at which the patches are extracted.

  * **mpp** – the resolution in micron per pixel at which the patches are extracted.

  * **mpp_rtol** – the acceptable relative tolerance for resolution in micro per pixel.

  * **mpp_atol** – the acceptable absolute tolerance for resolution in micro per pixel.

  * **channel_dim** – the desired dimension for color channel. Default to 0 (channel first).

  * **dtype** – the data type of output image. Defaults to np.uint8.

  * **device** – target device to put the extracted patch. Note that if device is “cuda””, the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.

  * **mode** – the output image color mode, “RGB” or “RGBA”. Defaults to “RGB”.

  * **kwargs** – additional args for tifffile.TiffFile module.

  * **Notes** – 
    * Objective power cannot be obtained via TiffFile backend.

    * Only one of resolution parameters, level or mpp, should be provided.
    

If such parameters are provided in get_data method, those will override the
values provided here. If none of them are provided here or in get_data,
level=0 will be used.

get_downsample_ratio(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L1067-L1076)[#](data.html#monai.data.TiffFileWSIReader.get_downsample_ratio
"Link to this definition")

    

Returns the down-sampling ratio of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the downsample ratio is calculated.

Return type:

    

`float`

_static _get_file_path(_wsi_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L1078-L1081)[#](data.html#monai.data.TiffFileWSIReader.get_file_path
"Link to this definition")

    

Return the file path for the WSI object

Return type:

    

`str`

_static _get_level_count(_wsi_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L1045-L1054)[#](data.html#monai.data.TiffFileWSIReader.get_level_count
"Link to this definition")

    

Returns the number of levels in the whole slide image.

Parameters:

    

**wsi** – a whole slide image object loaded from a file.

Return type:

    

`int`

get_mpp(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L1083-L1111)[#](data.html#monai.data.TiffFileWSIReader.get_mpp
"Link to this definition")

    

Returns the micro-per-pixel resolution of the whole slide image at a given
level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the mpp is calculated.

Return type:

    

`tuple`[`float`, `float`]

get_power(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L1113-L1125)[#](data.html#monai.data.TiffFileWSIReader.get_power
"Link to this definition")

    

Returns the objective power of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the objective power is calculated.

Return type:

    

`float`

get_size(_wsi_ , _level_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L1056-L1065)[#](data.html#monai.data.TiffFileWSIReader.get_size
"Link to this definition")

    

Returns the size (height, width) of the whole slide image at a given level.

Parameters:

    

  * **wsi** – a whole slide image object loaded from a file.

  * **level** (`int`) – the level number where the size is calculated.

Return type:

    

`tuple`[`int`, `int`]

read(_data_ , _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_reader.py#L1127-L1148)[#](data.html#monai.data.TiffFileWSIReader.read
"Link to this definition")

    

Read whole slide image objects from given file or list of files.

Parameters:

    

  * **data** – file name or a list of file names to read.

  * **kwargs** – additional args that overrides self.kwargs for existing keys.

Returns:

    

whole slide image object or list of such objects.

## Whole slide image datasets[#](data.html#whole-slide-image-datasets "Link to
this heading")

### PatchWSIDataset[#](data.html#patchwsidataset "Link to this heading")

_class _monai.data.PatchWSIDataset(_data_ , _patch_size =None_, _patch_level
=None_, _transform =None_, _include_label =True_, _center_location =True_,
_additional_meta_keys =None_, _reader ='cuCIM'_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_datasets.py#L32-L169)[#](data.html#monai.data.PatchWSIDataset
"Link to this definition")

    

This dataset extracts patches from whole slide images (without loading the
whole image) It also reads labels for each patch and provides each patch with
its associated class labels.

Parameters:

    

  * **data** – the list of input samples including image, location, and label (see the note below for more details).

  * **patch_size** – the size of patch to be extracted from the whole slide image.

  * **patch_level** – the level at which the patches to be extracted (default to 0).

  * **transform** – transforms to be executed on input data.

  * **include_label** – whether to load and include labels in the output

  * **center_location** – whether the input location information is the position of the center of the patch

  * **additional_meta_keys** – the list of keys for items to be copied to the output metadata from the input data

  * **reader** – 

the module to be used for loading whole slide imaging. If reader is

    * a string, it defines the backend of monai.data.WSIReader. Defaults to cuCIM.

    * a class (inherited from BaseWSIReader), it is initialized and set as wsi_reader.

    * an instance of a class inherited from BaseWSIReader, it is set as the wsi_reader.

  * **kwargs** – additional arguments to pass to WSIReader or provided whole slide reader class

Returns:

    

a dictionary of loaded image (in MetaTensor format) along with the labels (if
requested). {“image”: MetaTensor, “label”: torch.Tensor}

Return type:

    

dict

Note

The input data has the following form as an example:

    
    
    [
        {"image": "path/to/image1.tiff", "location": [200, 500], "label": 0},
        {"image": "path/to/image2.tiff", "location": [100, 700], "patch_size": [20, 20], "patch_level": 2, "label": 1}
    ]
    

### MaskedPatchWSIDataset[#](data.html#maskedpatchwsidataset "Link to this
heading")

_class _monai.data.MaskedPatchWSIDataset(_data_ , _patch_size =None_,
_patch_level =None_, _mask_level =7_, _transform =None_, _include_label
=False_, _center_location =False_, _additional_meta_keys =(mask_location,
name)_, _reader ='cuCIM'_, _** kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_datasets.py#L321-L416)[#](data.html#monai.data.MaskedPatchWSIDataset
"Link to this definition")

    

This dataset extracts patches from whole slide images at the locations where
foreground mask at a given level is non-zero.

Parameters:

    

  * **data** – the list of input samples including image, location, and label (see the note below for more details).

  * **patch_size** – the size of patch to be extracted from the whole slide image.

  * **patch_level** – the level at which the patches to be extracted (default to 0).

  * **mask_level** – the resolution level at which the mask is created.

  * **transform** – transforms to be executed on input data.

  * **include_label** – whether to load and include labels in the output

  * **center_location** – whether the input location information is the position of the center of the patch

  * **additional_meta_keys** – the list of keys for items to be copied to the output metadata from the input data

  * **reader** – 

the module to be used for loading whole slide imaging. Defaults to cuCIM. If
reader is

    * a string, it defines the backend of monai.data.WSIReader.

    * a class (inherited from BaseWSIReader), it is initialized and set as wsi_reader,

    * an instance of a class inherited from BaseWSIReader, it is set as the wsi_reader.

  * **kwargs** – additional arguments to pass to WSIReader or provided whole slide reader class

Note

The input data has the following form as an example:

    
    
    [
        {"image": "path/to/image1.tiff"},
        {"image": "path/to/image2.tiff", "size": [20, 20], "level": 2}
    ]
    

### SlidingPatchWSIDataset[#](data.html#slidingpatchwsidataset "Link to this
heading")

_class _monai.data.SlidingPatchWSIDataset(_data_ , _patch_size =None_,
_patch_level =None_, _mask_level =0_, _overlap =0.0_, _offset =(0, 0)_,
_offset_limits =None_, _transform =None_, _include_label =False_,
_center_location =False_, _additional_meta_keys =(mask_location, mask_size,
num_patches)_, _reader ='cuCIM'_, _seed =0_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/wsi_datasets.py#L172-L318)[#](data.html#monai.data.SlidingPatchWSIDataset
"Link to this definition")

    

This dataset extracts patches in sliding-window manner from whole slide images
(without loading the whole image). It also reads labels for each patch and
provides each patch with its associated class labels.

Parameters:

    

  * **data** – the list of input samples including image, location, and label (see the note below for more details).

  * **patch_size** – the size of patch to be extracted from the whole slide image.

  * **patch_level** – the level at which the patches to be extracted (default to 0).

  * **mask_level** – the resolution level at which the mask/map is created (for ProbMapProducer for instance).

  * **overlap** – the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0). If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.

  * **offset** – the offset of image to extract patches (the starting position of the upper left patch).

  * **offset_limits** – if offset is set to “random”, a tuple of integers defining the lower and upper limit of the random offset for all dimensions, or a tuple of tuples that defines the limits for each dimension.

  * **transform** – transforms to be executed on input data.

  * **include_label** – whether to load and include labels in the output

  * **center_location** – whether the input location information is the position of the center of the patch

  * **additional_meta_keys** – the list of keys for items to be copied to the output metadata from the input data

  * **reader** – 

the module to be used for loading whole slide imaging. Defaults to cuCIM. If
reader is

    * a string, it defines the backend of monai.data.WSIReader.

    * a class (inherited from BaseWSIReader), it is initialized and set as wsi_reader,

    * an instance of a class inherited from BaseWSIReader, it is set as the wsi_reader.

  * **seed** – random seed to randomly generate offsets. Defaults to 0.

  * **kwargs** – additional arguments to pass to WSIReader or provided whole slide reader class

Note

The input data has the following form as an example:

    
    
    [
        {"image": "path/to/image1.tiff"},
        {"image": "path/to/image2.tiff", "patch_size": [20, 20], "patch_level": 2}
    ]
    

Unlike MaskedPatchWSIDataset, this dataset does not filter any patches.

## Bounding box[#](data.html#module-monai.data.box_utils "Link to this
heading")

This utility module mainly supports rectangular bounding boxes with a few
different parameterizations and methods for converting between them. It
provides reliable access to the spatial coordinates of the box vertices in the
“canonical ordering”: [xmin, ymin, xmax, ymax] for 2D and [xmin, ymin, zmin,
xmax, ymax, zmax] for 3D. We currently define this ordering as
monai.data.box_utils.StandardMode and the rest of the detection pipelines
mainly assumes boxes in StandardMode.

_class _monai.data.box_utils.BoxMode[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L52-L137)[#](data.html#monai.data.box_utils.BoxMode
"Link to this definition")

    

An abstract class of a `BoxMode`.

A `BoxMode` is callable that converts box mode of `boxes`, which are Nx4 (2D)
or Nx6 (3D) torch tensor or ndarray. `BoxMode` has several subclasses that
represents different box modes, including

  * [`CornerCornerModeTypeA`](data.html#monai.data.box_utils.CornerCornerModeTypeA "monai.data.box_utils.CornerCornerModeTypeA"): represents [xmin, ymin, xmax, ymax] for 2D and [xmin, ymin, zmin, xmax, ymax, zmax] for 3D

  * [`CornerCornerModeTypeB`](data.html#monai.data.box_utils.CornerCornerModeTypeB "monai.data.box_utils.CornerCornerModeTypeB"): represents [xmin, xmax, ymin, ymax] for 2D and [xmin, xmax, ymin, ymax, zmin, zmax] for 3D

  * [`CornerCornerModeTypeC`](data.html#monai.data.box_utils.CornerCornerModeTypeC "monai.data.box_utils.CornerCornerModeTypeC"): represents [xmin, ymin, xmax, ymax] for 2D and [xmin, ymin, xmax, ymax, zmin, zmax] for 3D

  * [`CornerSizeMode`](data.html#monai.data.box_utils.CornerSizeMode "monai.data.box_utils.CornerSizeMode"): represents [xmin, ymin, xsize, ysize] for 2D and [xmin, ymin, zmin, xsize, ysize, zsize] for 3D

  * [`CenterSizeMode`](data.html#monai.data.box_utils.CenterSizeMode "monai.data.box_utils.CenterSizeMode"): represents [xcenter, ycenter, xsize, ysize] for 2D and [xcenter, ycenter, zcenter, xsize, ysize, zsize] for 3D

We currently define `StandardMode` =
[`CornerCornerModeTypeA`](data.html#monai.data.box_utils.CornerCornerModeTypeA
"monai.data.box_utils.CornerCornerModeTypeA"), and monai detection pipelines
mainly assume `boxes` are in `StandardMode`.

The implementation should be aware of:

  * remember to define class variable `name`, a dictionary that maps `spatial_dims` to [`BoxModeName`](utils.html#monai.utils.enums.BoxModeName "monai.utils.enums.BoxModeName").

  * [`boxes_to_corners()`](data.html#monai.data.box_utils.BoxMode.boxes_to_corners "monai.data.box_utils.BoxMode.boxes_to_corners") and [`corners_to_boxes()`](data.html#monai.data.box_utils.BoxMode.corners_to_boxes "monai.data.box_utils.BoxMode.corners_to_boxes") should not modify inputs in place.

_abstract _boxes_to_corners(_boxes_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L97-L116)[#](data.html#monai.data.box_utils.BoxMode.boxes_to_corners
"Link to this definition")

    

Convert the bounding boxes of the current mode to corners.

Parameters:

    

**boxes** (`Tensor`) – bounding boxes, Nx4 or Nx6 torch tensor

Returns:

    

corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch
tensor. It represents (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax,
ymax, zmax)

Return type:

    

`tuple`

Example

    
    
    boxes = torch.ones(10,6)
    boxmode = BoxMode()
    boxmode.boxes_to_corners(boxes) # will return a 6-element tuple, each element is a 10x1 tensor
    

_abstract _corners_to_boxes(_corners_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L118-L137)[#](data.html#monai.data.box_utils.BoxMode.corners_to_boxes
"Link to this definition")

    

Convert the given box corners to the bounding boxes of the current mode.

Parameters:

    

**corners** (`Sequence`) – corners of boxes, 4-element or 6-element tuple,
each element is a Nx1 torch tensor. It represents (xmin, ymin, xmax, ymax) or
(xmin, ymin, zmin, xmax, ymax, zmax)

Returns:

    

bounding boxes, Nx4 or Nx6 torch tensor

Return type:

    

`Tensor`

Example

    
    
    corners = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
    boxmode = BoxMode()
    boxmode.corners_to_boxes(corners) # will return a 10x4 tensor
    

_classmethod _get_name(_spatial_dims_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L84-L95)[#](data.html#monai.data.box_utils.BoxMode.get_name
"Link to this definition")

    

Get the mode name for the given spatial dimension using class variable `name`.

Parameters:

    

**spatial_dims** (`int`) – number of spatial dimensions of the bounding boxes.

Returns:

    

mode string name

Return type:

    

`str`

_class
_monai.data.box_utils.CenterSizeMode[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L289-L356)[#](data.html#monai.data.box_utils.CenterSizeMode
"Link to this definition")

    

A subclass of `BoxMode`.

Also represented as “ccwh” or “cccwhd”, with format of [xmin, ymin, xsize,
ysize] or [xmin, ymin, zmin, xsize, ysize, zsize].

Example

    
    
    CenterSizeMode.get_name(spatial_dims=2) # will return "ccwh"
    CenterSizeMode.get_name(spatial_dims=3) # will return "cccwhd"
    

boxes_to_corners(_boxes_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L305-L327)[#](data.html#monai.data.box_utils.CenterSizeMode.boxes_to_corners
"Link to this definition")

    

Convert the bounding boxes of the current mode to corners.

Parameters:

    

**boxes** (`Tensor`) – bounding boxes, Nx4 or Nx6 torch tensor

Returns:

    

corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch
tensor. It represents (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax,
ymax, zmax)

Return type:

    

`tuple`

Example

    
    
    boxes = torch.ones(10,6)
    boxmode = BoxMode()
    boxmode.boxes_to_corners(boxes) # will return a 6-element tuple, each element is a 10x1 tensor
    

corners_to_boxes(_corners_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L329-L356)[#](data.html#monai.data.box_utils.CenterSizeMode.corners_to_boxes
"Link to this definition")

    

Convert the given box corners to the bounding boxes of the current mode.

Parameters:

    

**corners** (`Sequence`) – corners of boxes, 4-element or 6-element tuple,
each element is a Nx1 torch tensor. It represents (xmin, ymin, xmax, ymax) or
(xmin, ymin, zmin, xmax, ymax, zmax)

Returns:

    

bounding boxes, Nx4 or Nx6 torch tensor

Return type:

    

`Tensor`

Example

    
    
    corners = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
    boxmode = BoxMode()
    boxmode.corners_to_boxes(corners) # will return a 10x4 tensor
    

_class
_monai.data.box_utils.CornerCornerModeTypeA[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L140-L164)[#](data.html#monai.data.box_utils.CornerCornerModeTypeA
"Link to this definition")

    

A subclass of `BoxMode`.

Also represented as “xyxy” or “xyzxyz”, with format of [xmin, ymin, xmax,
ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

Example

    
    
    CornerCornerModeTypeA.get_name(spatial_dims=2) # will return "xyxy"
    CornerCornerModeTypeA.get_name(spatial_dims=3) # will return "xyzxyz"
    

boxes_to_corners(_boxes_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L156-L159)[#](data.html#monai.data.box_utils.CornerCornerModeTypeA.boxes_to_corners
"Link to this definition")

    

Convert the bounding boxes of the current mode to corners.

Parameters:

    

**boxes** (`Tensor`) – bounding boxes, Nx4 or Nx6 torch tensor

Returns:

    

corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch
tensor. It represents (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax,
ymax, zmax)

Return type:

    

`tuple`

Example

    
    
    boxes = torch.ones(10,6)
    boxmode = BoxMode()
    boxmode.boxes_to_corners(boxes) # will return a 6-element tuple, each element is a 10x1 tensor
    

corners_to_boxes(_corners_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L161-L164)[#](data.html#monai.data.box_utils.CornerCornerModeTypeA.corners_to_boxes
"Link to this definition")

    

Convert the given box corners to the bounding boxes of the current mode.

Parameters:

    

**corners** (`Sequence`) – corners of boxes, 4-element or 6-element tuple,
each element is a Nx1 torch tensor. It represents (xmin, ymin, xmax, ymax) or
(xmin, ymin, zmin, xmax, ymax, zmax)

Returns:

    

bounding boxes, Nx4 or Nx6 torch tensor

Return type:

    

`Tensor`

Example

    
    
    corners = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
    boxmode = BoxMode()
    boxmode.corners_to_boxes(corners) # will return a 10x4 tensor
    

_class
_monai.data.box_utils.CornerCornerModeTypeB[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L167-L201)[#](data.html#monai.data.box_utils.CornerCornerModeTypeB
"Link to this definition")

    

A subclass of `BoxMode`.

Also represented as “xxyy” or “xxyyzz”, with format of [xmin, xmax, ymin,
ymax] or [xmin, xmax, ymin, ymax, zmin, zmax].

Example

    
    
    CornerCornerModeTypeB.get_name(spatial_dims=2) # will return "xxyy"
    CornerCornerModeTypeB.get_name(spatial_dims=3) # will return "xxyyzz"
    

boxes_to_corners(_boxes_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L183-L192)[#](data.html#monai.data.box_utils.CornerCornerModeTypeB.boxes_to_corners
"Link to this definition")

    

Convert the bounding boxes of the current mode to corners.

Parameters:

    

**boxes** (`Tensor`) – bounding boxes, Nx4 or Nx6 torch tensor

Returns:

    

corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch
tensor. It represents (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax,
ymax, zmax)

Return type:

    

`tuple`

Example

    
    
    boxes = torch.ones(10,6)
    boxmode = BoxMode()
    boxmode.boxes_to_corners(boxes) # will return a 6-element tuple, each element is a 10x1 tensor
    

corners_to_boxes(_corners_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L194-L201)[#](data.html#monai.data.box_utils.CornerCornerModeTypeB.corners_to_boxes
"Link to this definition")

    

Convert the given box corners to the bounding boxes of the current mode.

Parameters:

    

**corners** (`Sequence`) – corners of boxes, 4-element or 6-element tuple,
each element is a Nx1 torch tensor. It represents (xmin, ymin, xmax, ymax) or
(xmin, ymin, zmin, xmax, ymax, zmax)

Returns:

    

bounding boxes, Nx4 or Nx6 torch tensor

Return type:

    

`Tensor`

Example

    
    
    corners = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
    boxmode = BoxMode()
    boxmode.corners_to_boxes(corners) # will return a 10x4 tensor
    

_class
_monai.data.box_utils.CornerCornerModeTypeC[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L204-L237)[#](data.html#monai.data.box_utils.CornerCornerModeTypeC
"Link to this definition")

    

A subclass of `BoxMode`.

Also represented as “xyxy” or “xyxyzz”, with format of [xmin, ymin, xmax,
ymax] or [xmin, ymin, xmax, ymax, zmin, zmax].

Example

    
    
    CornerCornerModeTypeC.get_name(spatial_dims=2) # will return "xyxy"
    CornerCornerModeTypeC.get_name(spatial_dims=3) # will return "xyxyzz"
    

boxes_to_corners(_boxes_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L220-L228)[#](data.html#monai.data.box_utils.CornerCornerModeTypeC.boxes_to_corners
"Link to this definition")

    

Convert the bounding boxes of the current mode to corners.

Parameters:

    

**boxes** (`Tensor`) – bounding boxes, Nx4 or Nx6 torch tensor

Returns:

    

corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch
tensor. It represents (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax,
ymax, zmax)

Return type:

    

`tuple`

Example

    
    
    boxes = torch.ones(10,6)
    boxmode = BoxMode()
    boxmode.boxes_to_corners(boxes) # will return a 6-element tuple, each element is a 10x1 tensor
    

corners_to_boxes(_corners_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L230-L237)[#](data.html#monai.data.box_utils.CornerCornerModeTypeC.corners_to_boxes
"Link to this definition")

    

Convert the given box corners to the bounding boxes of the current mode.

Parameters:

    

**corners** (`Sequence`) – corners of boxes, 4-element or 6-element tuple,
each element is a Nx1 torch tensor. It represents (xmin, ymin, xmax, ymax) or
(xmin, ymin, zmin, xmax, ymax, zmax)

Returns:

    

bounding boxes, Nx4 or Nx6 torch tensor

Return type:

    

`Tensor`

Example

    
    
    corners = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
    boxmode = BoxMode()
    boxmode.corners_to_boxes(corners) # will return a 10x4 tensor
    

_class
_monai.data.box_utils.CornerSizeMode[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L240-L286)[#](data.html#monai.data.box_utils.CornerSizeMode
"Link to this definition")

    

A subclass of `BoxMode`.

Also represented as “xywh” or “xyzwhd”, with format of [xmin, ymin, xsize,
ysize] or [xmin, ymin, zmin, xsize, ysize, zsize].

Example

    
    
    CornerSizeMode.get_name(spatial_dims=2) # will return "xywh"
    CornerSizeMode.get_name(spatial_dims=3) # will return "xyzwhd"
    

boxes_to_corners(_boxes_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L256-L273)[#](data.html#monai.data.box_utils.CornerSizeMode.boxes_to_corners
"Link to this definition")

    

Convert the bounding boxes of the current mode to corners.

Parameters:

    

**boxes** (`Tensor`) – bounding boxes, Nx4 or Nx6 torch tensor

Returns:

    

corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch
tensor. It represents (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax,
ymax, zmax)

Return type:

    

`tuple`

Example

    
    
    boxes = torch.ones(10,6)
    boxmode = BoxMode()
    boxmode.boxes_to_corners(boxes) # will return a 6-element tuple, each element is a 10x1 tensor
    

corners_to_boxes(_corners_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L275-L286)[#](data.html#monai.data.box_utils.CornerSizeMode.corners_to_boxes
"Link to this definition")

    

Convert the given box corners to the bounding boxes of the current mode.

Parameters:

    

**corners** (`Sequence`) – corners of boxes, 4-element or 6-element tuple,
each element is a Nx1 torch tensor. It represents (xmin, ymin, xmax, ymax) or
(xmin, ymin, zmin, xmax, ymax, zmax)

Returns:

    

bounding boxes, Nx4 or Nx6 torch tensor

Return type:

    

`Tensor`

Example

    
    
    corners = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
    boxmode = BoxMode()
    boxmode.corners_to_boxes(corners) # will return a 10x4 tensor
    

monai.data.box_utils.StandardMode[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L140-L164)[#](data.html#monai.data.box_utils.StandardMode
"Link to this definition")

    

alias of
[`CornerCornerModeTypeA`](data.html#monai.data.box_utils.CornerCornerModeTypeA
"monai.data.box_utils.CornerCornerModeTypeA")

monai.data.box_utils.batched_nms(_boxes_ , _scores_ , _labels_ , _nms_thresh_
, _max_proposals=-1_ , _box_overlap_metric= <function
box_iou>_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L1144-L1189)[#](data.html#monai.data.box_utils.batched_nms
"Link to this definition")

    

Performs non-maximum suppression in a batched fashion. Each labels value
correspond to a category, and NMS will not be applied between elements of
different categories.

Adapted from [MIC-DKFZ/nnDetection](https://github.com/MIC-
DKFZ/nnDetection/blob/main/nndet/core/boxes/nms.py)

Parameters:

    

  * **boxes** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be `StandardMode`

  * **scores** (`Union`[`ndarray`, `Tensor`]) – prediction scores of the boxes, sized (N,). This function keeps boxes with higher scores.

  * **labels** (`Union`[`ndarray`, `Tensor`]) – indices of the categories for each one of the boxes. sized(N,), value range is (0, num_classes)

  * **nms_thresh** (`float`) – threshold of NMS. Discards all overlapping boxes with box_overlap > nms_thresh.

  * **max_proposals** (`int`) – maximum number of boxes it keeps. If `max_proposals` = -1, there is no limit on the number of boxes that are kept.

  * **box_overlap_metric** (`Callable`) – the metric to compute overlap between boxes.

Return type:

    

`Union`[`ndarray`, `Tensor`]

Returns:

    

Indexes of `boxes` that are kept after NMS.

monai.data.box_utils.box_area(_boxes_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L740-L779)[#](data.html#monai.data.box_utils.box_area
"Link to this definition")

    

This function computes the area (2D) or volume (3D) of each box. Half
precision is not recommended for this function as it may cause overflow,
especially for 3D images.

Parameters:

    

**boxes** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Nx4 or Nx6 torch
tensor or ndarray. The box mode is assumed to be `StandardMode`

Return type:

    

`Union`[`ndarray`, `Tensor`]

Returns:

    

area (2D) or volume (3D) of boxes, with size of (N,).

Example

    
    
    boxes = torch.ones(10,6)
    # we do computation with torch.float32 to avoid overflow
    compute_dtype = torch.float32
    area = box_area(boxes=boxes.to(dtype=compute_dtype))  # torch.float32, size of (10,)
    

monai.data.box_utils.box_centers(_boxes_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L634-L646)[#](data.html#monai.data.box_utils.box_centers
"Link to this definition")

    

Compute center points of boxes

Parameters:

    

**boxes** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Nx4 or Nx6 torch
tensor or ndarray. The box mode is assumed to be `StandardMode`

Return type:

    

`Union`[`ndarray`, `Tensor`]

Returns:

    

center points with size of (N, spatial_dims)

monai.data.box_utils.box_giou(_boxes1_ ,
_boxes2_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L858-L913)[#](data.html#monai.data.box_utils.box_giou
"Link to this definition")

    

Compute the generalized intersection over union (GIoU) of two sets of boxes.
The two inputs can have different shapes and the func return an NxM matrix,
(in contrary to
[`box_pair_giou()`](data.html#monai.data.box_utils.box_pair_giou
"monai.data.box_utils.box_pair_giou") , which requires the inputs to have the
same shape and returns `N` values).

Parameters:

    

  * **boxes1** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be `StandardMode`

  * **boxes2** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Mx4 or Mx6 torch tensor or ndarray. The box mode is assumed to be `StandardMode`

Return type:

    

`Union`[`ndarray`, `Tensor`]

Returns:

    

GIoU, with size of (N,M) and same data type as `boxes1`

Reference:

    

<https://giou.stanford.edu/GIoU.pdf>

monai.data.box_utils.box_iou(_boxes1_ ,
_boxes2_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L820-L855)[#](data.html#monai.data.box_utils.box_iou
"Link to this definition")

    

Compute the intersection over union (IoU) of two set of boxes.

Parameters:

    

  * **boxes1** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be `StandardMode`

  * **boxes2** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Mx4 or Mx6 torch tensor or ndarray. The box mode is assumed to be `StandardMode`

Return type:

    

`Union`[`ndarray`, `Tensor`]

Returns:

    

IoU, with size of (N,M) and same data type as `boxes1`

monai.data.box_utils.box_pair_giou(_boxes1_ ,
_boxes2_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L916-L991)[#](data.html#monai.data.box_utils.box_pair_giou
"Link to this definition")

    

Compute the generalized intersection over union (GIoU) of a pair of boxes. The
two inputs should have the same shape and the func return an (N,) array, (in
contrary to [`box_giou()`](data.html#monai.data.box_utils.box_giou
"monai.data.box_utils.box_giou") , which does not require the inputs to have
the same shape and returns `NxM` matrix).

Parameters:

    

  * **boxes1** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be `StandardMode`

  * **boxes2** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, same shape with boxes1. The box mode is assumed to be `StandardMode`

Return type:

    

`Union`[`ndarray`, `Tensor`]

Returns:

    

paired GIoU, with size of (N,) and same data type as `boxes1`

Reference:

    

<https://giou.stanford.edu/GIoU.pdf>

monai.data.box_utils.boxes_center_distance(_boxes1_ , _boxes2_ , _euclidean
=True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L680-L720)[#](data.html#monai.data.box_utils.boxes_center_distance
"Link to this definition")

    

Distance of center points between two sets of boxes

Parameters:

    

  * **boxes1** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be `StandardMode`

  * **boxes2** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Mx4 or Mx6 torch tensor or ndarray. The box mode is assumed to be `StandardMode`

  * **euclidean** (`bool`) – computed the euclidean distance otherwise it uses the l1 distance

Return type:

    

`tuple`[`Union`[`ndarray`, `Tensor`], `Union`[`ndarray`, `Tensor`],
`Union`[`ndarray`, `Tensor`]]

Returns:

    

  * The pairwise distances for every element in boxes1 and boxes2, with size of (N,M) and same data type as `boxes1`.

  * Center points of boxes1, with size of (N,spatial_dims) and same data type as `boxes1`.

  * Center points of boxes2, with size of (M,spatial_dims) and same data type as `boxes1`.

Reference:

    

[MIC-DKFZ/nnDetection](https://github.com/MIC-
DKFZ/nnDetection/blob/main/nndet/core/boxes/ops.py)

monai.data.box_utils.centers_in_boxes(_centers_ , _boxes_ , _eps
=0.01_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L649-L677)[#](data.html#monai.data.box_utils.centers_in_boxes
"Link to this definition")

    

Checks which center points are within boxes

Parameters:

    

  * **boxes** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be `StandardMode`.

  * **centers** (`Union`[`ndarray`, `Tensor`]) – center points, Nx2 or Nx3 torch tensor or ndarray.

  * **eps** (`float`) – minimum distance to border of boxes.

Return type:

    

`Union`[`ndarray`, `Tensor`]

Returns:

    

boolean array indicating which center points are within the boxes, sized (N,).

Reference:

    

[MIC-DKFZ/nnDetection](https://github.com/MIC-
DKFZ/nnDetection/blob/main/nndet/core/boxes/ops.py)

monai.data.box_utils.clip_boxes_to_image(_boxes_ , _spatial_size_ ,
_remove_empty =True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L1051-L1067)[#](data.html#monai.data.box_utils.clip_boxes_to_image
"Link to this definition")

    

This function clips the `boxes` to makes sure the bounding boxes are within
the image.

Parameters:

    

  * **boxes** – bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be `StandardMode`

  * **spatial_size** – The spatial size of the image where the boxes are attached. len(spatial_size) should be in [2, 3].

  * **remove_empty** – whether to remove the boxes that are actually empty

Returns:

    

  * clipped boxes, boxes[keep], does not share memory with original boxes

  * `keep`, it indicates whether each box in `boxes` are kept when `remove_empty=True`.

monai.data.box_utils.convert_box_mode(_boxes_ , _src_mode =None_, _dst_mode
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L546-L603)[#](data.html#monai.data.box_utils.convert_box_mode
"Link to this definition")

    

This function converts the boxes in src_mode to the dst_mode.

Parameters:

    

  * **boxes** – source bounding boxes, Nx4 or Nx6 torch tensor or ndarray.

  * **src_mode** – source box mode. If it is not given, this func will assume it is `StandardMode()`. It follows the same format with `mode` in [`get_boxmode()`](data.html#monai.data.box_utils.get_boxmode "monai.data.box_utils.get_boxmode").

  * **dst_mode** – target box mode. If it is not given, this func will assume it is `StandardMode()`. It follows the same format with `mode` in [`get_boxmode()`](data.html#monai.data.box_utils.get_boxmode "monai.data.box_utils.get_boxmode").

Returns:

    

bounding boxes with target mode, with same data type as `boxes`, does not
share memory with `boxes`

Example

    
    
    boxes = torch.ones(10,4)
    # The following three lines are equivalent
    # They convert boxes with format [xmin, ymin, xmax, ymax] to [xcenter, ycenter, xsize, ysize].
    convert_box_mode(boxes=boxes, src_mode="xyxy", dst_mode="ccwh")
    convert_box_mode(boxes=boxes, src_mode="xyxy", dst_mode=monai.data.box_utils.CenterSizeMode)
    convert_box_mode(boxes=boxes, src_mode="xyxy", dst_mode=monai.data.box_utils.CenterSizeMode())
    

monai.data.box_utils.convert_box_to_standard_mode(_boxes_ , _mode
=None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L606-L631)[#](data.html#monai.data.box_utils.convert_box_to_standard_mode
"Link to this definition")

    

Convert given boxes to standard mode. Standard mode is “xyxy” or “xyzxyz”,
representing box format of [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin,
xmax, ymax, zmax].

Parameters:

    

  * **boxes** – source bounding boxes, Nx4 or Nx6 torch tensor or ndarray.

  * **mode** – source box mode. If it is not given, this func will assume it is `StandardMode()`. It follows the same format with `mode` in [`get_boxmode()`](data.html#monai.data.box_utils.get_boxmode "monai.data.box_utils.get_boxmode").

Returns:

    

bounding boxes with standard mode, with same data type as `boxes`, does not
share memory with `boxes`

Example

    
    
    boxes = torch.ones(10,6)
    # The following two lines are equivalent
    # They convert boxes with format [xmin, xmax, ymin, ymax, zmin, zmax] to [xmin, ymin, zmin, xmax, ymax, zmax]
    convert_box_to_standard_mode(boxes=boxes, mode="xxyyzz")
    convert_box_mode(boxes=boxes, src_mode="xxyyzz", dst_mode="xyzxyz")
    

monai.data.box_utils.get_boxmode(_mode =None_, _* args_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L457-L516)[#](data.html#monai.data.box_utils.get_boxmode
"Link to this definition")

    

This function that return a [`BoxMode`](data.html#monai.data.box_utils.BoxMode
"monai.data.box_utils.BoxMode") object giving a representation of box mode

Parameters:

    

**mode** – a representation of box mode. If it is not given, this func will
assume it is `StandardMode()`.

Note

`StandardMode` =
[`CornerCornerModeTypeA`](data.html#monai.data.box_utils.CornerCornerModeTypeA
"monai.data.box_utils.CornerCornerModeTypeA"), also represented as “xyxy” for
2D and “xyzxyz” for 3D.

mode can be:

    

  1. str: choose from [`BoxModeName`](utils.html#monai.utils.enums.BoxModeName "monai.utils.enums.BoxModeName"), for example,
    
     * “xyxy”: boxes has format [xmin, ymin, xmax, ymax]

     * “xyzxyz”: boxes has format [xmin, ymin, zmin, xmax, ymax, zmax]

     * “xxyy”: boxes has format [xmin, xmax, ymin, ymax]

     * “xxyyzz”: boxes has format [xmin, xmax, ymin, ymax, zmin, zmax]

     * “xyxyzz”: boxes has format [xmin, ymin, xmax, ymax, zmin, zmax]

     * “xywh”: boxes has format [xmin, ymin, xsize, ysize]

     * “xyzwhd”: boxes has format [xmin, ymin, zmin, xsize, ysize, zsize]

     * “ccwh”: boxes has format [xcenter, ycenter, xsize, ysize]

     * “cccwhd”: boxes has format [xcenter, ycenter, zcenter, xsize, ysize, zsize]

  2. BoxMode class: choose from the subclasses of [`BoxMode`](data.html#monai.data.box_utils.BoxMode "monai.data.box_utils.BoxMode"), for example,
    
     * CornerCornerModeTypeA: equivalent to “xyxy” or “xyzxyz”

     * CornerCornerModeTypeB: equivalent to “xxyy” or “xxyyzz”

     * CornerCornerModeTypeC: equivalent to “xyxy” or “xyxyzz”

     * CornerSizeMode: equivalent to “xywh” or “xyzwhd”

     * CenterSizeMode: equivalent to “ccwh” or “cccwhd”

  3. BoxMode object: choose from the subclasses of [`BoxMode`](data.html#monai.data.box_utils.BoxMode "monai.data.box_utils.BoxMode"), for example,
    
     * CornerCornerModeTypeA(): equivalent to “xyxy” or “xyzxyz”

     * CornerCornerModeTypeB(): equivalent to “xxyy” or “xxyyzz”

     * CornerCornerModeTypeC(): equivalent to “xyxy” or “xyxyzz”

     * CornerSizeMode(): equivalent to “xywh” or “xyzwhd”

     * CenterSizeMode(): equivalent to “ccwh” or “cccwhd”

  4. None: will assume mode is `StandardMode()`

Returns:

    

BoxMode object

Example

    
    
    mode = "xyzxyz"
    get_boxmode(mode) # will return CornerCornerModeTypeA()
    

monai.data.box_utils.get_spatial_dims(_boxes =None_, _points =None_, _corners
=None_, _spatial_size =None_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L365-L454)[#](data.html#monai.data.box_utils.get_spatial_dims
"Link to this definition")

    

Get spatial dimension for the giving setting and check the validity of them.
Missing input is allowed. But at least one of the input value should be given.
It raises ValueError if the dimensions of multiple inputs do not match with
each other.

Parameters:

    

  * **boxes** – bounding boxes, Nx4 or Nx6 torch tensor or ndarray

  * **points** – point coordinates, [x, y] or [x, y, z], Nx2 or Nx3 torch tensor or ndarray

  * **corners** – corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch tensor or ndarray

  * **spatial_size** – The spatial size of the image where the boxes are attached. len(spatial_size) should be in [2, 3].

Returns:

    

spatial_dims, number of spatial dimensions of the bounding boxes.

Return type:

    

`int`

Example

    
    
    boxes = torch.ones(10,6)
    get_spatial_dims(boxes, spatial_size=[100,200,200]) # will return 3
    get_spatial_dims(boxes, spatial_size=[100,200]) # will raise ValueError
    get_spatial_dims(boxes) # will return 3
    

monai.data.box_utils.is_valid_box_values(_boxes_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L723-L737)[#](data.html#monai.data.box_utils.is_valid_box_values
"Link to this definition")

    

This function checks whether the box size is non-negative.

Parameters:

    

**boxes** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Nx4 or Nx6 torch
tensor or ndarray. The box mode is assumed to be `StandardMode`

Return type:

    

`bool`

Returns:

    

whether `boxes` is valid

monai.data.box_utils.non_max_suppression(_boxes_ , _scores_ , _nms_thresh_ ,
_max_proposals=-1_ , _box_overlap_metric= <function
box_iou>_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L1070-L1141)[#](data.html#monai.data.box_utils.non_max_suppression
"Link to this definition")

    

Non-maximum suppression (NMS).

Parameters:

    

  * **boxes** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be `StandardMode`

  * **scores** (`Union`[`ndarray`, `Tensor`]) – prediction scores of the boxes, sized (N,). This function keeps boxes with higher scores.

  * **nms_thresh** (`float`) – threshold of NMS. Discards all overlapping boxes with box_overlap > nms_thresh.

  * **max_proposals** (`int`) – maximum number of boxes it keeps. If `max_proposals` = -1, there is no limit on the number of boxes that are kept.

  * **box_overlap_metric** (`Callable`) – the metric to compute overlap between boxes.

Return type:

    

`Union`[`ndarray`, `Tensor`]

Returns:

    

Indexes of `boxes` that are kept after NMS.

Example

    
    
    boxes = torch.ones(10,6)
    scores = torch.ones(10)
    keep = non_max_suppression(boxes, scores, num_thresh=0.1)
    boxes_after_nms = boxes[keep]
    

monai.data.box_utils.spatial_crop_boxes(_boxes_ , _roi_start_ , _roi_end_ ,
_remove_empty =True_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L994-L1048)[#](data.html#monai.data.box_utils.spatial_crop_boxes
"Link to this definition")

    

This function generate the new boxes when the corresponding image is cropped
to the given ROI. When `remove_empty=True`, it makes sure the bounding boxes
are within the new cropped image.

Parameters:

    

  * **boxes** – bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be `StandardMode`

  * **roi_start** – voxel coordinates for start of the crop ROI, negative values allowed.

  * **roi_end** – voxel coordinates for end of the crop ROI, negative values allowed.

  * **remove_empty** – whether to remove the boxes that are actually empty

Returns:

    

  * cropped boxes, boxes[keep], does not share memory with original boxes

  * `keep`, it indicates whether each box in `boxes` are kept when `remove_empty=True`.

monai.data.box_utils.standardize_empty_box(_boxes_ ,
_spatial_dims_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/box_utils.py#L519-L543)[#](data.html#monai.data.box_utils.standardize_empty_box
"Link to this definition")

    

When boxes are empty, this function standardize it to shape of (0,4) or (0,6).

Parameters:

    

  * **boxes** (`Union`[`ndarray`, `Tensor`]) – bounding boxes, Nx4 or Nx6 or empty torch tensor or ndarray

  * **spatial_dims** (`int`) – number of spatial dimensions of the bounding boxes.

Return type:

    

`Union`[`ndarray`, `Tensor`]

Returns:

    

bounding boxes with shape (N,4) or (N,6), N can be 0.

Example

    
    
    boxes = torch.ones(0,)
    standardize_empty_box(boxes, 3)
    

## Video datasets[#](data.html#video-datasets "Link to this heading")

### VideoDataset[#](data.html#videodataset "Link to this heading")

_class _monai.data.video_dataset.VideoDataset(_video_source_ , _transform
=None_, _max_num_frames =None_, _color_order =RGB_, _multiprocessing =False_,
_channel_dim =0_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/video_dataset.py#L62-L151)[#](data.html#monai.data.video_dataset.VideoDataset
"Link to this definition")

    

### VideoFileDataset[#](data.html#videofiledataset "Link to this heading")

_class _monai.data.video_dataset.VideoFileDataset(_* args_, _**
kwargs_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/video_dataset.py#L154-L209)[#](data.html#monai.data.video_dataset.VideoFileDataset
"Link to this definition")

    

Video dataset from file.

This class requires that OpenCV be installed.

### CameraDataset[#](data.html#cameradataset "Link to this heading")

_class _monai.data.video_dataset.CameraDataset(_video_source_ , _transform
=None_, _max_num_frames =None_, _color_order =RGB_, _multiprocessing =False_,
_channel_dim =0_)[[source]](https://github.com/Project-
MONAI/MONAI/blob/dev/monai/data/video_dataset.py#L212-L251)[#](data.html#monai.data.video_dataset.CameraDataset
"Link to this definition")

    

Video dataset from a capture device (e.g., webcam).

This class requires that OpenCV be installed.

Parameters:

    

  * **video_source** – index of capture device. get_num_devices can be used to determine possible devices.

  * **transform** – transform to be applied to each frame.

  * **max_num_frames** – Max number of frames to iterate across. If None is passed, then the dataset will iterate infinitely.

Raises:

    

**RuntimeError** – OpenCV not installed.

__On this page

### This Page

  * [Show Source](_sources/data.rst.txt)


