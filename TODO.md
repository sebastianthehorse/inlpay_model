* Many duplicate windows with identical labels,
* A per-horse BCE objective that never forces a single-horse prediction,
* Race-wise normalisation that leaks context you won’t have in production. (fixed?)
* Think about scaling each race individually: It will hide differences between races
* Try mini batching single races

* Consider whether the binary_result_array logic (1 if result == 1, else 0): if we want to predict all ranks, switch to a multi-class label array via the existing result_array
* Speed up data loading, pre processing and feature engineering:
    * Pre-compute once → reuse many epochs: Serialize windows (torch.save((X, y), 'race123.pt')) in a pre-processing job. Training then just torch.loads tensors—no pandas.
    * Vectorise / NumPy-only preprocessing: Replace per-row pandas code with pure NumPy or Numba functions. Typically 3-10× faster and GIL-free.
    * On-GPU windowing: If your GPU memory allows, you can move the sliding-window creation into a small CUDA kernel (as_strided + reshape) after loading the raw sequence—skips Python loops entirely.