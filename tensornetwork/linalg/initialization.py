# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to initialize Tensor using a NumPy-like syntax."""

import warnings
from typing import Optional, Sequence, Tuple, Any, Union, Type, Callable, List
from typing import Text
import numpy as np
from tensornetwork.backends import abstract_backend
from tensornetwork import backend_contextmanager
from tensornetwork import backends
from tensornetwork.tensor import Tensor

AbstractBackend = abstract_backend.AbstractBackend


def initialize_tensor(fname: Text,
                      *fargs: Any,
                      backend: Optional[Union[Text, AbstractBackend]] = None,
                      **fkwargs: Any) -> Tensor:
  """Return a Tensor wrapping data obtained by an initialization function
  implemented in a backend. The Tensor will have the same shape as the
  underlying array that function generates, with all Edges dangling.
  This function is not intended to be called directly, but doing so should
  be safe enough.
  Args:
    fname:  Name of the method of backend to call (a string).
    *fargs: Positional arguments to the initialization method.
    backend: The backend or its name.
    **fkwargs: Keyword arguments to the initialization method.
  Returns:
    tensor: A Tensor wrapping data generated by
          (the_backend).fname(*fargs, **fkwargs), with one dangling edge per
          axis of data.
  """
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend_obj = backends.backend_factory.get_backend(backend)
  func = getattr(backend_obj, fname)
  data = func(*fargs, **fkwargs)
  tensor = Tensor(data, backend=backend)
  return tensor


def eye(N: int,
        dtype: Optional[Type[np.number]] = None,
        M: Optional[int] = None,
        backend: Optional[Union[Text, AbstractBackend]] = None) -> Tensor:
  """Return a Tensor representing a 2D array with ones on the diagonal and
  zeros elsewhere. The Tensor has two dangling Edges.
  Args:
    N (int): The first dimension of the returned matrix.
    dtype, optional: dtype of array (default np.float64).
    M (int, optional): The second dimension of the returned matrix.
    backend (optional): The backend or its name.
  Returns:
    I : Tensor of shape (N, M)
        Represents an array of all zeros except for the k'th diagonal of all
        ones.
  """
  the_tensor = initialize_tensor("eye", N, backend=backend, dtype=dtype, M=M)
  return the_tensor


def zeros(shape: Sequence[int],
          dtype: Optional[Type[np.number]] = None,
          backend: Optional[Union[Text, AbstractBackend]] = None) -> Tensor:
  """Return a Tensor of shape `shape` of all zeros.
  The Tensor has one dangling Edge per dimension.
  Args:
    shape : Shape of the array.
    dtype, optional: dtype of array (default np.float64).
    backend (optional): The backend or its name.
  Returns:
    the_tensor : Tensor of shape `shape`. Represents an array of all zeros.
  """
  the_tensor = initialize_tensor("zeros", shape, backend=backend, dtype=dtype)
  return the_tensor


def ones(shape: Sequence[int],
         dtype: Optional[Type[np.number]] = None,
         backend: Optional[Union[Text, AbstractBackend]] = None) -> Tensor:
  """Return a Tensor of shape `shape` of all ones.
  The Tensor has one dangling Edge per dimension.
  Args:
    shape : Shape of the array.
    dtype, optional: dtype of array (default np.float64).
    backend (optional): The backend or its name.
  Returns:
    the_tensor : Tensor of shape `shape`
        Represents an array of all ones.
  """
  the_tensor = initialize_tensor("ones", shape, backend=backend, dtype=dtype)
  return the_tensor


def randn(shape: Sequence[int],
          dtype: Optional[Type[np.number]] = None,
          seed: Optional[int] = None,
          backend: Optional[Union[Text, AbstractBackend]] = None) -> Tensor:
  """Return a Tensor of shape `shape` of Gaussian random floats.
  The Tensor has one dangling Edge per dimension.
  Args:
    shape : Shape of the array.
    dtype, optional: dtype of array (default np.float64).
    seed, optional: Seed for the RNG.
    backend (optional): The backend or its name.
  Returns:
    the_tensor : Tensor of shape `shape` filled with Gaussian random data.
  """
  the_tensor = initialize_tensor("randn", shape, backend=backend, seed=seed,
                                 dtype=dtype)
  return the_tensor


def random_uniform(shape: Sequence[int],
                   dtype: Optional[Type[np.number]] = None,
                   seed: Optional[int] = None,
                   boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                   backend: Optional[Union[Text, AbstractBackend]]
                   = None) -> Tensor:
  """Return a Tensor of shape `shape` of uniform random floats.
  The Tensor has one dangling Edge per dimension.
  Args:
    shape : Shape of the array.
    dtype, optional: dtype of array (default np.float64).
    seed, optional: Seed for the RNG.
    boundaries : Values lie in [boundaries[0], boundaries[1]).
    backend (optional): The backend or its name.
  Returns:
    the_tensor : Tensor of shape `shape` filled with uniform random data.
  """
  the_tensor = initialize_tensor("random_uniform", shape, backend=backend,
                                 seed=seed, boundaries=boundaries, dtype=dtype)
  return the_tensor
