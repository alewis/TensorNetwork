import math

class Array():
  def __init__(self,
               array: Any,
               backend: Optional[Union[Text, BaseBackend]] = None) -> None:
    if backend is None:
      backend = backend_contextmanager.get_default_backend()
    backend_obj = backends.backend_factory.get_backend(backend)
    self.backend = backend_obj
    self.array = array
    self.shape = array.shape
    self.size = math.prod(self.shape)
    self.real = array.real
    self.imag = array.imag
    self.ndim = len(self.shape)

  def dtype(self) -> Any: # To maintain backend independence
    return self.array.dtype

  def T(self, axes: Sequence[int]):
    array_T = self.backend.transpose(self.array, axes=axes)
    return Array(array_T, backend=self.backend)

  def H(self, axes: Sequence[int]):
    star = self.backend.conj(self.array)
    array_H = self.backend.transpose(star, axes=axes)
    return Array(array_H, backend=self.backend)
