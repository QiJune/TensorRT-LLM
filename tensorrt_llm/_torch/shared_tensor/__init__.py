from .shared_tensor import (SharedTensorContainer,
                            _SharedTensorRebuildMethodRegistry,
                            release_exported_handles)

# Initialize the registry when the package is imported
_SharedTensorRebuildMethodRegistry.initialize()

__all__ = [
    'SharedTensorContainer',
    'release_exported_handles',
]
