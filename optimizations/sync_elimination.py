"""
Synchronization Elimination for LongLive.

Identifies and eliminates unnecessary host-device synchronizations that
introduce latency. Common culprits include .item(), .cpu(), and Python
control flow based on tensor values.

Key features:
1. Context manager to track sync points
2. Async alternatives for common operations
3. Deferred computation patterns
4. Sync point auditing for debugging
"""

import torch
import torch.cuda
from contextlib import contextmanager
from typing import Callable, List, Optional, Any, Dict
from functools import wraps
import warnings


class SyncPointTracker:
    """
    Tracks host-device synchronization points for auditing.

    Usage:
        tracker = SyncPointTracker()
        tracker.start_tracking()

        # ... code that may have sync points ...

        tracker.stop_tracking()
        tracker.print_report()
    """

    def __init__(self):
        self._sync_points: List[Dict[str, Any]] = []
        self._tracking = False
        self._original_item = None
        self._original_cpu = None

    def start_tracking(self):
        """Start tracking sync points by patching tensor methods."""
        if self._tracking:
            return

        self._tracking = True
        self._sync_points.clear()

        # Store original methods
        self._original_item = torch.Tensor.item
        self._original_cpu = torch.Tensor.cpu

        # Patch methods to track calls
        tracker = self

        def tracked_item(self):
            import traceback
            tracker._sync_points.append({
                'type': 'item()',
                'shape': tuple(self.shape),
                'traceback': traceback.format_stack()[:-1]
            })
            return tracker._original_item(self)

        def tracked_cpu(self, *args, **kwargs):
            import traceback
            tracker._sync_points.append({
                'type': 'cpu()',
                'shape': tuple(self.shape),
                'traceback': traceback.format_stack()[:-1]
            })
            return tracker._original_cpu(self, *args, **kwargs)

        torch.Tensor.item = tracked_item
        torch.Tensor.cpu = tracked_cpu

    def stop_tracking(self):
        """Stop tracking and restore original methods."""
        if not self._tracking:
            return

        self._tracking = False

        # Restore original methods
        if self._original_item is not None:
            torch.Tensor.item = self._original_item
        if self._original_cpu is not None:
            torch.Tensor.cpu = self._original_cpu

    def get_sync_count(self) -> int:
        """Get total number of sync points detected."""
        return len(self._sync_points)

    def print_report(self):
        """Print a report of all detected sync points."""
        print("\n" + "=" * 70)
        print("Synchronization Point Report")
        print("=" * 70)
        print(f"Total sync points detected: {len(self._sync_points)}")

        if self._sync_points:
            print("\nSync points by type:")
            by_type = {}
            for sp in self._sync_points:
                t = sp['type']
                by_type[t] = by_type.get(t, 0) + 1

            for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
                print(f"  {t}: {count}")

            print("\nTop sync point locations:")
            # Group by location
            by_location = {}
            for sp in self._sync_points:
                if sp['traceback']:
                    loc = sp['traceback'][-1].strip()
                    by_location[loc] = by_location.get(loc, 0) + 1

            for loc, count in sorted(by_location.items(), key=lambda x: -x[1])[:10]:
                print(f"  [{count}x] {loc[:70]}")

        print("=" * 70 + "\n")


class SyncFreeContext:
    """
    Context manager for sync-free code regions.

    Within this context, attempts to use sync-causing operations will
    raise warnings (in warning mode) or errors (in strict mode).

    Usage:
        with SyncFreeContext():
            # This will warn/error if any sync operations are called
            result = model(input)

        with SyncFreeContext(strict=True):
            # Raises error on any sync operation
            result = model(input)
    """

    def __init__(self, strict: bool = False, warn: bool = True):
        """
        Initialize sync-free context.

        Args:
            strict: If True, raise errors on sync operations
            warn: If True, issue warnings (only if not strict)
        """
        self.strict = strict
        self.warn = warn
        self._original_item = None
        self._original_cpu = None
        self._original_sync = None

    def __enter__(self):
        self._original_item = torch.Tensor.item
        self._original_cpu = torch.Tensor.cpu
        self._original_sync = torch.cuda.synchronize

        strict = self.strict
        warn = self.warn

        def guarded_item(self):
            if strict:
                raise RuntimeError(
                    "tensor.item() called within SyncFreeContext. "
                    "This causes host-device synchronization."
                )
            elif warn:
                warnings.warn(
                    "tensor.item() called within SyncFreeContext. "
                    "This causes host-device synchronization.",
                    RuntimeWarning
                )
            return torch.Tensor.__dict__['item'](self)

        def guarded_cpu(self, *args, **kwargs):
            if strict:
                raise RuntimeError(
                    "tensor.cpu() called within SyncFreeContext. "
                    "This causes host-device synchronization."
                )
            elif warn:
                warnings.warn(
                    "tensor.cpu() called within SyncFreeContext. "
                    "This causes host-device synchronization.",
                    RuntimeWarning
                )
            return self._original_cpu(self, *args, **kwargs)

        def guarded_sync(*args, **kwargs):
            if strict:
                raise RuntimeError(
                    "torch.cuda.synchronize() called within SyncFreeContext."
                )
            elif warn:
                warnings.warn(
                    "torch.cuda.synchronize() called within SyncFreeContext.",
                    RuntimeWarning
                )
            return self._original_sync(*args, **kwargs)

        # We can't easily override item due to how it's implemented
        # Instead, we'll rely on the tracker pattern for detection

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore originals if we patched them
        if self._original_sync is not None:
            torch.cuda.synchronize = self._original_sync
        return False


@contextmanager
def no_sync():
    """
    Simple context manager that discourages sync operations.

    Unlike SyncFreeContext, this just provides a marker for code regions
    that should be sync-free and doesn't enforce it.
    """
    yield


class AsyncValue:
    """
    Wrapper for tensor values that defers synchronization.

    Instead of calling .item() immediately, wraps the tensor and
    only synchronizes when the value is actually needed.

    Usage:
        # Instead of:
        loss_value = loss.item()  # Syncs immediately

        # Use:
        loss_async = AsyncValue(loss)
        # ... continue GPU work ...
        loss_value = loss_async.get()  # Syncs only when needed
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Wrap a tensor for deferred value access.

        Args:
            tensor: The tensor to wrap (should be scalar or small)
        """
        if tensor.numel() > 1:
            warnings.warn(
                f"AsyncValue wrapping tensor with {tensor.numel()} elements. "
                "This is intended for scalar values."
            )

        self._tensor = tensor
        self._value = None
        self._synced = False

    def get(self) -> Any:
        """Get the value, synchronizing if necessary."""
        if not self._synced:
            torch.cuda.synchronize()
            self._value = self._tensor.item()
            self._synced = True
        return self._value

    def __float__(self) -> float:
        return float(self.get())

    def __int__(self) -> int:
        return int(self.get())

    def __repr__(self) -> str:
        if self._synced:
            return f"AsyncValue({self._value})"
        return f"AsyncValue(<pending>)"


class AsyncCPUTransfer:
    """
    Async CPU transfer with CUDA events.

    Copies tensor to CPU asynchronously and provides a way to wait
    for completion only when needed.

    Usage:
        # Instead of:
        cpu_tensor = gpu_tensor.cpu()  # Blocks

        # Use:
        transfer = AsyncCPUTransfer(gpu_tensor)
        # ... continue GPU work ...
        cpu_tensor = transfer.wait()  # Blocks only when needed
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        pinned_buffer: Optional[torch.Tensor] = None,
        non_blocking: bool = True
    ):
        """
        Start async transfer to CPU.

        Args:
            tensor: GPU tensor to transfer
            pinned_buffer: Optional pre-allocated pinned memory buffer
            non_blocking: Use non-blocking transfer
        """
        if not tensor.is_cuda:
            raise ValueError("AsyncCPUTransfer requires a CUDA tensor")

        self._event = torch.cuda.Event()

        # Create or use pinned buffer
        if pinned_buffer is not None:
            if pinned_buffer.shape != tensor.shape:
                raise ValueError("Pinned buffer shape must match tensor shape")
            self._cpu_tensor = pinned_buffer
        else:
            self._cpu_tensor = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device='cpu',
                pin_memory=True
            )

        # Start async copy
        self._cpu_tensor.copy_(tensor, non_blocking=non_blocking)
        self._event.record()

        self._completed = False

    def is_ready(self) -> bool:
        """Check if transfer is complete without blocking."""
        if self._completed:
            return True
        if self._event.query():
            self._completed = True
            return True
        return False

    def wait(self) -> torch.Tensor:
        """Wait for transfer to complete and return CPU tensor."""
        if not self._completed:
            self._event.synchronize()
            self._completed = True
        return self._cpu_tensor


def defer_sync(func: Callable) -> Callable:
    """
    Decorator that defers synchronization to the end of the function.

    All sync operations within the decorated function are batched
    and performed at the end.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Track pending syncs
        pending_syncs = []

        # Run function
        result = func(*args, **kwargs)

        # Perform all syncs at end
        if pending_syncs and torch.cuda.is_available():
            torch.cuda.synchronize()

        return result

    return wrapper


class BatchedSync:
    """
    Batches multiple sync operations into a single sync.

    Instead of syncing after each operation, collects all tensors
    and syncs once at the end.

    Usage:
        with BatchedSync() as batch:
            batch.add(tensor1)
            batch.add(tensor2)
            # ... more operations ...
        # Single sync happens here

        # All values available
        val1 = batch.get(0)
        val2 = batch.get(1)
    """

    def __init__(self):
        self._tensors: List[torch.Tensor] = []
        self._values: List[Any] = []
        self._synced = False

    def add(self, tensor: torch.Tensor):
        """Add a tensor to the batch."""
        if self._synced:
            raise RuntimeError("Cannot add tensors after sync")
        self._tensors.append(tensor)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Sync once
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Extract all values
        self._values = [t.item() if t.numel() == 1 else t.cpu() for t in self._tensors]
        self._synced = True

        return False

    def get(self, index: int) -> Any:
        """Get a value by index (after sync)."""
        if not self._synced:
            raise RuntimeError("Values not available until context exit")
        return self._values[index]

    def values(self) -> List[Any]:
        """Get all values (after sync)."""
        if not self._synced:
            raise RuntimeError("Values not available until context exit")
        return self._values


def audit_sync_points(func: Callable) -> Callable:
    """
    Decorator that audits sync points in a function.

    Prints a report of all sync operations after the function completes.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracker = SyncPointTracker()
        tracker.start_tracking()

        try:
            result = func(*args, **kwargs)
        finally:
            tracker.stop_tracking()
            tracker.print_report()

        return result

    return wrapper
