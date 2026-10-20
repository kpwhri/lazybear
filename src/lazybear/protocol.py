from typing import runtime_checkable, Protocol


@runtime_checkable
class LazyBearFrameCompatible(Protocol):

    def __as_lazybearframe__(self) -> 'LazyBearFrame':
        ...
