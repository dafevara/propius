__version__ = "0.1"


from propius import data
from propius import model

from .data import stream_csv
from .model import SparseCorrelationMatrix

__all__ = [
    "stream_csv",
    "SparseCorrelationMatrix"
]
