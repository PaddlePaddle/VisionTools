"""
module to provide source reader of data which may stored on different places,
like: local disk, hdfs, and so on
"""

from .source import DataSource
from .source import load
from .local_source import LocalSource
