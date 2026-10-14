"""Dataset loaders for the MemSat benchmarks.

`longmemeval` is the primary substrate. `locomo` and the Arm B filler sessions
land in P3 and P7 respectively; the split names they will occupy are declared
here so `available_splits` reports the gap rather than hiding it.
"""

from __future__ import annotations

from msat.datasets.longmemeval import (
    RAM_CEILING_BYTES,
    RAW_FILES,
    assert_data_integrity,
    available_splits,
    count_records,
    iter_metadata,
    iter_questions,
    iter_splits,
    raw_path,
    resolve_data_dir,
)
from msat.datasets.schema import (
    ABSTENTION_SUFFIX,
    DATASET_NAME,
    DATASET_VERSION,
    QUESTION_TYPES,
    SCALES,
    CorpusStats,
    DataIntegrityError,
    Question,
    Session,
    Turn,
    is_abstention,
)

__all__ = [
    "ABSTENTION_SUFFIX",
    "DATASET_NAME",
    "DATASET_VERSION",
    "QUESTION_TYPES",
    "RAM_CEILING_BYTES",
    "RAW_FILES",
    "SCALES",
    "CorpusStats",
    "DataIntegrityError",
    "Question",
    "Session",
    "Turn",
    "assert_data_integrity",
    "available_splits",
    "count_records",
    "is_abstention",
    "iter_metadata",
    "iter_questions",
    "iter_splits",
    "raw_path",
    "resolve_data_dir",
]
