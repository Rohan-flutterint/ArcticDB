"""
Copyright 2024 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.

As of the Change Date specified in that file, in accordance with the Business Source License, use of this software will be governed by the Apache License, version 2.0.
"""

from arcticdb.version_store.processing import QueryBuilder

class LazyDataFrame(QueryBuilder):
    def __init__(
            self,
            lib,
            symbol,
            as_of,
            date_range,
            row_range,
            columns,
    ):
        super().__init__()
        self._lib = lib
        self._symbol = symbol
        self._as_of = as_of
        self._date_range = date_range
        self._row_range = row_range
        self._columns = columns

    def collect(self):
        return self._lib.read(
            self._symbol,
            self._as_of,
            self._date_range,
            self._row_range,
            self._columns,
            self,
        )
