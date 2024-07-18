"""
Copyright 2024 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.

As of the Change Date specified in that file, in accordance with the Business Source License, use of this software will be governed by the Apache License, version 2.0.
"""
import numpy as np
import pandas as pd

from arcticdb.util.test import assert_frame_equal


def test_lazy_filter(lmdb_library):
    lib = lmdb_library
    sym = "test_lazy_filter"
    df = pd.DataFrame(
        {"col1": np.arange(10), "col2": np.arange(100, 110)}, index=pd.date_range("2000-01-01", periods=10)
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True)
    lazy_df = lazy_df[lazy_df["col1"].isin(0, 3, 6, 9)]
    received = lazy_df.collect().data
    expected = df.query("col1 in [0, 3, 6, 9]")

    assert_frame_equal(expected, received)