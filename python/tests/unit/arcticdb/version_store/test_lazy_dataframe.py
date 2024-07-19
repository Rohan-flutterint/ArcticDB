"""
Copyright 2024 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.

As of the Change Date specified in that file, in accordance with the Business Source License, use of this software will be governed by the Apache License, version 2.0.
"""
import numpy as np
import pandas as pd
import pytest

from arcticdb.util.test import assert_frame_equal


def test_lazy_read(lmdb_library):
    lib = lmdb_library
    sym = "test_lazy_read"
    df = pd.DataFrame(
        {"col1": np.arange(10), "col2": np.arange(100, 110)}, index=pd.date_range("2000-01-01", periods=10)
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True)
    received = lazy_df.collect().data

    assert_frame_equal(df, received)


def test_lazy_date_range(lmdb_library):
    lib = lmdb_library
    sym = "test_lazy_date_range"
    df = pd.DataFrame(
        {"col1": np.arange(10), "col2": np.arange(100, 110)}, index=pd.date_range("2000-01-01", periods=10)
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True)
    lazy_df = lazy_df.date_range((pd.Timestamp("2000-01-02"), pd.Timestamp("2000-01-09")))
    received = lazy_df.collect().data
    expected = df.iloc[1:9]

    assert_frame_equal(expected, received)


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


def test_lazy_apply_1(lmdb_library):
    lib = lmdb_library
    sym = "test_lazy_apply_1"
    df = pd.DataFrame(
        {"col1": np.arange(10), "col2": np.arange(100, 110)}, index=pd.date_range("2000-01-01", periods=10)
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True)
    lazy_df = lazy_df.apply("new_col", lazy_df["col1"] + lazy_df["col2"])
    received = lazy_df.collect().data
    expected = df
    expected["new_col"] = expected["col1"] + expected["col2"]

    assert_frame_equal(expected, received)


def test_lazy_apply_2(lmdb_library):
    lib = lmdb_library
    sym = "test_lazy_apply_2"
    df = pd.DataFrame(
        {"col1": np.arange(10), "col2": np.arange(100, 110)}, index=pd.date_range("2000-01-01", periods=10)
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True)
    lazy_df["new_col"] = lazy_df["col1"] + lazy_df["col2"]
    received = lazy_df.collect().data
    expected = df
    expected["new_col"] = expected["col1"] + expected["col2"]

    assert_frame_equal(expected, received)


def test_lazy_groupby(lmdb_library):
    lib = lmdb_library
    sym = "test_lazy_groupby"
    df = pd.DataFrame({"col1": [0, 1, 0, 1, 2, 2], "col2": np.arange(6)})
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True)
    lazy_df = lazy_df.groupby("col1").agg({"col2": "sum"})
    received = lazy_df.collect().data
    received.sort_index(inplace=True)
    expected = df.groupby("col1").agg({"col2": "sum"})

    assert_frame_equal(expected, received)


def test_lazy_resample(lmdb_library):
    lib = lmdb_library
    sym = "test_lazy_resample"
    df = pd.DataFrame(
        {"col1": np.arange(10), "col2": np.arange(100, 110)}, index=pd.date_range("2000-01-01", periods=10)
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True)
    lazy_df = lazy_df.resample("D").agg({"col1": "sum", "col2": "first"})
    received = lazy_df.collect().data
    expected = df.resample("D").agg({"col1": "sum", "col2": "first"})

    assert_frame_equal(expected, received)


def test_lazy_chaining(lmdb_library):
    lib = lmdb_library
    sym = "test_lazy_chaining"
    idx = [0, 1, 2, 3, 1000, 1001]
    idx = np.array(idx, dtype="datetime64[ns]")
    df = pd.DataFrame({"col": np.arange(6)}, index=idx)
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True).resample("us").agg({"col": "sum"})
    lazy_df["new_col"] = lazy_df["col"] * 3
    received = lazy_df.collect().data

    expected = df.resample("us").agg({"col": "sum"})
    expected["new_col"] = expected["col"] * 3
    assert_frame_equal(expected, received, check_dtype=False)