"""
Copyright 2024 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.

As of the Change Date specified in that file, in accordance with the Business Source License, use of this software will be governed by the Apache License, version 2.0.
"""
import numpy as np
import pandas as pd
import pytest

from arcticdb import Col, LazyDataFrame, LazyDataFrameCollection
from arcticdb.util.test import assert_frame_equal


def test_lazy_read(lmdb_library):
    lib = lmdb_library
    sym = "test_lazy_read"
    df = pd.DataFrame(
        {"col1": np.arange(10), "col2": np.arange(100, 110)}, index=pd.date_range("2000-01-01", periods=10)
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True)
    assert isinstance(lazy_df, LazyDataFrame)
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


def test_lazy_apply(lmdb_library):
    lib = lmdb_library
    sym = "test_lazy_apply"
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


def test_lazy_apply_inline_col(lmdb_library):
    lib = lmdb_library
    sym = "test_lazy_apply_inline_col"
    df = pd.DataFrame(
        {"col1": np.arange(10), "col2": np.arange(100, 110)}, index=pd.date_range("2000-01-01", periods=10)
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True).apply("new_col", Col("col1") + Col("col2"))
    received = lazy_df.collect().data
    expected = df
    expected["new_col"] = expected["col1"] + expected["col2"]

    assert_frame_equal(expected, received)


def test_lazy_project(lmdb_library):
    lib = lmdb_library
    sym = "test_lazy_project"
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


def test_lazy_batch_one_query(lmdb_library):
    lib = lmdb_library
    syms = [f"test_lazy_batch_one_query_{idx}" for idx in range(3)]
    df = pd.DataFrame(
        {"col1": np.arange(10), "col2": np.arange(100, 110)}, index=pd.date_range("2000-01-01", periods=10)
    )
    for sym in syms:
        lib.write(sym, df)
    lazy_dfs = LazyDataFrameCollection(lib.read_batch(syms, lazy=True))
    lazy_dfs = lazy_dfs[lazy_dfs["col1"].isin(0, 3, 6, 9)]
    received = lazy_dfs.collect()
    expected = df.query("col1 in [0, 3, 6, 9]")
    for vit in received:
        assert_frame_equal(expected, vit.data)


def test_lazy_batch_collect_separately(lmdb_library):
    lib = lmdb_library
    syms = [f"test_lazy_batch_collect_separately_{idx}" for idx in range(3)]
    df = pd.DataFrame(
        {"col1": np.arange(10), "col2": np.arange(100, 110)}, index=pd.date_range("2000-01-01", periods=10)
    )
    for sym in syms:
        lib.write(sym, df)
    lazy_dfs = lib.read_batch(syms, lazy=True)
    lazy_df_0 = lazy_dfs[0]
    lazy_df_1 = lazy_dfs[1]
    lazy_df_2 = lazy_dfs[2]
    lazy_df_0 = lazy_df_0[lazy_df_0["col1"].isin(0, 3, 6, 9)]
    lazy_df_2 = lazy_df_2[lazy_df_2["col1"].isin(2, 4, 8)]
    expected_0 = df.query("col1 in [0, 3, 6, 9]")
    expected_1 = df
    expected_2 = df.query("col1 in [2, 4, 8]")
    received_0 = lazy_df_0.collect().data
    received_1 = lazy_df_1.collect().data
    received_2 = lazy_df_2.collect().data
    assert_frame_equal(expected_0, received_0)
    assert_frame_equal(expected_1, received_1)
    assert_frame_equal(expected_2, received_2)


def test_lazy_batch_separate_queries_collect_together(lmdb_library):
    lib = lmdb_library
    syms = [f"test_lazy_batch_separate_queries_collect_together_{idx}" for idx in range(3)]
    df = pd.DataFrame(
        {"col1": np.arange(10), "col2": np.arange(100, 110)}, index=pd.date_range("2000-01-01", periods=10)
    )
    for sym in syms:
        lib.write(sym, df)
    lazy_dfs = lib.read_batch(syms, lazy=True)
    lazy_df_0 = lazy_dfs[0]
    lazy_df_2 = lazy_dfs[2]
    lazy_df_0 = lazy_df_0[lazy_df_0["col1"].isin(0, 3, 6, 9)]
    lazy_df_2 = lazy_df_2[lazy_df_2["col1"].isin(2, 4, 8)]
    expected_0 = df.query("col1 in [0, 3, 6, 9]")
    expected_1 = df
    expected_2 = df.query("col1 in [2, 4, 8]")

    received = LazyDataFrameCollection(lazy_dfs).collect()
    assert_frame_equal(expected_0, received[0].data)
    assert_frame_equal(expected_1, received[1].data)
    assert_frame_equal(expected_2, received[2].data)
