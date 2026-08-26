"""Read-only structural inspection for the additional (non-schema) data files.

"""

import os
import sys

import numpy as np
import pandas as pd


def _fmt(v):
    # Compact (type, shape, dtype) description of an arbitrary value.
    return (
        type(v).__name__,
        getattr(v, "shape", None),
        getattr(getattr(v, "dtype", None), "name", None),
    )


def _inspect_npy(path):
    try:
        a = np.load(path, allow_pickle=False)
    except ValueError:
        # object arrays (e.g. ragged) need pickle
        a = np.load(path, allow_pickle=True)
    print("npy:", a.shape, a.dtype, f"{a.nbytes / 1e6:.1f} MB")


def _inspect_csv(path):
    head = pd.read_csv(path, nrows=5)
    print("csv columns:", list(head.columns))
    print("csv dtypes :", {c: str(t) for c, t in head.dtypes.items()})
    #  row count without loading the whole frame
    with open(path) as fh:
        n = sum(1 for _ in fh) - 1
    print(f"csv rows   : {n}   on-disk: {os.path.getsize(path) / 1e6:.1f} MB")


def _inspect_dataframe(df, label="df"):
    print(f"{label}: shape={df.shape}")
    print(f"{label} columns:", list(df.columns))
    print(f"{label} dtypes :", {c: str(t) for c, t in df.dtypes.items()})
    try:
        print(f"{label} memory : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    except Exception:  # purely informational
        pass


def _inspect_object(obj):
    # Introspect an arbitrary pickled object (e.g. a non_local_detector Environment).
    print("object type:", type(obj).__module__ + "." + type(obj).__name__)
    state = vars(obj) if hasattr(obj, "__dict__") else {}
    if not state:
        print("  (no __dict__; public attrs:)", [a for a in dir(obj) if not a.startswith("_")])
        return
    for name, val in state.items():
        kind, shape, dtype = _fmt(val)
        extra = ""
        if isinstance(val, pd.DataFrame):
            extra = f" columns={list(val.columns)}"
        elif isinstance(val, (tuple, list)):
            extra = f" len={len(val)} elem0={_fmt(val[0]) if len(val) else None}"
        elif hasattr(val, "nodes") and hasattr(val, "edges"):  # networkx graph
            extra = f" GRAPH nodes={val.number_of_nodes()} edges={val.number_of_edges()}"
        print(f"  {name}: {kind} shape={shape} dtype={dtype}{extra}")


def _inspect_pkl(path):
    # big_df pickles are pandas objects; env pickles were written by replay_trajectory_classification
    # (often not installed here) so they may be unloadable - report instead of crashing
    if os.path.basename(path).startswith("environment_"):
        print("env pickle: not introspected (written by replay_trajectory_classification; rebuilt "
              "from the DB via TrackGraph at populate/reconstruct time).")
        return
    try:
        obj = pd.read_pickle(path)
    except Exception as e:  
        print(f"could not load pickle: {type(e).__name__}: {e}")
        return
    if isinstance(obj, dict):
        print(f"pkl dict with {len(obj)} keys:", list(obj.keys()))
        for k, v in obj.items():
            if isinstance(v, pd.DataFrame):
                _inspect_dataframe(v, label=f"  [{k}]")
            else:
                print(f"  [{k}]:", _fmt(v))
    elif isinstance(obj, pd.DataFrame):
        _inspect_dataframe(obj)
    else:
        _inspect_object(obj)


def _inspect_nc(path):
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        print("nc dims  :", dict(ds.dims))
        for c in ds.coords:
            print(f"  coord {c}: shape={ds[c].shape} dtype={ds[c].dtype}")
        for v in ds.data_vars:
            print(f"  var   {v}: dims={ds[v].dims} shape={ds[v].shape} dtype={ds[v].dtype}")
        if ds.attrs:
            print("nc attrs :", dict(ds.attrs))
    finally:
        ds.close()


def _inspect_jld2(path):
    # .jld2 is HDF5 under the hood; go through every group/dataset.
    import h5py

    with h5py.File(path, "r") as f:
        def _visit(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"  GROUP    {name}  attrs={dict(obj.attrs)}")
            elif isinstance(obj, h5py.Dataset):
                print(f"  DATASET  {name}  shape={obj.shape} dtype={obj.dtype}")
            else:  # e.g. h5py.Datatype (JLD2 committed types) - has no shape
                print(f"  {type(obj).__name__.upper()} {name}")

        f.visititems(_visit)
        print("root attrs:", dict(f.attrs))


_DISPATCH = {
    ".npy": _inspect_npy,
    ".csv": _inspect_csv,
    ".pkl": _inspect_pkl,
    ".nc": _inspect_nc,
    ".jld2": _inspect_jld2,
}


def inspect(path):
    # Print the internal structure of one source file, dispatching on extension.
    ext = os.path.splitext(path)[1].lower()
    print(f"### {path}  ({ext}, {os.path.getsize(path) / 1e6:.1f} MB)")
    fn = _DISPATCH.get(ext)
    if fn is None:
        print(f"  no inspector for extension {ext!r}")
        return
    fn(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        try:
            inspect(p)
        except Exception as e:  # keep going so remaining files still report
            print(f"  ERROR inspecting {p}: {type(e).__name__}: {e}")
        print()
