
import os
import re
import glob
import json
import functools

import datajoint as dj
import numpy as np
import pandas as pd
import xarray as xr
import joblib

from spyglass.common import Nwbfile
from spyglass.common.custom_nwbfile import AnalysisNwbfile
from spyglass.utils import SpyglassMixin, SpyglassMixinPart
from spyglass.utils.nwb_helper_fn import get_nwb_file


schema = dj.schema("alison_additional_data_files")

_BIG_DF_DIR = "/stelmo/alison/big_df_pkls/"
_DECODE_DIR = "/stelmo/alison/decodes/"
_THETA_DIR = "/stelmo/alison/figs26/"
_FILE_TYPES = ("pkl_bigdf", "csv", "npy", "nc", "pkl_env")


def _san(name, maxlen=64):
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    return s[:maxlen] or "obj"


def _nwb_safe_array(values):
    arr = np.asarray(values)
    if arr.dtype.kind in ("O", "U"):
        return arr.astype("S")
    return arr


def _decode_if_bytes(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8")
    return arr


def _json_to_array(obj):
    return np.frombuffer(json.dumps(obj).encode("utf-8"), dtype=np.uint8)


def _array_to_json(arr):
    return json.loads(bytes(np.asarray(arr)).decode("utf-8"))


def _json_default(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _df_to_storage(df):
    trivial = isinstance(df.index, pd.RangeIndex) and df.index.name is None
    store = df.reset_index(drop=True) if trivial else df.reset_index()
    n_index = 0 if trivial else store.shape[1] - df.shape[1]
    orig_cols = [str(c) for c in store.columns]
    safe_cols, seen = [], {}
    for c in orig_cols:
        s = base = re.sub(r"[^0-9A-Za-z_.\-]", "_", c); k = 1
        while s in seen:
            s = f"{base}_{k}"; k += 1
        seen[s] = True; safe_cols.append(s)
    store.columns = safe_cols
    meta = {
        "index_cols": safe_cols[:n_index],
        "dtypes": {safe_cols[i]: str(store.dtypes.iloc[i]) for i in range(len(safe_cols))},
        "json_cols": [safe_cols[i] for i in range(len(safe_cols)) if store.dtypes.iloc[i] == object],
        "col_rename": {s: o for s, o in zip(safe_cols, orig_cols) if s != o},
    }
    for c in meta["json_cols"]:
        store[c] = store[c].map(lambda v: json.dumps(v, default=_json_default))
    return store, meta


def _restore_df(store_df, meta):
    df = store_df.copy()

    json_cols = set(meta.get("json_cols", []))
    for c in json_cols:
        if c in df.columns:
            df[c] = df[c].map(json.loads).astype(object)

    for c, dt in meta["dtypes"].items():
        if c in df.columns and c not in json_cols and str(df[c].dtype) != dt:
            try:
                df[c] = df[c].astype(dt)
            except (TypeError, ValueError):
                pass
    rename = meta.get("col_rename", {})
    if rename:
        df = df.rename(columns=rename)
    if meta["index_cols"]:
        df = df.set_index([rename.get(c, c) for c in meta["index_cols"]])
    return df


def _df_objects(name, df, source_key):
    store, meta = _df_to_storage(df)
    return [
        (name, store, "data", source_key, ""),
        (f"meta__{name}", _json_to_array(meta), "meta", source_key, ""),
    ]


def _env_to_nwb_objects(env):
    out = []
    for name, val in vars(env).items():
        if val is None:
            continue
        if isinstance(val, pd.DataFrame):
            out.append((name, val))
        elif isinstance(val, np.ndarray):
            out.append((name, val))
        elif isinstance(val, tuple) and len(val) and isinstance(val[0], np.ndarray):
            for i, arr in enumerate(val):
                out.append((f"{name}_{i}", np.asarray(arr)))
        elif hasattr(val, "nodes") and hasattr(val, "edges"):
            node_positions = np.asarray(
                [v.get("pos") for v in dict(val.nodes).values()], dtype=float
            )
            edges = np.asarray(list(val.edges))
            out.append((f"{name}_node_positions", node_positions))
            out.append((f"{name}_edges", edges))
    return out


_ANIMALS = ("senor", "chimi", "wilbur", "peanut", "j16")

_SOURCE_ROOT = "/stelmo/alison/"
_DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def _animal_from_path(path):
    low = os.path.basename(path).lower()
    for a in _ANIMALS:
        if a in low:
            return a
    return None


def _first_nwb_for_animal(animal):
    import re

    names = Nwbfile().fetch("nwb_file_name")
    cands = []
    for n in names:
        m = re.match(rf"^{re.escape(animal)}(\d{{8}})_?\.nwb$", n, flags=re.IGNORECASE)
        if m:
            cands.append((m.group(1), n))
    if not cands:
        raise ValueError(f"no Nwbfile found for animal {animal!r}")
    return min(cands)[1]


@functools.lru_cache(maxsize=1)
def _nwb_name_map():
    return {n.lower(): n for n in Nwbfile().fetch("nwb_file_name")}


def _resolve_nwb_name(name):
    return _nwb_name_map().get(name.lower(), name)


def _nwb_from_decode_path(path):
    base = os.path.basename(path)
    if base.startswith("environment_"):
        base = base[len("environment_"):]
    return _resolve_nwb_name(base.split(".nwb", 1)[0] + ".nwb")


def anchor_for_path(path):
    if os.path.dirname(path).rstrip("/").endswith("decodes"):
        return _nwb_from_decode_path(path)
    animal = _animal_from_path(path)
    if animal is None:
        raise ValueError(f"cannot determine animal for {path}")
    return _first_nwb_for_animal(animal)


@schema
class AdditionalDataFilesSelection(dj.Manual):
    definition = """
    # One row per additional (non-schema) source file to fold into the Export as NWB.
    -> Nwbfile                    # anchor session: whose registered NWB create() copies (also the index)
    source_path : varchar(255)    # absolute /stelmo/alison/... path; the only key we need (also what we log)
    ---
    file_type   : enum('pkl_bigdf','csv','npy','nc','pkl_env')
    """


@schema
class AdditionalDataFiles(SpyglassMixin, dj.Computed):
    definition = """
    -> AdditionalDataFilesSelection
    ---
    -> AnalysisNwbfile
    n_objects : int               # number of NWB objects this file produced
    """

    class Object(SpyglassMixinPart):
        definition = """
        -> master
        object_name : varchar(80)     # unique within this file's analysis file (add_scratch requires unique)
        ---
        object_id   : varchar(40)     # returned by add_nwb_object
        object_kind : enum('dynamictable','scratch')
        role        : enum('data','coord','meta')  # data_var / coordinate / pandas index+dtype sidecar
        source_key  : varchar(200)    # sub-name in source (dict key / data_var / coord / hdf5 path / env field)
        dims        : varchar(200)    # comma-joined dim names in order (e.g. 'time,state,position'); '' if n/a
        """

    def make(self, key):
        sel = (AdditionalDataFilesSelection & key).fetch1()
        ftype = sel["file_type"]
        path = sel["source_path"]
        nwb_file_name = sel["nwb_file_name"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"source_path does not exist: {path}")

        nwb_analysis_file = AnalysisNwbfile()
        analysis_file_name = nwb_analysis_file.create(nwb_file_name)
        key["analysis_file_name"] = analysis_file_name

        objects = self._read_objects(ftype, path)

        part_pk = {"nwb_file_name": nwb_file_name, "source_path": path}
        part_rows = []
        for object_name, obj, role, source_key, dims_csv in objects:
            object_id = nwb_analysis_file.add_nwb_object(
                analysis_file_name=analysis_file_name,
                nwb_object=obj,
                table_name=object_name,
            )
            part_rows.append(
                dict(
                    part_pk,
                    object_name=object_name,
                    object_id=object_id,
                    object_kind="dynamictable" if isinstance(obj, pd.DataFrame) else "scratch",
                    role=role,
                    source_key=source_key,
                    dims=dims_csv,
                )
            )

        nwb_analysis_file.add(nwb_file_name, analysis_file_name)
        self.insert1(dict(key, n_objects=len(part_rows)))
        self.Object.insert(part_rows)
        print(f"Populated AdditionalDataFiles ({len(part_rows)} objects) for {path}")

    @staticmethod
    def _read_objects(ftype, path):
        objects = []

        if ftype == "csv":
            objects += _df_objects("table", pd.read_csv(path), "csv")

        elif ftype == "npy":
            try:
                arr = np.load(path, allow_pickle=False)
            except ValueError:
                arr = np.load(path, allow_pickle=True)
            objects.append(("array", np.asarray(arr), "data", "npy", ""))

        elif ftype == "pkl_bigdf":
            blob = pd.read_pickle(path)
            items = blob.items() if isinstance(blob, dict) else [("table", blob)]
            for subkey, df in items:
                objects += _df_objects(_san(subkey), df, str(subkey))

        elif ftype == "nc":
            ds = xr.open_dataset(path)
            try:
                for var in ds.data_vars:
                    da = ds[var]
                    objects.append(
                        (_san(var), _nwb_safe_array(da.values), "data", str(var), ",".join(da.dims))
                    )
                for coord in ds.coords:
                    objects.append(
                        (
                            _san(f"coord__{coord}"),
                            _nwb_safe_array(ds[coord].values),
                            "coord",
                            str(coord),
                            str(coord),
                        )
                    )
            finally:
                ds.close()

        elif ftype == "pkl_env":
            env = _build_environment_from_path(path)
            for fld, obj in _env_to_nwb_objects(env):
                if isinstance(obj, np.ndarray):
                    obj = _nwb_safe_array(obj)
                objects.append((_san(fld), obj, "data", fld, ""))

        else:
            raise ValueError(f"unknown file_type {ftype!r}")

        return objects

    def _load_objects(self):
        key = self.fetch1("KEY")
        analysis_file_name = (self & key).fetch1("analysis_file_name")
        abs_path = AnalysisNwbfile().get_abs_path(analysis_file_name)
        nwbf = get_nwb_file(abs_path)
        out = {}
        for p in (self.Object & key).fetch(as_dict=True):
            obj = nwbf.objects[p["object_id"]]
            if p["object_kind"] == "dynamictable":
                out[p["object_name"]] = obj.to_dataframe().reset_index(drop=True)
            elif hasattr(obj, "data"):
                out[p["object_name"]] = np.asarray(obj.data[()])
            else:
                out[p["object_name"]] = obj
        return out

    def fetch_object(self, object_name):
        return self._load_objects()[object_name]

    def _dataframes(self):
        loaded = self._load_objects()
        parts = (self.Object & self.fetch1("KEY") & {"role": "data", "object_kind": "dynamictable"}).fetch(
            as_dict=True
        )
        out = {}
        for p in parts:
            df = loaded[p["object_name"]]
            meta_name = f"meta__{p['object_name']}"
            if meta_name in loaded:
                df = _restore_df(df, _array_to_json(loaded[meta_name]))
            out[p["source_key"]] = df
        return out

    def fetch1_dataframe(self):
        dfs = self._dataframes()
        assert len(dfs) == 1, f"expected exactly one table, found {len(dfs)}"
        return next(iter(dfs.values()))

    def fetch1_xarray(self):
        key = self.fetch1("KEY")
        loaded = self._load_objects()
        parts = (self.Object & key).fetch(as_dict=True)
        coords = {
            p["source_key"]: _decode_if_bytes(loaded[p["object_name"]])
            for p in parts
            if p["role"] == "coord"
        }
        data_vars = {
            p["source_key"]: (p["dims"].split(","), _decode_if_bytes(loaded[p["object_name"]]))
            for p in parts
            if p["role"] == "data"
        }
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def reconstruct_to(self, data_dir=_DEFAULT_DATA_DIR, source_root=_SOURCE_ROOT, overwrite=False):
        sel = (AdditionalDataFilesSelection & self.fetch1("KEY")).fetch1()
        src, ftype = sel["source_path"], sel["file_type"]
        out = os.path.join(data_dir, os.path.relpath(src, source_root))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and not overwrite:
            return out

        if ftype == "csv":
            df = self.fetch1_dataframe()
            df.to_csv(out, index=bool(df.index.name) or isinstance(df.index, pd.MultiIndex))
        elif ftype == "npy":
            np.save(out, self.fetch_object("array"))
        elif ftype == "pkl_bigdf":
            dfs = self._dataframes()
            if set(dfs) == {"table"}:
                pd.to_pickle(dfs["table"], out)
            else:
                pd.to_pickle(dfs, out)
        elif ftype == "nc":
            self.fetch1_xarray().to_netcdf(out)
        elif ftype == "pkl_env":
            _reconstruct_environment(src, out)
        else:
            raise ValueError(f"unknown file_type {ftype!r}")
        return out

    def verify_reconstruction(self, source_root=_SOURCE_ROOT, rtol=1e-6, atol=1e-9):
        import tempfile

        sel = (AdditionalDataFilesSelection & self.fetch1("KEY")).fetch1()
        src, ftype = sel["source_path"], sel["file_type"]
        with tempfile.TemporaryDirectory() as td:
            out = self.reconstruct_to(data_dir=td, source_root=source_root, overwrite=True)
            try:
                if ftype == "nc":
                    a, b = xr.open_dataset(src), xr.open_dataset(out)
                    try:
                        for v in a.data_vars:
                            if not np.allclose(a[v].values, b[v].values, rtol=rtol, atol=atol,
                                               equal_nan=True):
                                return {"ok": False, "detail": f"data_var {v} differs"}
                    finally:
                        a.close(); b.close()
                elif ftype == "pkl_env":
                    a = _build_environment_from_path(src)
                    b = joblib.load(out); b.fit_place_grid()
                    for attr in ("is_track_interior_", "place_bin_centers_", "place_bin_edges_"):
                        if not np.allclose(np.asarray(getattr(a, attr)),
                                           np.asarray(getattr(b, attr)), rtol=rtol, atol=atol):
                            return {"ok": False, "detail": f"env attr {attr} differs"}
                elif ftype == "npy":
                    if not np.allclose(np.load(src), np.load(out), rtol=rtol, atol=atol,
                                       equal_nan=True):
                        return {"ok": False, "detail": "array differs"}
                elif ftype in ("csv", "pkl_bigdf"):
                    a = pd.read_csv(src) if ftype == "csv" else pd.read_pickle(src)
                    b = pd.read_csv(out) if ftype == "csv" else pd.read_pickle(out)
                    if isinstance(a, dict):
                        ok = a.keys() == b.keys() and all(a[k].shape == b[k].shape for k in a)
                    else:
                        ok = a.shape == b.shape and list(a.columns) == list(b.columns)
                    if not ok:
                        return {"ok": False, "detail": "dataframe shape/columns differ"}
                    if ftype == "pkl_bigdf" and not isinstance(a, dict):
                        if not (a.dtypes.astype(str) == b.dtypes.astype(str)).all():
                            return {"ok": False, "detail": "pkl dtypes differ"}
            except Exception as e:
                return {"ok": False, "detail": f"{type(e).__name__}: {e}"}
        return {"ok": True, "detail": f"{ftype} reconstructed OK"}


def _track_graph_name_from_env_path(path):
    core = os.path.basename(path)[len("environment_"):-len(".pkl")]
    _nwb, rest = core.split("_trackgraph_", 1)
    return rest.rsplit("_", 1)[0]


def _build_environment_from_path(path):
    from non_local_detector.environment import Environment
    from spyglass.common import TrackGraph

    tg_query = TrackGraph() & {"track_graph_name": _track_graph_name_from_env_path(path)}
    env = Environment(
        track_graph=tg_query.get_networkx_track_graph(),
        edge_order=tg_query.fetch1("linear_edge_order"),
        edge_spacing=tg_query.fetch1("linear_edge_spacing"),
    )
    env.fit_place_grid()
    return env


def _reconstruct_environment(src_path, out_path):
    joblib.dump(_build_environment_from_path(src_path), out_path)


def enumerate_decode_files():
    from alison_decoding import ClusterlessResults

    rows = []
    fetched = ClusterlessResults().fetch(
        "nwb_file_name", "clusterless_results_path", "environment_path", as_dict=True
    )
    for r in fetched:
        rows.append(
            dict(nwb_file_name=r["nwb_file_name"], source_path=r["clusterless_results_path"], file_type="nc")
        )
        rows.append(
            dict(nwb_file_name=r["nwb_file_name"], source_path=r["environment_path"], file_type="pkl_env")
        )
    return rows


def enumerate_glob_files(directory, file_type, pattern="*", anchor_fn=anchor_for_path):
    rows = []
    for p in sorted(glob.glob(os.path.join(directory, pattern))):
        rows.append(dict(nwb_file_name=anchor_fn(p), source_path=p, file_type=file_type))
    return rows


_DIR_TO_FTYPE = {
    "big_df_pkls": "pkl_bigdf",
    "ex_events": "csv",
    "ex_events_july": "csv",
    "figs26": "npy",
    "behavior_csvs": "csv",
    "behavior_model_csvs": "csv",
}


def _ftype_for_path(path):
    d = os.path.basename(os.path.dirname(path).rstrip("/"))
    if d == "decodes":
        return "pkl_env" if os.path.basename(path).startswith("environment_") else "nc"
    return _DIR_TO_FTYPE.get(d)


def enumerate_from_pathlist(txt_path):
    rows, skipped = [], 0
    with open(txt_path) as fh:
        for line in fh:
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            ftype = _ftype_for_path(p)
            if ftype is None:
                skipped += 1
                continue
            rows.append(dict(nwb_file_name=anchor_for_path(p), source_path=p, file_type=ftype))
    if skipped:
        print(f"  ({skipped} paths skipped: out-of-scope directories, e.g. jld2)")
    return rows


def insert_files(rows, skip_duplicates=True):
    for r in rows:
        assert r["file_type"] in _FILE_TYPES, f"bad file_type {r['file_type']!r}"
        assert len(r["source_path"]) <= 255, f"source_path too long for PK: {r['source_path']}"
    AdditionalDataFilesSelection.insert(rows, skip_duplicates=skip_duplicates)
    print(f"Inserted {len(rows)} AdditionalDataFilesSelection rows.")
