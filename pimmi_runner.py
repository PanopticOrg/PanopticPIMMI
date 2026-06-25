import asyncio
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _import_pimmi():
    import multiprocessing
    # pimmi calls set_start_method('fork') at import; neutralize it to avoid clashes.
    orig = multiprocessing.set_start_method
    multiprocessing.set_start_method = lambda *a, **k: None
    try:
        import faiss
        import pimmi  # noqa: F401
    finally:
        multiprocessing.set_start_method = orig
    # Keep faiss single threaded so forking pimmi workers does not deadlock on the OpenMP pool.
    faiss.omp_set_num_threads(1)


def _configure(params):
    import pimmi
    from pimmi.cli.config import parameters as prm
    cfg = Path(pimmi.__file__).parent / "cli" / "config.yml"
    prm.set_config_as_attributes(prm.load_config_file(str(cfg)))
    prm.index_type = params["index_type"]
    prm.sift_nfeatures = params["sift_nfeatures"]
    prm.query_sift_knn = params["query_sift_knn"]
    return prm


def _run_paths(run_dir):
    run = Path(run_dir)
    return (run / "index.faiss", run / "index.meta", run / "query.csv", run / "clusters.csv")


def _install_sift_cache(sift_dir, sha1_by_path, prm):
    """Cache SIFT descriptors by image content so fill/query never recompute them across runs."""
    import numpy as np
    import pimmi.pimmi as pp

    cache = Path(sift_dir)
    cache.mkdir(parents=True, exist_ok=True)
    sig = hashlib.sha1(f"{prm.sift_nfeatures}|{prm.sift_nOctaveLayers}|{prm.sift_contrastThreshold}|"
                       f"{prm.sift_edgeThreshold}|{prm.sift_sigma}".encode()).hexdigest()[:8]
    original = pp.extract_sift

    def cached(file, image_id, sift, pack=-1):
        sha1 = sha1_by_path.get(os.path.realpath(file))
        cfile = (cache / f"{sha1}.{sig}.npz") if sha1 else None
        if cfile is not None and cfile.is_file():
            d = np.load(cfile)
            pts, desc, w, h = d["pts"], d["desc"], int(d["w"]), int(d["h"])
        else:
            ids, kp, desc, w, h = original(file, image_id, sift, pack)
            # Only descriptors are cacheable. Unreadable / zero-keypoint images give desc=None and
            # ids None or empty; pass them through (normalizing None to empty so len() never crashes).
            if desc is None or ids is None or len(ids) == 0:
                return (np.empty(0, dtype="int64") if ids is None else ids), kp, desc, w, h
            pts = np.array([k.pt for k in kp], dtype=np.float32)
            if cfile is not None:
                np.savez_compressed(cfile, pts=pts, desc=desc, w=w, h=h)
        if pp.grid_aspect_ratio:
            gx = np.floor(pts[:, 0]).astype(np.int64)
            gy = np.floor(pts[:, 1]).astype(np.int64)
        else:
            gx = np.floor(pp.grid_d * pts[:, 0] / w).astype(np.int64)
            gy = np.floor(pp.grid_d * pts[:, 1] / h).astype(np.int64)
        ids = (np.int64(image_id) << pp.grid_shift) + pp.grid_d * gy + gx
        return ids, None, desc, w, h

    pp.extract_sift = cached


# --- subprocess stages -------------------------------------------------------

def stage_fill(payload):
    from pimmi import create_index_mt, save_index
    from pimmi.toolbox import Sift

    params = payload["params"]
    prm = _configure(params)
    if params["reuse_cache"]:
        _install_sift_cache(payload["sift_dir"], payload["sha1_by_path"], prm)
    faiss_file, meta_file, _, _ = _run_paths(payload["run_dir"])
    sift = Sift(prm.sift_nfeatures, prm.sift_nOctaveLayers, prm.sift_contrastThreshold,
                prm.sift_edgeThreshold, prm.sift_sigma, prm.nb_threads)
    index = create_index_mt(prm.index_type, payload["image_paths"], payload["root"], sift, verbose=False)
    save_index(index, str(faiss_file), str(meta_file))


def stage_query_cluster(payload):
    import pimmi.pimmi_parameters as constants
    from pimmi import load_index, get_index_images, query_index_mt
    from pimmi.clusters import generate_clusters

    params = payload["params"]
    prm = _configure(params)
    if params["reuse_cache"]:
        _install_sift_cache(payload["sift_dir"], payload["sha1_by_path"], prm)
    faiss_file, meta_file, query_csv, clusters_csv = _run_paths(payload["run_dir"])

    index = load_index(str(faiss_file), str(meta_file))
    images = get_index_images(index, payload["root"])
    images.sort(key=lambda x: x[constants.dff_image_path])
    with open(query_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=constants.query_fieldnames,
                                extrasaction="ignore", quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        result = query_index_mt(index, images, payload["root"])
        if result:
            for ir in result.images:
                writer.writerow(ir.as_dict())

    with open(clusters_csv, "w", newline="") as f:
        try:
            generate_clusters(str(query_csv), str(meta_file), f, params["nb_match_ransac"], params["algo"])
        except UnboundLocalError:
            # pimmi raises this when the similarity graph is empty (no matches): leave only the header.
            pass


# --- parent side (no pimmi import) -------------------------------------------

def _run_stage(stage, payload, run_dir):
    # Blocking on purpose: run in a thread (see run_clustering) so we waitpid directly and do not
    # depend on the server event loop's child watcher, which can leave proc.wait() hanging forever.
    payload_path = run_dir / (stage + ".payload.json")
    payload_path.write_text(json.dumps(payload))
    log = run_dir / (stage + ".log")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with open(log, "wb") as lf:
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), stage, str(payload_path)],
                            stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True).returncode
    if rc != 0:
        raise RuntimeError(f"pimmi {stage} stage failed:\n" + log.read_text(errors="replace")[-4000:])


async def run_clustering(image_paths, sha1_by_path, root, data_dir, params):
    """Build index + query + cluster in a throwaway run dir; only the SIFT cache persists on disk."""
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    sift_dir = data / "sift_cache"
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=data))
    try:
        payload = {"image_paths": image_paths, "sha1_by_path": sha1_by_path, "root": root,
                   "sift_dir": str(sift_dir), "run_dir": str(run_dir), "params": params}
        await asyncio.to_thread(_run_stage, "fill", payload, run_dir)
        await asyncio.to_thread(_run_stage, "query_cluster", payload, run_dir)
        return _parse_clusters(run_dir / "clusters.csv")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _parse_clusters(clusters_csv):
    clusters = {}
    with open(clusters_csv, newline="") as f:
        for row in csv.DictReader(f):
            cluster = clusters.setdefault(row["cluster_id"], {"quality": float(row["quality"]), "ids": []})
            cluster["ids"].append(int(row["image_id"]))
    return clusters


if __name__ == "__main__":
    _stage = sys.argv[1]
    _payload = json.loads(Path(sys.argv[2]).read_text())
    _import_pimmi()
    if _stage == "fill":
        stage_fill(_payload)
    elif _stage == "query_cluster":
        stage_query_cluster(_payload)
    else:
        raise SystemExit(f"unknown stage {_stage}")
