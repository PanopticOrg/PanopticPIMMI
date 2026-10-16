import os
from enum import Enum

from pydantic import BaseModel

from panoptic.core.plugin.plugin import APlugin
from panoptic.models.action_models import ActionContext, ActionResult, Group, Notif, NotifType, Score

from .pimmi_runner import run_clustering


class AlgoEnum(Enum):
    components = "components"
    louvain = "louvain"


class PluginParams(BaseModel):
    """
    PIMMI parameters, see https://github.com/nrv/pimmi
    @index_type: faiss index type; falls back to an exact index when there are too few points to train it
    @sift_nfeatures: max number of SIFT points kept per image
    @query_sift_knn: nearest neighbours retrieved for each SIFT point
    @nb_match_ransac: min number of geometrically consistent matches to link two images
    @algo: community detection: connected components or louvain
    @reuse_cache: keep SIFT points between runs (plugin_data folder)
    @nb_threads: worker processes, 0 = all CPU cores
    """
    index_type: str = "IVF1024,Flat"
    sift_nfeatures: int = 1000
    query_sift_knn: int = 1000
    nb_match_ransac: int = 10
    algo: AlgoEnum = AlgoEnum.components
    reuse_cache: bool = True
    nb_threads: int = 0


class PanopticPimmi(APlugin):
    """
    Cluster images by visual reuse (copies, crops, edits) using PIMMI.
    Returns clusters as groups; use Panoptic's "save clusters" button to persist them.
    """

    def __init__(self, name: str, project, plugin_path: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.params = PluginParams()
        self.add_action_easy(self.compute_pimmi_clusters, ['group'])

    def update_params(self, params):
        # The base class copies without validation and saves the Enum as is, which is not JSON serializable.
        data = params if isinstance(params, dict) else params.dict()
        self.params = PluginParams(**{**self.params.dict(), **data})
        self.project.save_params(self.params.model_dump(mode="json"))
        return self.params

    def compute_pimmi_clusters(self, context: ActionContext) -> ActionResult:
        """Group images that reuse the same visual content (copies, crops, edits) with PIMMI."""
        if context.instance_ids:
            instances = self.project.get_instances(id=context.instance_ids)
        else:
            instances = self.project.get_instances()
        paths = self._local_paths({i.file_id for i in instances if i.file_id is not None})

        # Pimmi assigns image ids in this order, so the index in the list is the pimmi image_id.
        sha1s_by_id = list(paths)
        image_paths = list(paths.values())
        notifs = []
        skipped = len({i.sha1 for i in instances}) - len(sha1s_by_id)
        if skipped:
            notifs.append(Notif(NotifType.WARNING, name="Skipped",
                                message=f"{skipped} images without a local file were ignored"))
        if len(image_paths) < 2:
            notifs.append(Notif(NotifType.ERROR, name="NoData", message="PIMMI needs at least 2 local images"))
            return ActionResult(notifs=notifs)

        try:
            root = os.path.commonpath(image_paths)
        except ValueError:
            notifs.append(Notif(NotifType.ERROR, name="Paths", message="Images must be on the same drive"))
            return ActionResult(notifs=notifs)
        if os.path.isfile(root):
            root = os.path.dirname(root)

        params = self.params.dict()
        params["algo"] = self.params.algo.value
        params["nb_threads"] = self.params.nb_threads or os.cpu_count() or 4

        sha1_by_path = {os.path.realpath(p): s for s, p in paths.items()}
        clusters = run_clustering(image_paths, sha1_by_path, root, str(self.data_path), params)

        groups = []
        for cluster in sorted(clusters.values(), key=lambda c: c["quality"], reverse=True):
            sha1s = list(dict.fromkeys(sha1s_by_id[i] for i in cluster["ids"] if 0 <= i < len(sha1s_by_id)))
            if len(sha1s) < 2:
                continue
            score = Score(value=cluster["quality"], min=0, max=1, max_is_best=True,
                          description="PIMMI cluster match quality")
            groups.append(Group(sha1s=sha1s, name=f"Pimmi cluster {len(groups)}", score=score))

        if not groups:
            notifs.append(Notif(NotifType.INFO, name="NoCluster", message="PIMMI found no cluster"))
            return ActionResult(notifs=notifs)
        notifs.append(Notif(NotifType.INFO, name="PimmiClusters", message=f"{len(groups)} clusters found"))
        return ActionResult(groups=groups, notifs=notifs)

    def _local_paths(self, file_ids: set[int]) -> dict[str, str]:
        """sha1 -> path of one existing local file showing it, resolved with a few batched queries."""
        files = self.project.get_files(id=list(file_ids))
        folders = {f.id: f for f in self.project.get_folders(id=list({f.folder_id for f in files}))}
        local_sources = {s.id for s in self.project.get_file_sources() if s.dtype == 'local'}
        paths = {}
        for file in sorted(files, key=lambda f: f.id):
            folder = folders.get(file.folder_id)
            if file.sha1 in paths or not file.name or folder is None or folder.path is None \
                    or folder.source_id not in local_sources:
                continue
            path = os.path.join(folder.path, file.name)
            if os.path.isfile(path):
                paths[file.sha1] = path
        return paths
