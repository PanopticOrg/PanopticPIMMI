import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from panoptic.core.plugin.plugin import APlugin
from panoptic.core.project.project import Project
from panoptic.models import ActionContext
from panoptic.models.results import ActionResult, Group, Notif, NotifType, Score
from panoptic.utils import group_by_sha1

from .pimmi_runner import run_clustering


class AlgoEnum(Enum):
    components = "components"
    louvain = "louvain"


class PluginParams(BaseModel):
    index_type: str = "IVF1024,Flat"
    sift_nfeatures: int = 1000
    query_sift_knn: int = 1000
    nb_match_ransac: int = 10
    algo: AlgoEnum = AlgoEnum.components
    reuse_cache: bool = True


class PanopticPimmi(APlugin):
    """
    Cluster images by visual reuse (copies, crops, edits) using PIMMI.
    Returns clusters as groups; use Panoptic's "save clusters" button to persist them.
    """

    def __init__(self, project: Project, plugin_path: str, name: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.params = PluginParams()
        self.add_action_easy(self.compute_pimmi_clusters, ['group'])

    async def compute_pimmi_clusters(self, context: ActionContext):
        instances = await self.project.get_instances(context.instance_ids)
        sha1_to_instances = group_by_sha1(instances)
        if not sha1_to_instances:
            return ActionResult(notifs=[Notif(NotifType.ERROR, name="NoData", message="No instance found")])

        # pimmi assigns image ids in this order, so index in the list == pimmi image_id.
        ordered = list(sha1_to_instances.items())
        sha1s_by_id = [sha1 for sha1, _ in ordered]
        image_paths = [insts[0].url for _, insts in ordered]
        root = Path(os.path.commonpath(image_paths))
        if root.is_file():
            root = root.parent

        sha1_by_path = {os.path.realpath(url): sha1 for sha1, url in zip(sha1s_by_id, image_paths)}

        params = self.params.dict()
        params["algo"] = self.params.algo.value

        clusters = await run_clustering(image_paths, sha1_by_path, str(root), str(self.data_path), params)
        if not clusters:
            return ActionResult(notifs=[Notif(NotifType.INFO, name="NoCluster", message="PIMMI found no cluster")])

        selection = set(sha1s_by_id)
        groups = []
        for i, cluster in enumerate(sorted(clusters.values(), key=lambda c: c["quality"], reverse=True)):
            sha1s = [sha1s_by_id[image_id] for image_id in cluster["ids"] if 0 <= image_id < len(sha1s_by_id)]
            sha1s = [s for s in dict.fromkeys(sha1s) if s in selection]
            if not sha1s:
                continue
            score = Score(value=cluster["quality"], min=0, max=1, max_is_best=True,
                          description="PIMMI cluster match quality")
            groups.append(Group(sha1s=sha1s, name=f"Pimmi cluster {i}", score=score))

        notif = Notif(NotifType.INFO, name="PimmiClusters", message=f"{len(groups)} clusters found")
        return ActionResult(groups=groups, notifs=[notif])
