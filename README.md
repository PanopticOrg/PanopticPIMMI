# PanopticPIMMI

Panoptic plugin that groups images reusing the same visual content (exact copies, crops, resized, recolored or slightly edited versions) with [PIMMI](https://medialab.sciencespo.fr/outils/pimmi/), the image mining tool of the Sciences Po médialab ([code](https://github.com/nrv/pimmi)).

Unlike the similarity plugin, which groups images that *look alike*, PIMMI matches local SIFT points between images and checks that they are geometrically consistent, so it only links images that share an actual piece of picture.

Requires Panoptic 1.0 or later. Install it from `https://github.com/PanopticOrg/PanopticPIMMI`; the `pimmi` package is installed with it.

## Usage

Open "Create clusters" from the menu of "All" or of a group, and choose **compute_pimmi_clusters**. It runs on the images of that group, or on the whole project. Each cluster gets a score between 0 and 1: the higher it is, the more its images are matched with each other. Like any clustering, it can be saved into a property with the floppy disk icon.

Only images stored in local folders are processed: images from IIIF sources are skipped, with a warning.

## Params

Set in the plugin params:

| Param | Default | Meaning |
|---|---|---|
| `index_type` | `IVF1024,Flat` | [faiss index](https://github.com/facebookresearch/faiss/wiki/The-index-factory). With too few points to train an IVF index (up to a few hundred images), an exact index is used instead |
| `sift_nfeatures` | 1000 | max number of SIFT points per image |
| `query_sift_knn` | 1000 | nearest neighbours retrieved per SIFT point |
| `nb_match_ransac` | 10 | min number of consistent matches to link two images: raise it to get fewer, stricter links |
| `algo` | `components` | `components` keeps every connected group of images, `louvain` splits large groups into communities |
| `reuse_cache` | on | keep the SIFT points of each image in `plugin_data`, so later runs skip their extraction |
| `nb_threads` | 0 | worker processes, 0 uses every CPU core |

## How it works

PIMMI runs in separate Python processes, never in the Panoptic server: it extracts the SIFT points of each image, indexes them with faiss, queries every image against the index, and builds the clusters from the graph of matched images. The SIFT points are cached by image content (sha1) and by SIFT params; the index itself is rebuilt on every run, in a temporary folder.
