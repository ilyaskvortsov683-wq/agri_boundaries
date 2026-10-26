# Running open Sentinel-2 field-boundary tools in Google Colab (late 2026)

Research note, checked 2026-10-26. Network limits during this research: the official docs sites (fieldsofthe.world, opengeos.org, samgeo.gishub.org, torchgeo.readthedocs.io, arxiv.org, huggingface.co, developers.google.com, research.google.com) could not be fetched from the sandbox. The facts below therefore come mainly from source files fetched directly from GitHub (READMEs, pyproject files, code, notebooks with saved outputs), PyPI JSON metadata, GitHub issue lists, and web-search snippets. Each claim names its source. Snippet-only sources are marked "(search snippet)". Nothing was executed in Colab during this research.

---

## Q1. How to run each tool in Colab today: install, notebooks, GPU, weights, I/O, known issues

### Takeaway
Three tools run in Colab today with official notebooks and pretrained Sentinel-2 weights: **ftw-tools** (FTW U-Net/PRUE), **Delineate Anything** (YOLO11 instance segmentation), and **geoai**. geoai can train Mask R-CNN on FTW or on your own labels, but ships no ready field model with a documented checkpoint. samgeo (SAM/SAM2/SAM3) runs zero-shot on any 8-bit 3-band image. torchgeo, smp and TerraTorch are libraries for fine-tuning. DECODE (MXNet), eo-learn field-delineation (archived) and cultionet (Python <3.11, no weights) do not run in Colab without real effort.

### Cited Findings

**A. ftw-tools / FTW CLI (Fields of The World)**
- **Where the code lives.** The FTW CLI is published as `ftw-tools` but still lives in the `ftw-baselines` repo "due to legacy reasons". A move to an `ftw-tools` repo is planned — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- **Versions on PyPI.** The latest *stable* release is 1.4.3 (2025-08-19). It requires Python `>=3.11,<3.13` and pins `pyarrow==16.1.0`, `dask[distributed]==2025.5.1` and `fiboa-cli==0.7.0`. Pre-releases 2.0.0b1 to 2.0.0b5 also exist. Extras are `all`, `delineate-anything` and `dev` — [PyPI ftw-tools JSON](https://pypi.org/project/ftw-tools/)
- **Current main branch.** `main` is version `2.0.0b5`. It requires Python `>=3.12,<3.15`, `torch>=2.2,<3`, `torchgeo>=0.10,<1`, `lightning>=2.6.6`, `odc-stac`, `planetary-computer` and `fsspec[http]>=2025.7,<2026`. The extra `delineate-anything` adds `ultralytics>=8.3.187,<9` — [ftw-baselines pyproject.toml](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/pyproject.toml)
- **Official install method.** The README only documents a uv dev install (`pip install uv; uv venv; uv sync --all-extras --dev`). It gives no plain-pip Colab recipe — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- **Official Colab notebook (verified with saved outputs).** "Agricultural Monitoring with FTW" (ICLR 2026 ML4RS tutorial): `https://colab.research.google.com/github/fieldsoftheworld/iclr2026-ml4rs-tutorial/blob/main/1.0-agriculture-monitoring-with-ftw.ipynb`. In Colab it runs `git clone …iclr2026-ml4rs-tutorial; PIP_CONSTRAINT=constraints.txt pip install .`. The package pins `ftw-tools[all]==2.0.0b4` and requires Python `>=3.12,<3.14`. The install log shows it *replaced* Colab's torch with `torch-2.7.1`, `torchgeo-0.8.1` and `ultralytics-8.4.12`. The constraints file contains only `setuptools<78` — [tutorial notebook](https://github.com/fieldsoftheworld/iclr2026-ml4rs-tutorial/blob/main/1.0-agriculture-monitoring-with-ftw.ipynb), [tutorial pyproject](https://github.com/fieldsoftheworld/iclr2026-ml4rs-tutorial)
- **Measured speed in that notebook (Colab GPU).** The notebook downloaded and stacked 2 scenes for a bbox in 14.1 s. `ftw inference run … --gpu 0 --model FTW_PRUE_EFNET_B3_CCBY` on a **2257×1683 px** input (patch 1024, padding 64) finished in **6.53 s**, and the 50.8 MB checkpoint downloaded automatically from GitHub releases. Polygonizing took 0.51 s. Colab warned "suggested max number of worker … is 2" while the default `--num_workers` is 4 — [tutorial notebook outputs](https://github.com/fieldsoftheworld/iclr2026-ml4rs-tutorial/blob/main/1.0-agriculture-monitoring-with-ftw.ipynb)
- **CLI commands.**
  - `ftw model list`
  - `ftw model show FTW_PRUE_EFNET_B5`
  - `ftw inference all --bbox=… --year=… --out=… --model=…` (scene selection from a crop calendar, download, inference, polygonize)
  - `ftw inference download --win_a <STAC id> --win_b <STAC id> --out x.tif --bbox …`
  - `ftw inference run x.tif --model … --out y.tif --gpu 0`
  - `ftw inference filter-by-lulc`
  - `ftw inference polygonize y.tif --simplify 20`
  - `ftw inference instance-segmentation-all` (Delineate Anything)
  - `ftw data download --countries=…`
  - `ftw model fit -c config.yaml`

  Source: [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- **Limits of `ftw inference all`.** It only accepts `--year` in the range **2015–2025**. `--gpu` defaults to **-1 (CPU)**, `--stac_host` is `mspc` (Planetary Computer, the default) or `earthsearch` (AWS, collection `sentinel-2-c1-l2a` by default), and `--cloud_cover_max` defaults to 20 — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- **Model registry.**
  - Default: `FTW_PRUE_EFNET_B5`.
  - Also `FTW_PRUE_EFNET_B3`/`B7` (v3, "Mixed Open Licenses"), `FTW_PRUE_EFNET_B3/B5/B7_CCBY` (v3.1, CC-BY-4.0) and legacy v1/v2 models (including `FTW_v2_3_Class_FULL_singleWindow`, the only single-window model).
  - `FTP_PRUE_EFNET_B3/B7` were trained on 3 m PlanetScope (normalization 10000, CC-BY-NC).
  - `DelineateAnything`, `DelineateAnything-S` and `DelineateAnythingV2` are AGPL-3.
  - Checkpoints download from GitHub releases or HF automatically.
  - Every Sentinel-2 FTW model uses `normalization = 3000.0` ("divisor applied to input imagery") and "requires two time windows, at the start and end of the growing season".

  Source: [model_registry.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/model_registry.py)
- **README model name does not match the registry.** The README examples use `--model=FTW_3_Class_FULL_multiWindow`, but the registry key is `FTW_v2_3_Class_FULL_multiWindow` — [README](https://github.com/fieldsoftheworld/ftw-baselines) vs [model_registry.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/model_registry.py)
- **Input contract.**
  - An 8-band GeoTIFF in the order `B04_t1, B03_t1, B02_t1, B08_t1, B04_t2, B03_t2, B02_t2, B08_t2` — [README](https://github.com/fieldsoftheworld/ftw-baselines)
  - `ftw inference download` builds it with `odc.stac.load(..., dtype="uint16", resampling="bilinear", bbox=bbox)` in the scenes' native CRS. For processing baseline ≥4 it applies `(data - 1000).clip(min=0)` to bring the data back to "processing version 3.0" scaling. It writes a deflate-compressed, tiled uint16 GeoTIFF — [download_img.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/download/download_img.py)
  - Inference divides by 3000 (`DEFAULT_NORMALIZATION = 3000.0`).
  - `patch_size` must be a multiple of 32 and no larger than the image. Padding defaults to `ceil(min(1024, patch)/16)`. A `resize_factor` (default 2) upsamples patches before the model.

    Source: [inference.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/inference.py)
- **Outputs.** `inference_data.tif`, `inference_output.tif` (classes: background / field / boundary, or softmax scores 0–255 with `--save_scores`) and `polygons.parquet`. Polygonize writes fiboa GeoParquet (default), `.gpkg`, `.fgb` or `.geojson`. `filter-by-lulc` uses `io-lulc-annual-v02` or `esa-worldcover` — [README](https://github.com/fieldsoftheworld/ftw-baselines), [settings.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/settings.py)
- **Known model error.** "FTW models are known to make some errors where land parcels that are not cropland (for example, pasture) are segmented as fields" — [README](https://github.com/fieldsoftheworld/ftw-baselines)
- **Issues on GitHub (114 in total).**
  - #261 "NaN/nodata pixels cause entire patches to be predicted as background" (closed Jun 2026)
  - #271 "Make vector postprocessing CRS-safe" (open)
  - #268 "Make FTW downloads resumable and checksum-safe" (open)
  - #262 validation against MapBiomas Argentina (open)
  - #238 `ImportError: cannot import name 'AugmentationSequential' from 'torchgeo.transforms'` (version skew)
  - #214 "Update Crop Calendar" (open)

  Source: [ftw-baselines issues](https://github.com/fieldsoftheworld/ftw-baselines/issues?q=is%3Aissue)
- **`filter-by-lulc` crash.** In the saved outputs of the official Colab tutorial, the `ftw inference filter-by-lulc` cell printed only "Error in sys.excepthook … Original exception was:". Later cells still ran — [tutorial notebook](https://github.com/fieldsoftheworld/iclr2026-ml4rs-tutorial/blob/main/1.0-agriculture-monitoring-with-ftw.ipynb)
- **Training countries.** The FTW training data covers 24–25 countries: Austria, Belgium, Brazil, Cambodia, Corsica, Croatia, Denmark, Estonia, Finland, France, Germany, India, Kenya, Latvia, Lithuania, Luxembourg, Netherlands, Portugal, Rwanda, Slovakia, Slovenia, South Africa, Spain, Sweden and Vietnam. **No Russia, Ukraine or Kazakhstan** — [torchgeo ftw.py](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/datasets/ftw.py)
- **Other FTW tools.** The organisation also has `ftw-inference-app` (a web app running FTW inference and polygonization), `ftw-qgis-plugin` (alpha) and `ftw-prue` (PRUE paper code) — [fieldsoftheworld org](https://github.com/fieldsoftheworld)

**B. FTW Global precomputed polygons (not a tool, but the lowest-effort option)**
- **What it is.** A global 10 m field-boundary dataset: "~1.63 billion for 2024 and 1.58 billion for 2025" polygons, made by running PRUE (U-Net, EfficientNet-B7, trained on the CC-BY subset of FTW) on "cloud-free Sentinel-2 mosaics". It is published as fiboa GeoParquet partitioned by country, plus PMTiles, a 500 m confidence COG, the Sentinel-2 planting/harvest median composites used as model input (COG plus a global EPSG:4326 Zarr), and softmax probabilities (Zarr). Licence CC-BY-4.0, anonymous HTTP/S3 read. The README includes a DuckDB example query by country partition — [ftw-data-catalog README](https://github.com/fieldsoftheworld/ftw-data-catalog/blob/main/catalog/README.md)
- **Russia partition.** It holds **13,621,854 polygons** covering 2024–2025, stored in EPSG:4326 at `…/results-by-admin-conf/admin:country_code=RU/Russia.parquet` with a `.pmtiles` alongside. "69% fall within the modeled-confidence layer's coverage" — [Russia STAC item](https://github.com/fieldsoftheworld/ftw-data-catalog/blob/main/catalog/predictions/vectors/alpha/results-by-admin-conf/admin%3Acountry_code%3DRU/Russia.json)
- **Columns and confidence.** Columns are `id, geometry, bbox, metrics:area, metrics:perimeter, determination:datetime, determination:method, admin:country_code, admin:subdivision_code, confidence`. The recommended reliability filter is `confidence >= 69`. A null confidence means the polygon lies outside the confidence layer, not that confidence is low. The confidence layer "fully cover[s] mainly the 24 FTW-labelled training countries" — [vectors llms.txt](https://github.com/fieldsoftheworld/ftw-data-catalog/blob/main/catalog/predictions/vectors/llms.txt)
- **Stated caveats.** Polygons are "remote-sensing field unit[s] … not a cadastral/legal parcel". Thresholds are at softmax 0.5. The confidence layer "is conservative outside the FTW training distribution" — [catalog README](https://github.com/fieldsoftheworld/ftw-data-catalog/blob/main/catalog/README.md)

**C. Delineate Anything (DA v1/v2)**
- **Repo, versions, training data.** [Lavreniuk/Delineate-Anything](https://github.com/Lavreniuk/Delineate-Anything) (AGPL-3.0, built on Ultralytics). DA v2 was accepted at ECCV 2026 and trained on FBIS-73M (61 countries, 0.25–10 m imagery).
- **Speed.** Mapping "all of Ukraine (603,000 km²) in 5.4 hours on a regular PC with 1 GPU NVIDIA RTX 5070 Ti 16 GB" — [DA README](https://github.com/Lavreniuk/Delineate-Anything)
- **Accuracy and weights.** On a 100-country benchmark, mAP@0.5 is 0.559 for v2 vs 0.275 for v1, and 0.612 for v2 in Europe. Weights are `DelineateAnythingv2.pt` / `DelineateAnything.pt` (125 MB) on HF `MykolaL/DelineateAnything` — [DA README](https://github.com/Lavreniuk/Delineate-Anything)
- **Official no-install demos.** A Colab demo (`https://colab.research.google.com/drive/10KSLwYDTgU-WhpqqG39yyvB6K8MdB0X9`) and an HF Space (`hugging-apps/delineate-anything-v2`). The browser app Autobounds is also mentioned — [DA README](https://github.com/Lavreniuk/Delineate-Anything) (WebFetch summary)
- **Local install.** `conda install -c conda-forge gdal`, an optional `pip install torch==2.6.0`, then `pip install -r requirements.txt`. requirements.txt pins `ultralytics==8.3.148, rasterio==1.4.3, shapely==2.1.1, opencv-python==4.11.0.86, numba==0.62.1, huggingface-hub==0.32.4`. Run with `python delineate.py -b batch_sample.yaml`. Output is a GeoPackage in `data/delineated/` — [README](https://github.com/Lavreniuk/Delineate-Anything), [requirements.txt](https://github.com/Lavreniuk/Delineate-Anything/blob/main/requirements.txt)
- **Input config (`conf_sample.yaml`).**
  - `data_loader.bands: [3, 2, 1]` (any 3 bands, 1-based)
  - `min`/`max: null`, meaning "automatic estimation" of normalization
  - `nodata_value: [0,0,0]`
  - `super_resolution: null`, meaning auto ×2 for ≥4 m imagery

  Source: [conf_sample.yaml](https://github.com/Lavreniuk/Delineate-Anything/blob/main/conf_sample.yaml)
- **Config guide notes.**
  - "RGB generally performs best".
  - GPU memory: "1 image ≈ <1 GB VRAM", so set `batch_size` close to VRAM in GB.
  - Tile-seam and region-seam merging are on by default.
  - The `giant_filter` (default on) removes polygons "chained across many tiles over low-contrast land". It "Requires a projected CRS".
  - An optional LULC mask can filter or clip polygons.

  Source: [delineation_config_guide.md](https://github.com/Lavreniuk/Delineate-Anything/blob/main/delineation_config_guide.md)
- **DA through ftw-tools.** The `DelineateAnything` wrapper takes RGB, applies `x / 3000.0` then `clip(0,1)`, uses `resize_factor=2` and `conf_threshold=0.05`, and requests a single time window — [ftw_tools/inference/models.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/models.py), [model_registry.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/model_registry.py)
- **Duplicate polygons via the FTW CLI.** "Inference uses patching with overlap which will result in duplicate polygons … this can become computationally slow" — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- **GitHub issues.**
  - #11: fine-tuning. The maintainer said "We used regular yolo11x model for instance segmentation" and pointed to the Ultralytics training docs. There is no dedicated fine-tune script.
  - #6: a user got `PROJ: cea: Invalid latitude` (EPSG:6933) errors and poor output. The maintainer said the territory "is mostly non agricultural, and very different from what we trained on Europe data" and suggested annotating and fine-tuning.
  - #9: "pip install torch==2.6.0 really needed?" (open).
  - #18: GDAL axis-order bug (fixed).
  - #16: sawtooth artifacts from float32 coordinates (fixed).

  Sources: [issue #11](https://github.com/Lavreniuk/Delineate-Anything/issues/11), [issue #6](https://github.com/Lavreniuk/Delineate-Anything/issues/6), [issue list](https://github.com/Lavreniuk/Delineate-Anything/issues?q=is%3Aissue)

**D. geoai (opengeos, Qiusheng Wu)**
- **Package.** `geoai-py` 0.43.1 (2026-09-21) requires **Python >=3.12** and **`torch>=2.10.0`**, plus torchgeo, timm, transformers, leafmap and planetary-computer. Extras include `terratorch` — [PyPI geoai-py JSON](https://pypi.org/project/geoai-py/)
- **Official notebook "Field Boundary Detection with Instance Segmentation"** (has a Colab badge). Steps:
  - `%pip install geoai-py`
  - `geoai.download_ftw(countries=["luxembourg"])`
  - `geoai.prepare_ftw(...)`, which "rescales Sentinel-2 reflectance (0–10,000) to uint8 (0–255)" and uses 256×256 chips
  - `geoai.train_instance_segmentation_model(..., num_channels=4, num_epochs=20, batch_size=4, instance_labels=True)` (Mask R-CNN, ResNet-50 FPN)
  - `geoai.instance_segmentation(..., window_size=256, overlap=128, vectorize=True)`
  - `clean_instance_mask`, `raster_to_vector`, `add_geometric_properties`

  The notebook uses window A only and suggests stacking both windows for 8 bands. It trains from scratch on FTW and **does not ship pretrained field weights** — [field_boundary_detection.ipynb](https://github.com/opengeos/geoai/blob/main/docs/examples/field_boundary_detection.ipynb)
- **Generic custom-label recipe.** `geoai.export_geotiff_tiles(in_raster, out_folder, in_class_data=<vector labels>, tile_size=512, stride=256)` → `train_instance_segmentation_model(num_channels=3, ...)` → `instance_segmentation(...)` → `orthogonalize`/vectorize — [train_instance_segmentation_model.ipynb](https://github.com/opengeos/geoai/blob/main/docs/examples/train_instance_segmentation_model.ipynb)
- **Sentinel-2 download from Planetary Computer.** `geoai.download_pc_stac_item(item_url, bands=[...], merge_bands=True, cell_size=10)` — [download_sentinel2.ipynb](https://github.com/opengeos/geoai/blob/main/docs/examples/download_sentinel2.ipynb)
- **Other exports.** geoai also exports `AgricultureFieldDelineator`, a Mask R-CNN wrapper for 12-band Sentinel-2 with optional NDVI and a default `model_path="field_boundary_detector.pth"`. Its docs do not show which public checkpoint to use. geoai has Prithvi helpers (`load_prithvi_model`, `prithvi_inference`) but they cover MAE reconstruction, not segmentation — [geoai/extract.py](https://github.com/opengeos/geoai/blob/main/geoai/extract.py), [geoai/__init__.py](https://github.com/opengeos/geoai/blob/main/geoai/__init__.py), [prithvi.ipynb](https://github.com/opengeos/geoai/blob/main/docs/examples/prithvi.ipynb)
- **GitHub issues.** #703: `display_ftw_samples` renders with an inverted Y-axis. #696: "Inference issue on sentinel-2 images that are not perfectly square". #187/#139/#123: installation problems — [geoai issues (colab search)](https://github.com/opengeos/geoai/issues?q=is%3Aissue%20colab)

**E. segment-geospatial (samgeo) / SAM2 / SAM3**
- **Package.** `segment-geospatial` 1.4.2 (2026-08-22) requires Python >=3.10. Extras are `samgeo2`, `samgeo3`, `hq`, `fast`, `text` and others — [PyPI segment-geospatial JSON](https://pypi.org/project/segment-geospatial/)
- **Official Colab notebook `sam2_automatic.ipynb`.** It asks for a GPU runtime and uses:
  - `%pip install -U segment-geospatial leafmap`
  - `SamGeo2(model_id="sam2-hiera-large", automatic=True)`
  - tunable `points_per_side=32, pred_iou_thresh=0.7, stability_score_thresh=0.92, crop_n_layers=1, min_mask_region_area=25, use_m2m=True`
  - `generate()` → `save_masks()` → `raster_to_vector("masks.tif","masks.gpkg")`

  Source: [sam2_automatic.ipynb](https://github.com/opengeos/segment-geospatial/blob/main/docs/examples/sam2_automatic.ipynb)
- **Tiled SAM3 for large GeoTIFFs.** `pip install "segment-geospatial[samgeo3]"`, then `SamGeo3(backend="meta").generate_masks_tiled(source, prompt=..., tile_size=1024, overlap=128)`, `raster_to_vector` and `smooth_vector` — [sam3_tiled_segmentation.ipynb](https://github.com/opengeos/segment-geospatial/blob/main/docs/examples/sam3_tiled_segmentation.ipynb)
- **No field-specific notebook.** The samgeo examples include none for agriculture or fields, and none for fine-tuning — [samgeo docs/examples](https://github.com/opengeos/segment-geospatial/tree/main/docs/examples)
- **Published evidence for SAM on seasonal Sentinel-2 composites.** Ha et al. (MethodsX 2026) applied SAM to seasonal Sentinel-2 RGB composites for the Canadian Prairies. They report IoU 0.86 vs manual segmentation over >32 M ha and released 656,082 polygons. Post-processing was done in ArcGIS Pro — [PMC12857264](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12857264/) (search snippet; full text not fetched), [Mendeley dataset](https://data.mendeley.com/datasets/2y568rt76w) (search snippet)
- **Tutorial.** A Towards Data Science tutorial covers SAM2 on a clear Sentinel-2 image in Colab with segment-geospatial — [TDS](https://towardsdatascience.com/field-boundary-detection-in-satellite-imagery-using-the-sam2-model-b556aa97bf7a/) (search snippet)

**F. torchgeo + segmentation-models-pytorch**
- **Package.** `torchgeo` 0.10.1 (2026-09-29) requires Python >=3.12, `segmentation-models-pytorch>=0.5` and `lightning`. `segmentation-models-pytorch` 0.6.0 was released 2026-10-26 — [PyPI torchgeo](https://pypi.org/project/torchgeo/), [PyPI smp](https://pypi.org/project/segmentation-models-pytorch/)
- **Pretrained FTW U-Nets in torchgeo.**
  - `Unet_Weights.SENTINEL2_2CLASS_FTW`, `SENTINEL2_3CLASS_FTW` (CC-BY-4.0)
  - `SENTINEL2_2CLASS_NC_FTW`, `SENTINEL2_3CLASS_NC_FTW` (non-commercial)
  - `SENTINEL2_FTW_PRUE_EFNETB3/B5/…` (non-commercial)
  - All have `in_chans: 8` and `encoder: efficientnet-b3` (PRUE B5/B7 use their own encoders), with transform `Normalize(mean=0, std=3000)`.
  - Built with `torchgeo.models.unet(weights=..., classes=...)`, which wraps `smp.create_model(arch='Unet')`.

  Source: [torchgeo/models/unet.py](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/unet.py)
- **FTW dataset class.** `torchgeo.datasets.FieldsOfTheWorld(countries=..., target='2-class'|'3-class'|'instance')` downloads from Source Cooperative. Its plotting code uses `win/3000` clipped to [0,1] — [torchgeo/datasets/ftw.py](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/datasets/ftw.py)
- **Pretrained Sentinel-2 backbones in torchgeo.**
  - ResNet-50: `SENTINEL2_ALL_MOCO` (SSL4EO-S12), `SENTINEL2_RGB_SECO`, `SENTINEL2_ALL_DINO/DECUR/SOFTCON`, `SENTINEL2_SI_MS_SATLAS`, `SENTINEL2_MI_MS_SATLAS`, `SENTINEL2_SI_RGB_SATLAS`, `SENTINEL2_ALL_SECO_ECO`.
  - `SENTINEL2_ALL_NDVI_SECO_ECO` takes 9 channels, the last being NDVI.
  - Swin-V2 versions of the Satlas weights.

  Sources: [resnet.py](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/resnet.py), [swin.py](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/swin.py)
- **FTW training config template.** `CustomSemanticSegmentationTask` with `model: unet`, `backbone: efficientnet-b3`, `in_channels: 8`, `num_classes: 3`, `loss: jaccard`, `FTWDataModule(batch_size=32, …)`, run with `ftw model fit -c` — [configs/example_config.yaml](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/configs/example_config.yaml)

**G. DECODE / FracTAL ResUNet**
- **Original code.** `waldnerf/decode` is the "Official mxnet implementation", with "models … built with the mxnet DL framework (version < 2.0)" and "We do not provide pre-trained weights" — [waldnerf/decode](https://github.com/waldnerf/decode). The same holds for `feevos/ceecnet` — [ceecnet](https://github.com/feevos/ceecnet)
- **PyTorch port.** `ftw-prue/decode/` holds a PyTorch FracTAL ResUNet (`fractal_resunet` package, `train_test.py`, `eval.py`, `visuals_and_inference.ipynb`) trained on FTW, with `temporal_option` set to stacked, windowA or windowB. No weight URL is given in that README — [ftw-prue decode README](https://github.com/fieldsoftheworld/ftw-prue/tree/main/decode)
- **ftw-prue repo.** Python `>=3.11,<3.14` via uv. It contains DECODE, SAM2 fine-tuning (`sam2_ftw/`), Mask2Former and GFM encoders (Clay, TerraMind, Prithvi, SatLAS, DOFA, CROMA …). It links the Colab tutorial in section A — [ftw-prue README](https://github.com/fieldsoftheworld/ftw-prue)
- **PRUE benchmark result.** PRUE "scored 76% IoU / 47% object-F1 on the Fields of the World benchmark, outperforming instance-segmentation and geospatial-foundation-model approaches" (out of 18 models) — [vectors llms.txt](https://github.com/fieldsoftheworld/ftw-data-catalog/blob/main/catalog/predictions/vectors/llms.txt)

**H. TerraTorch / Prithvi-EO-2.0**
- **Package.** `terratorch` 1.2.13 (2026-08-31) requires Python >=3.10 and `torchgeo<0.10,>=0.9.0`, `lightning>=2.6.0`, `segmentation-models-pytorch>=0.5.0`, `numpy>=2.2` — [PyPI terratorch](https://pypi.org/project/terratorch/)
- **Install options.** `pip install terratorch`, `conda install -c conda-forge terratorch`, or `pip install git+https://github.com/torchgeo/terratorch.git` (the repo moved to the torchgeo org). GDAL "can be hard to install" — [terratorch README](https://github.com/IBM/terratorch)
- **Prithvi example notebooks.** Prithvi-EO-2.0 ships fine-tuning notebooks (landslide, multi-temporal crop segmentation) and TerraTorch configs. The crop notebook:
  - installs with `pip install git+https://github.com/IBM/terratorch.git`
  - uses `BANDS = ["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]`, `NUM_FRAMES = 3`, backbone `prithvi_eo_v2_300_tl` with UperNet
  - sets `BATCH_SIZE=16`, `EPOCHS=50`, `precision="bf16-mixed"`

  Sources: [Prithvi-EO-2.0 README](https://github.com/NASA-IMPACT/Prithvi-EO-2.0), [example_multitemporalcrop.ipynb](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/examples/example_multitemporalcrop.ipynb)
- **Colab T4 note.** The HF model cards list the multitemporal crop notebook with ">>Try it on Colab<< (Choose T4 GPU runtime)" and say `terratorch>=1.1` is required — [HF Prithvi-EO-2.0-tiny-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL) (search snippet)
- **No field-boundary config.** No Prithvi or TerraTorch config or notebook targets field-boundary delineation. The listed downstream tasks are crop segmentation, land cover, floods, burns, landslides and others — [Prithvi-EO-2.0 README](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)

**I. eo-learn field-delineation and cultionet**
- **eo-learn field-delineation.** `sentinel-hub/field-delineation` is "ARCHIVED … not maintained". It downloads B-G-R-NIR through the Sentinel Hub Batch API with S3 buckets (paid) and uses a ResUNet-a model. A paid EuroDataCube service is offered instead — [field-delineation README](https://github.com/sentinel-hub/field-delineation). The latest `eo-learn` release is 1.5.7 (2024-09-27) — [PyPI eo-learn](https://pypi.org/project/eo-learn/)
- **cultionet.** It requires `python_requires >=3.9,<3.11` and `numpy<2`. Input is a satellite **time series** (bands or VIs, shaped batch × bands × time × H × W) on 1 km × 1 km grids that "must have different start and end years". The time series must be built outside the library. Install with `pip install git@github.com:jgrss/cultionet.git`. No pretrained weights are mentioned — [cultionet README](https://github.com/jgrss/cultionet), [setup.cfg](https://github.com/jgrss/cultionet/blob/main/setup.cfg)

**J. The Colab environment itself**
- **Python version.** Colab runtimes were upgraded to Python 3.12 on 2025-08-27 — [Colab release notes](https://developers.google.com/colab/release-notes) (search snippet)
- **Torch version.** The 2026.04 runtime snapshot lists Python 3.12.13, numpy 2.0.2 and PyTorch 2.10.0. It is not confirmed whether this is the current default — [Colab runtime FAQ](https://research.google.com/colaboratory/runtime-version-faq.html) (search snippet)

### Inferences
- **ftw-tools version to install.** Plain `pip install ftw-tools` resolves to stable **1.4.3** (Aug 2025), which predates the PRUE v3 registry entries. To get `FTW_PRUE_EFNET_*` and the multi-host download, pin the pre-release used by the official tutorial: `pip install "ftw-tools[all]==2.0.0b4"`, or `==2.0.0b5` / `pip install --pre`. Expect pip to reinstall torch, as the tutorial log shows torch 2.7.1, adding a few minutes. Restart the runtime after install.
- **Library conflict.** **ftw-tools 2.0.0b5 (torchgeo>=0.10) and terratorch 1.2.13 (torchgeo<0.10) cannot share one environment.** Use separate Colab sessions, or pin ftw-tools 2.0.0b4, which the tutorial ran with torchgeo 0.8.1. geoai (torch>=2.10) may also clash with the tutorial's torch 2.7.1 pin.
- **GPU needs.** A free T4 (16 GB) is more than enough for inference with every pretrained model here:
  - FTW U-Net/EffNet-B3 at patch 1024 with batch 2 is small.
  - DA quotes <1 GB VRAM per image.

  CPU inference works for FTW (the CLI default is CPU) but is slower. Colab's 2 vCPUs and ~12 GB RAM, rather than the GPU, will limit polygonization and DA post-processing. Use `--num_workers 2`.
- **Scaling estimate.** The tutorial's 2257×1683 px (~3.8×10⁶ px, ~380 km²) took 6.5 s on a Colab GPU. Samara oblast (~53,600 km² ≈ 5.4×10⁸ px at 10 m) would then need roughly 15–20 min of GPU time with B3, B5/B7 longer. The bottlenecks are imagery download/export, RAM for one giant mosaic, and polygonization. Process per tile (e.g. per MGRS tile or 0.5° blocks).
- **DA on a T4.** Scaling DA's Ukraine figure (≈31 km²/s on an RTX 5070 Ti) gives about 30 min for Samara on that card. A T4 is several times slower (no measured figure found), so plan for 1–3 h with Colab's limited CPUs for post-processing.
- **bf16 on T4.** The Prithvi crop notebook uses `precision="bf16-mixed"`. The T4 (Turing) has no native bf16, so switch to `"16-mixed"` or `32` on a free Colab T4.
- **Legacy tools.** Getting DECODE to work means using the PyTorch port in `ftw-prue/decode` and training it yourself, since the MXNet originals have no weights and MXNet <2.0 is unmaintained. cultionet needs a Python 3.10 side environment (e.g. condacolab or a uv venv) and has no weights. eo-learn field-delineation depends on paid Sentinel Hub. None of these three is a "hours not weeks" path.

### Gaps
- No source found that measures ftw-tools 2.0.0b5 or geoai 0.43.1 installing cleanly on the *current* (Oct 2026) Colab image. The only verified Colab run is the ICLR tutorial with 2.0.0b4.
- No public T4 benchmark for Delineate Anything v2 or for SAM2 automatic mask generation on Sentinel-2 tiles.
- The checkpoint and training data behind geoai's `AgricultureFieldDelineator` are undocumented in the files read.
- No weights URL found for the PyTorch DECODE port.
- The DA Colab demo is a Google Drive notebook that could not be fetched, so its cells (install steps, Sentinel-2 example) are unverified.
- The SAM3 model gating or licence on HF was not verified.

---

## Q2. Which tools give field polygons for a new AOI in Colab with the least effort?

### Takeaway
Ranked by effort for a Samara-type AOI:
1. **Download the FTW Global PRUE polygons for Russia (2024/2025).** Takes minutes, needs no GPU. Quality there is unvalidated.
2. **`ftw inference` with an FTW_PRUE model on a 2-window Sentinel-2 stack.** About an hour, mostly spent preparing input.
3. **Delineate Anything v2** on a single-date or composite RGB. About an hour, best in the standalone repo or the official Colab demo.
4. **SAM2 automatic masks on any 3-band 8-bit image**, including the user's NDVI composite. Hours, mostly tuning and cleaning.

Fine-tuning any of these on local labels takes about a day or more.

### Cited Findings
- **FTW Global coverage.** It has a Russia partition (13.6 M polygons, 2024–2025) with `admin:subdivision_code` and `confidence`, queryable over HTTP/S3 with DuckDB or as PMTiles, CC-BY-4.0 — [Russia item](https://github.com/fieldsoftheworld/ftw-data-catalog/blob/main/catalog/predictions/vectors/alpha/results-by-admin-conf/admin%3Acountry_code%3DRU/Russia.json), [catalog README](https://github.com/fieldsoftheworld/ftw-data-catalog/blob/main/catalog/README.md)
- **DuckDB query from the catalog README.** `INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;` then `read_parquet('s3://us-west-2.opendata.source.coop/ftw/global-data/predictions/vectors/alpha/results-by-admin-conf/admin:country_code=FR/France.parquet') WHERE confidence >= 80` — [catalog README](https://github.com/fieldsoftheworld/ftw-data-catalog/blob/main/catalog/README.md)
- **FTW CLI in 4 commands.** "In 4 lines of code, you obtained an FTW model, downloaded S2 data, ran model inference on that data, and polygonized the output" (download → run → polygonize). `ftw inference all` does scene selection too, but only for years 2015–2025 — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- **Verified Colab run.** The official Colab tutorial ran download (14 s), inference (6.5 s on a GPU for ~3.8 Mpx) and polygonize (0.5 s) end to end — [tutorial notebook](https://github.com/fieldsoftheworld/iclr2026-ml4rs-tutorial/blob/main/1.0-agriculture-monitoring-with-ftw.ipynb)
- **DA output.** DA goes straight to polygons (GeoPackage) with no raster polygonization step, and has built-in seam merging and a giant filter. Official Colab and HF Space demos exist — [DA README](https://github.com/Lavreniuk/Delineate-Anything), [config guide](https://github.com/Lavreniuk/Delineate-Anything/blob/main/delineation_config_guide.md)
- **DA outside its training domain.** The DA maintainer said that on a territory "very different from what we trained on Europe data" output was poor, and suggested annotating and fine-tuning — [DA issue #6](https://github.com/Lavreniuk/Delineate-Anything/issues/6)
- **SAM2 in samgeo.** Automatic mask generation runs in a few lines on any GeoTIFF and vectorizes with `raster_to_vector` — [sam2_automatic.ipynb](https://github.com/opengeos/segment-geospatial/blob/main/docs/examples/sam2_automatic.ipynb). SAM on seasonal Sentinel-2 composites reached IoU 0.86 in the Canadian Prairies — [PMC12857264](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12857264/) (search snippet)
- **geoai's field notebook trains a model rather than running one.** It downloads FTW, trains Mask R-CNN for 20 epochs, then infers. No pretrained field model is loaded — [field_boundary_detection.ipynb](https://github.com/opengeos/geoai/blob/main/docs/examples/field_boundary_detection.ipynb)
- **Pasture as fields.** FTW models confuse pasture with fields. `filter-by-lulc` with ESA WorldCover or IO LULC is the suggested fix — [README](https://github.com/fieldsoftheworld/ftw-baselines)

### Inferences
- **Fastest route for the user.** Pull FTW-Global `Russia.parquet` filtered to Samara: `admin:subdivision_code = 'SAM'` (ISO 3166-2 RU-SAM; the exact stored code was not checked) or a bbox filter. Then compare it against the user's manual digitization of 103 test fields with the existing `tests/gt_eval.py` metrics. This tells within an hour whether pretrained FTW quality is acceptable in the Samara steppe, before investing in any Colab pipeline. Samara is outside FTW's training countries and possibly outside the confidence layer, so expect null or low confidence.
- **Running FTW on 2026 data.** `ftw inference all` refuses `--year 2026`. Either use `ftw inference download --win_a/--win_b` with two Planetary Computer or Earth Search STAC IDs per MGRS tile, or export the 8-band stack from GEE (Q4) and call `ftw inference run` + `polygonize` directly.
- **DA as a check on the "giants" problem.** DA v2's `giant_filter` and seam merging target the same failure as the user's current pipeline (huge chained polygons, straight tile-edge cuts; see HANDOFF §7). That makes DA a useful independent cross-check. DA needs a projected CRS (UTM 39N for Samara) for that filter.
- **SAM2 is the only tool that can use the NDVI composite as is**, but it needs heavy tuning (`points_per_side`, IoU and stability thresholds, tiling) and post-filtering. It produces class-agnostic segments, so forest strips, villages and roads become polygons too.

### Gaps
- No published accuracy for FTW-Global, FTW PRUE or DA in Russia, Kazakhstan or the Volga steppe.
- Whether `admin:subdivision_code` uses "SAM" for Samara was not verified (the Parquet file was not read).
- Whether FTW-Global 2025 used mosaics that are cloud-free over Samara was not verified.

---

## Q3. What Sentinel-2 input must be prepared? (The user only has a 3-band NDVI composite: max, min, max−min, scaled 0–255)

### Takeaway
None of the pretrained FTW models can use the NDVI composite. They need **raw L2A reflectance DN, bands B4, B3, B2, B8 (10 m), from two dates or windows, as uint16 ordered `B04_t1,B03_t1,B02_t1,B08_t1,B04_t2,B03_t2,B02_t2,B08_t2`, on the PB<04 scale** (subtract 1000 for baseline ≥04, or use GEE `S2_SR_HARMONIZED`). The model divides by 3000 internally. Delineate Anything wants a 3-band RGB, single date or composite, and auto-normalizes. SAM2 and models trained from scratch (geoai Mask R-CNN, smp U-Net) can use the NDVI composite directly.

### Cited Findings
- **FTW band order.** `B04_t1, B03_t1, B02_t1, B08_t1, B04_t2, B03_t2, B02_t2, B08_t2`, with "two aligned Sentinel-2 scenes … in two seasons (e.g. planting and harvesting seasons)" — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- **FTW download output.** `odc.stac.load(dtype="uint16", resampling="bilinear")` in the scenes' native CRS (UTM). For processing baseline ≥4.0, "Rescaling data to processing version 3.0" via `(data - 1000).clip(min=0)`. Planetary Computer band names are `["B04","B03","B02","B08"]`; Earth Search names are `["red","green","blue","nir"]` — [download_img.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/download/download_img.py), [settings.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/settings.py)
- **FTW normalization.** Divide by 3000 (`DEFAULT_NORMALIZATION = 3000.0`), with a per-model `normalization` (10000 for the PlanetScope FTP models) — [inference.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/inference.py), [model_registry.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/model_registry.py). torchgeo FTW weights use `Normalize(mean=[0], std=[3000])` — [unet.py](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/unet.py)
- **Patch sizes.** FTW benchmark chips are 256×256 px at 10 m with 4 bands (R, G, B, NIR) per window — [geoai field_boundary_detection.ipynb](https://github.com/opengeos/geoai/blob/main/docs/examples/field_boundary_detection.ipynb). Inference patches default to 1024 px, must be a multiple of 32, and are upsampled ×2 — [inference.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/inference.py)
- **Choosing windows.** FTW chooses window A and window B "from the beginning and end of an agricultural season … the contrasting views … help identify which parcels are agricultural (as opposed to … grassland)". The tutorial picks the least-cloudy scene per window (<20% cloud) — [tutorial notebook](https://github.com/fieldsoftheworld/iclr2026-ml4rs-tutorial/blob/main/1.0-agriculture-monitoring-with-ftw.ipynb)
- **Composites vs single scenes.** FTW-Global used "Planting- and harvest-season median composites over ~5–10 quality-masked Sentinel-2 scenes" as model input. So median composites per window are an accepted input in place of single scenes — [catalog README](https://github.com/fieldsoftheworld/ftw-data-catalog/blob/main/catalog/README.md)
- **Ready-made composites.** The same composites are published for 2024 and 2025 as per-tile COGs and a global EPSG:4326 Zarr at 8.983119e-5° (~10 m) — [catalog README](https://github.com/fieldsoftheworld/ftw-data-catalog/blob/main/catalog/README.md)
- **Nodata handling.** "NaN/nodata pixels cause entire patches to be predicted as background" was a bug, closed Jun 2026. `run` has a `nan_fill_value` parameter defaulting to 0.0 — [ftw issues](https://github.com/fieldsoftheworld/ftw-baselines/issues?q=is%3Aissue), [inference.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/inference.py)
- **Delineate Anything input.** 3 bands, "RGB generally performs best". `min`/`max` normalization is auto-estimated when null. Super-resolution ×2 is applied automatically for ≥4 m data. The giant filter needs a projected CRS — [conf_sample.yaml](https://github.com/Lavreniuk/Delineate-Anything/blob/main/conf_sample.yaml), [config guide](https://github.com/Lavreniuk/Delineate-Anything/blob/main/delineation_config_guide.md). Through ftw-tools: RGB/3000 clipped to [0,1], single window — [models.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/models.py)
- **geoai Mask R-CNN input.** It expects uint8 GeoTIFF chips with any `num_channels` (3 for RGB, 4 for RGBN) — [field_boundary_detection.ipynb](https://github.com/opengeos/geoai/blob/main/docs/examples/field_boundary_detection.ipynb), [train_instance_segmentation_model.ipynb](https://github.com/opengeos/geoai/blob/main/docs/examples/train_instance_segmentation_model.ipynb)
- **Prithvi-EO-2.0 input.** 6 HLS bands (BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2) with multiple frames (3 in the crop example) — [example_multitemporalcrop.ipynb](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/examples/example_multitemporalcrop.ipynb)
- **NDVI-capable backbone.** A torchgeo backbone accepts NDVI as an extra channel: `ResNet50_Weights.SENTINEL2_ALL_NDVI_SECO_ECO` uses bands `B2,B3,B4,B5,B6,B7,B8,B8A,NDVI` — [resnet.py](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/resnet.py)
- **User's current input.** Per the project handoff, a single-season composite in EPSG:4326 with ~6×10 m pixels. Bands: 1 = max NDVI, 2 = min NDVI, 3 = max−min (all 0–255), 4 = optional validity mask — [project HANDOFF_field_pipeline.md](/home/user/agri_boundaries/HANDOFF_field_pipeline.md)

### Inferences
- **NDVI composite × pretrained models.** It is incompatible with every pretrained FTW/PRUE checkpoint: wrong channel count, wrong semantics, wrong scale. Feeding it would be meaningless. Delineate Anything *will* accept it, since it reads any 3 bands and auto-normalizes, but that is out of distribution (trained on true-colour RGB). It costs minutes, so it is a cheap experiment. Native RGB from the same dates is the proper input.
- **Get raw bands for FTW.** For FTW-class models the user should export from GEE (or Planetary Computer) **two median composites (spring and late summer), B4/B3/B2/B8, uint16 DN on the harmonized scale, 10 m, UTM 39N (EPSG:32639) for Samara**, and not reuse the EPSG:4326 6×10 m grid. All Sentinel-2 FTW models work on the native 10 m UTM grid. A non-square geographic pixel changes apparent field shapes and areas (FTW issue #271 notes CRS-unsafe vector post-processing).
- **Window choice for Samara.** Choose windows from the regional crop calendar, roughly early season (late April–May) and late season/post-harvest (mid-August–September). The user's composite already spans 05-04…09-20, so the same dates can define the two windows. FTW's crop calendar issue (#214, open) suggests the automatic calendar may be imperfect for Russia.
- **Reusing the NDVI composite elsewhere.** It remains useful as (a) extra channels for a self-trained model, (b) a post-filter (e.g. drop FTW polygons whose min-NDVI says perennial grass or fallow, mirroring FTW's pasture confusion), and (c) SAM2 input.

### Gaps
- Whether the free FTW-Global "features" Zarr covers Samara with usable (cloud-free) 2025 composites was not checked by reading the store.
- No study found that measures FTW/PRUE accuracy when the input is a median composite rather than single scenes, outside the FTW-Global paper itself (not read: arXiv 2605.11055 could not be fetched).
- How DA v2 behaves on false-colour or NDVI-derived inputs is undocumented.

---

## Q4. How to get that input through Google Earth Engine (the user has a GEE project)

### Takeaway
Use `COPERNICUS/S2_SR_HARMONIZED`, which has the baseline-04 offset removed and so matches FTW's PB-03 scale, masked with Cloud Score+ (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`, `cs_cdf >= 0.5–0.65`). Build two median composites, select B4, B3, B2, B8, stack in FTW order, cast to uint16 and export at 10 m in UTM. No official FTW-to-GEE example exists, so the code below is assembled from the documented dataset semantics plus the FTW input contract.

### Cited Findings
- **`COPERNICUS/S2_SR_HARMONIZED`.** "After 2022-01-25, Sentinel-2 scenes with PROCESSING_BASELINE '04.00' or above have their DN (value) range shifted by 1000. The HARMONIZED collection shifts data in newer scenes to be in the same range as in older scenes." — [GEE S2_SR_HARMONIZED](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) (search snippet)
- **FTW subtracts the same offset.** Its own downloader removes the same 1000 for PB ≥4 ("Rescaling data to processing version 3.0"), which shows the models expect harmonized, PB<04-style DN — [download_img.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/download/download_img.py)
- **Cloud Score+.** `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` has bands `cs` and `cs_cdf` (0–1). The official example does `S2_SR_HARMONIZED.filterBounds(...).filterDate(...).linkCollection(csPlus, ['cs_cdf']).map(img => img.updateMask(img.select('cs_cdf').gte(0.60))).median()`. "Values between 0.50 and 0.65 generally work well" — [GEE Cloud Score+](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED) (search snippet)
- **No official GEE route.** FTW's own download paths are Planetary Computer (default) and Earth Search (`ftw inference download`, `--stac_host mspc|earthsearch`). The README does not mention GEE — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- **Planetary Computer alternative in Python.** `geoai.download_pc_stac_item(item_url, bands=[...], merge_bands=True, cell_size=10)` — [download_sentinel2.ipynb](https://github.com/opengeos/geoai/blob/main/docs/examples/download_sentinel2.ipynb). Through FTW: `ftw inference download --win_a <id> --win_b <id> --bbox … --stac_host mspc|earthsearch` — [README](https://github.com/fieldsoftheworld/ftw-baselines)

### Inferences
Example GEE Python code. It was written for this note and **not executed**. Adjust the AOI, dates and project.

```python
# Colab cell 1: GEE → FTW 8-band input (uint16, PB-03 scale, 10 m, UTM 39N)
import ee
ee.Authenticate()
ee.Initialize(project="YOUR_GEE_PROJECT")

aoi = ee.Geometry.Rectangle([50.0, 53.0, 50.5, 53.3])   # small test block first; Samara oblast ≈ 47.9–52.6E, 51.8–54.7N
S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
CS = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

def window(start, end, thr=0.60):
    col = (S2.filterBounds(aoi).filterDate(start, end)
             .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
             .linkCollection(CS, ["cs_cdf"])
             .map(lambda im: im.updateMask(im.select("cs_cdf").gte(thr))))
    return col.select(["B4", "B3", "B2", "B8"]).median()

a = window("2026-04-20", "2026-05-31").rename(["B04_t1", "B03_t1", "B02_t1", "B08_t1"])
b = window("2026-08-10", "2026-09-25").rename(["B04_t2", "B03_t2", "B02_t2", "B08_t2"])
img = a.addBands(b).toUint16().clip(aoi)          # keep masked pixels masked (nodata) rather than 0 if possible

task = ee.batch.Export.image.toDrive(
    image=img, description="ftw_input_test", folder="ftw", fileNamePrefix="ftw_input_test",
    region=aoi, scale=10, crs="EPSG:32639", maxPixels=1e13, fileFormat="GeoTIFF")
task.start()
```

```bash
# Colab cell 2 (after the export finishes and Drive is mounted)
pip install -q "ftw-tools[all]==2.0.0b4"      # the version verified in the official Colab tutorial; restart the runtime after install
ftw model list
ftw inference run /content/drive/MyDrive/ftw/ftw_input_test.tif --model FTW_PRUE_EFNET_B5 \
    --out pred.tif --gpu 0 --num_workers 2 --overwrite
ftw inference polygonize pred.tif --out fields.gpkg --simplify 20 -f
```

Notes on the code:
- **Delineate Anything** needs only one window: export `B4,B3,B2` (one median, e.g. early or mid-season) as uint16 at 10 m in UTM. Either set `data_loader.bands: [1,2,3]` with auto min/max in DA's config, or run `ftw inference instance-segmentation-all` / `run-instance-segmentation` on it, which divides by 3000.
- **Size.** Samara oblast is ≈5.4×10⁸ px × 8 bands × 2 B ≈ 8–9 GB uncompressed. GEE will split the export into several GeoTIFFs. Run inference per file, then merge polygons and dissolve along seams, or export per ~0.5° block from the start.
- **Masked pixels.** `unmask(0)` hands the model zeros, which looked like the nodata issue #261. Keep coverage complete by widening the windows or lowering the `cs_cdf` threshold rather than zero-filling.
- **Skipping GEE.** For 2024/2025, the FTW-Global "features" Zarr already contains planting/harvest composites in model-ready form, so GEE is not needed for those years.

### Gaps
- The `cs_cdf` threshold, the window dates and the use of medians for a Samara-type phenology were not validated against FTW accuracy.
- The exact scaling of the FTW-Global feature composites (whether they are already PB-03 DN) was not verified because the Zarr was not opened.
- No official GEE example from FTW, DA or geoai was found. A web search for "FTW Google Earth Engine export" returned only Planetary Computer–based workflows.

---

## Q5. How much effort to fine-tune on a small custom label set (~100–300 polygons) in Colab?

### Takeaway
It is feasible on a free T4 in about a day of work. Most of the time goes into making complete-coverage label chips; training itself takes minutes to under an hour. The lowest-friction routes are (a) a **torchgeo `unet(weights=Unet_Weights.SENTINEL2_3CLASS_FTW)`** fine-tuned on 8-band GEE chips with 3-class masks, or (b) **geoai Mask R-CNN** trained on the user's 3-band composite (or RGBN) with `export_geotiff_tiles`. **Delineate Anything** can be fine-tuned only through the generic Ultralytics YOLO11x-seg training workflow, and **TerraTorch/Prithvi** needs the most setup.

### Cited Findings
- **geoai pipeline.** `export_geotiff_tiles(in_raster, out_folder, in_class_data=vector, tile_size=512, stride=256)` → `train_instance_segmentation_model(num_channels=3|4, num_classes=2, batch_size=4, num_epochs=10–20, val_split=0.2)` → `instance_segmentation(...)` (sliding window) → vectorize. Documented with a Colab badge — [train_instance_segmentation_model.ipynb](https://github.com/opengeos/geoai/blob/main/docs/examples/train_instance_segmentation_model.ipynb)
- **geoai on FTW.** The field notebook trains on the FTW Luxembourg subset (20 epochs; "increase to 50–100 for production") and suggests `confidence_threshold` and `min_area`/`max_hole_area` tuning — [field_boundary_detection.ipynb](https://github.com/opengeos/geoai/blob/main/docs/examples/field_boundary_detection.ipynb)
- **FTW training template.** `ftw model fit -c` with `CustomSemanticSegmentationTask` (unet, efficientnet-b3, in_channels 8, num_classes 3, jaccard loss) and `FTWDataModule`, which reads FTW-format country folders (`s2_images/window_a`, `window_b`, `label_masks/semantic_3class`, `chips_<country>.parquet`) — [example_config.yaml](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/configs/example_config.yaml), [torchgeo ftw.py](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/datasets/ftw.py)
- **Pretrained FTW U-Net weights in torchgeo** (8-channel, 2- or 3-class, efficientnet-b3, CC-BY or NC) can be loaded with `torchgeo.models.unet(weights=...)` and fine-tuned like any smp model — [unet.py](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/unet.py)
- **DA fine-tuning.** "If you want to finetune the model, please check [Ultralytics train docs] … We used regular yolo11x model for instance segmentation" — [DA issue #11](https://github.com/Lavreniuk/Delineate-Anything/issues/11)
- **ftw-prue extras.** The repo provides SAM2 fine-tuning (`sam2_ftw/`, needs `SAM2_CHECKPOINT_PATH`) and DECODE training scripts on FTW-format data — [ftw-prue README](https://github.com/fieldsoftheworld/ftw-prue)
- **TerraTorch/Prithvi.** The fine-tuning example uses Lightning with `BATCH_SIZE=16`, `EPOCHS=50`, the 300M-TL backbone and UperNet. The model card points to a Colab T4 runtime — [example_multitemporalcrop.ipynb](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/examples/example_multitemporalcrop.ipynb), [HF Prithvi tiny-TL card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL) (search snippet)
- **User's existing labels.** 103 manually digitized fields on a ~9,800 ha test scene, plus held-out scenes, with the evaluation harness in `tests/gt_eval.py` — [HANDOFF_field_pipeline.md](/home/user/agri_boundaries/HANDOFF_field_pipeline.md)

### Inferences
- **How much data 100–300 polygons gives.** Samara fields are large (median several tens of ha), so 100–300 polygons cover roughly 50–300 km². That is only about 10–50 chips of 256×256 px (2.56 km) at 10 m. This is a small set: do **not** train from scratch. Fine-tune a pretrained FTW U-Net (8-band) or DA, freeze the encoder at first, use heavy flips/rotations, and hold out a separate scene for validation.
- **Labels must be exhaustive inside each chip.** Every field in a chip needs a label, otherwise unlabeled fields are learned as background. Rasterize to FTW 3-class masks (background / interior / 1–2 px boundary) for semantic models, or to instance-ID masks for Mask R-CNN and YOLO. The user's GT contains holes and boundaries that are invisible on the composite (see HANDOFF §4), so clean those up first.
- **Effort estimate by route, on a free T4:**

  | Route | Data prep | Training | Total |
  |---|---|---|---|
  | (a) torchgeo FTW-UNet fine-tune (GEE export, rasterize, small Lightning/PyTorch loop) | 3–5 h | 10–30 min | ~1 day including inference and polygonization |
  | (b) geoai Mask R-CNN on the 3-band composite (`export_geotiff_tiles` works directly with the GPKG labels) | 1–2 h | 20–60 min | ~half a day; lowest code effort, but no pretraining on fields, so expect weaker results with this little data |
  | (c) DA YOLO11x-seg fine-tune (polygons to YOLO-seg txt format, Ultralytics `model.train(data=..., imgsz=512, batch≈8)`) | 2–4 h | 0.5–1.5 h (memory-heavy x-model) | ~1 day |
  | (d) Prithvi/TerraTorch | needs 6-band HLS-like multi-frame chips, a custom datamodule config, a separate environment (torchgeo version conflict) and bf16→fp16 changes | — | 2–3 days |

  - Route (a) is the fastest *good* path because the encoder and decoder already know Sentinel-2 field boundaries.
  - For route (c), DA uses its own post-processing pipeline (`delineate.py`). Loading a re-trained `.pt` into it is not documented and needs checking.
- **Colab session limits matter more than the GPU.** Keep data on Drive, checkpoint every epoch, and keep each training run under about 1 h.

### Gaps
- No published few-shot (≤300 polygons) fine-tuning result for FTW U-Net, DA or Mask R-CNN on Sentinel-2 field boundaries was found.
- No Colab T4 training-time measurement for any of these field models was found.
- Whether DA's `delineate.py` accepts a custom fine-tuned YOLO checkpoint (model names are hard-coded as "small", "large", "large_v2" in the config) was not verified.
