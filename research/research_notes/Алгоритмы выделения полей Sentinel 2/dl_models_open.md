# Open-source deep-learning models for field boundary delineation from Sentinel-2 (10 m): code, weights, training data, accuracy, licences (state as of 2026-10-26)

Research notes. Method note: arxiv.org, huggingface.co and most publisher sites could not be fetched directly from this sandbox. Primary sources actually read in full were: GitHub READMEs, source files and LICENSE files (via raw.githubusercontent.com), the FTW global-map paper PDF (Microsoft Research copy), and PyPI metadata. Everything else comes from web-search extracts of the primary pages (arXiv abstracts, HF cards, journal pages) and is marked "(search extract)" where the detail could not be checked against the full text.

## Key question 1: Which models publish pretrained weights usable out-of-the-box, and what exact input do they need (bands, time steps, resolution, normalisation)? (full catalogue)

### Takeaway
Only two families have maintained, documented, out-of-the-box Sentinel-2 weights plus a working end-to-end CLI: (1) **Fields of The World (FTW) / PRUE** U-Nets (8-channel input: B4,B3,B2,B8 × 2 seasonal dates, divided by 3000; 3 classes: background / field interior / boundary; MIT code; CC-BY-4.0 or non-commercial weights), and (2) **Delineate Anything v1/v2** (YOLO11-seg instance segmentation on a single 3-band RGB image, resolution-agnostic 0.25–10 m, auto min/max stretch; AGPL-3.0 code and weights). Both run through `ftw-tools` (`ftw inference all` and `ftw inference instance-segmentation-all`), through TorchGeo, and through the `agribound` wrapper. Everything else is code-only (DECODE/FracTAL, cultionet, Sentinel Hub ResUNet-a, most SAM work), has weights for a different sensor or region (Rwanda/PlanetScope winner, U-TAE on PASTIS, FTP on PlanetScope), or is closed (Google ALU, Planet/Sinergise products).

### Cited Findings

#### A. Fields of The World (FTW) baselines: U-Net, Sentinel-2 bi-temporal (the main open S2 option)
- Paper: Kerner, Chaudhari, Ghosh, Robinson, Ahmad, Choi, Jacobs, Holmes, Mohr, Dodhia et al., "Fields of The World: A Machine Learning Benchmark Dataset for Global Agricultural Field Boundary Segmentation", AAAI 2025, vol. 39, pp. 28151–28159 (arXiv 2409.16252). — [FTW global-map paper reference list](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf); [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/35034)
- Dataset: 70,462 samples, about 1.6M field polygons, 24 countries on 4 continents, with instance and semantic masks paired with multi-date Sentinel-2 imagery. — [arXiv 2409.16252](https://arxiv.org/abs/2409.16252) (search extract); [FTW ecosystem page](https://fieldsofthe.world/ecosystem.html) (search extract)
- The code lists 25 regions: austria, belgium, brazil, cambodia, corsica, croatia, denmark, estonia, finland, france, germany, india, kenya, latvia, lithuania, luxembourg, netherlands, portugal, rwanda, slovakia, slovenia, south_africa, spain, sweden, vietnam. 21 of them have "presence and absence" (full) labels. Brazil, India, Kenya and Rwanda are presence-only. — [ftw-baselines settings.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/settings.py)
- Default paper model: a U-Net with an EfficientNet-b3 backbone on concatenated 4-channel RGB-NIR images from Window A and Window B. U-Nets beat DeepLabv3+, and EfficientNet backbones did best. — [arXiv 2409.16252](https://arxiv.org/abs/2409.16252) (search extract)
- Exact input: the bands of two aligned Sentinel-2 L2A scenes in the order B04_t1, B03_t1, B02_t1, B08_t1, B04_t2, B03_t2, B02_t2, B08_t2, i.e. planting/start-of-season and harvest/end-of-season windows. — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- Normalisation: the model registry's default `normalization` is a 3000.0 divisor applied to input imagery for all FTW models, and 10000.0 for the PlanetScope FTP models. TorchGeo uses the same `Normalize(mean=0, std=3000)`. — [model_registry.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/model_registry.py); [TorchGeo unet.py](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/unet.py)
- **Conflict:** the global-map paper says each patch was "normalized ... to surface reflectance units (division by 10,000 with BOA offset correction)" for PRUE global inference. — [FTW global map paper, Methods](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf). The CLI registry still says 3000 for the FTW models. Check the checkpoint config before use.
- Released checkpoints, from the registry as read on 2026-10-26 (`ftw model list`):
  - v1 (legacy): `2_Class_CCBY`, `2_Class_FULL`, `3_Class_CCBY`, `3_Class_FULL` .ckpt files.
  - v2 (legacy): `3_Class_FULL_FTW_Pretrained_v2.ckpt` (multi-window) and `..._singleWindow_v2.ckpt`. The single-window model needs only one date and "will perform a bit less well than the latest 2 window models".
  - v3 PRUE: `prue_efnet3/5/7_checkpoint.ckpt` (EfficientNet-B3/B5/B7). **B5 is the CLI default.**
  - v3.1: `prue_efnetb3/b5/b7_ccby_checkpoint.ckpt`.
  - Two PlanetScope FTP models (below), plus three Delineate Anything models.
  — [model_registry.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/model_registry.py)
- Download URLs: GitHub releases `https://github.com/fieldsoftheworld/ftw-baselines/releases/download/{v1,v2,v3,v3.1}/...`. TorchGeo mirrors them on HF (`hf.co/torchgeo/ftw/...` for the v1 effb3 models; `hf.co/isaaccorley/ftw-prue` and `hf.co/isaaccorley/ftw-prue-ccby` for PRUE B3/B5/B7). The TorchGeo enum names are `Unet_Weights.SENTINEL2_2CLASS_FTW`, `SENTINEL2_3CLASS_FTW`, `SENTINEL2_2CLASS_NC_FTW`, `SENTINEL2_3CLASS_NC_FTW`, `SENTINEL2_FTW_PRUE_EFNETB3/5/7` and `SENTINEL2_FTW_PRUE_CCBY_EFNETB3/5/7`, each with in_chans=8. — [TorchGeo unet.py](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/unet.py); [model_registry.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/model_registry.py)
- End-to-end CLI: `ftw inference all --bbox=... --year=... --model=...` selects S2 scenes from crop-calendar harvest dates, downloads them from Microsoft Planetary Computer or EarthSearch (EarthSearch default `sentinel-2-c1-l2a`), runs inference, and polygonises. Defaults:
  - `--year` must be 2015–2025; `--cloud_cover_max` 20; `--buffer_days` 14; `--resize_factor` 2; patch 1024; padding 64; `--gpu -1` (CPU).
  - Outputs `inference_data.tif`, `inference_output.tif` and `polygons.parquet` (fiboa GeoParquet).
  - Optional `filter-by-lulc` uses io-lulc-annual-v02 or esa-worldcover, because "FTW models are known to make some errors where land parcels that are not cropland (for example, pasture) are segmented as fields".
  - `polygonize` defaults: simplify 15 m, min_size 500 m², plus optional merge_adjacent, morphological opening/closing and Zhang-Suen thinning.
  — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- Crop calendars for automatic window selection come from source.coop (`ftw/ftw-inference-input/global-crop-calendar/`, summer and winter SOS/EOS COGs). — [settings.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/settings.py)
- Packaging: the code is MIT and published on PyPI as `ftw-tools`. The latest upload is pre-release 2.0.0b5 (2026-03-24); the latest stable is 1.4.3. — [PyPI ftw-tools](https://pypi.org/project/ftw-tools/); [ftw-baselines LICENSE](https://github.com/fieldsoftheworld/ftw-baselines)
- Post-processing in the FTW pipeline is argmax of the 3 classes, then connected components / polygonisation of the field-interior class. There is no watershed in the CLI. — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines); [FTW global map paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)

#### B. PRUE (CVPR 2026): improved FTW recipe; FTW v3 / v3.1 checkpoints
- Paper: Muhawenayo, Robinson, Khanal, Fang, Corley, Wollam, Gao, Strnad, Avery, Estes, Tárano, Jacobs, Kerner, "PRUE: A Practical Recipe for Field Boundary Segmentation at Scale", CVPR 2026, pp. 6484–6495, arXiv 2603.27101. Code: github.com/fieldsoftheworld/ftw-prue (MIT). — [ftw-prue README](https://github.com/fieldsoftheworld/ftw-prue)
- Recipe:
  - U-Net with an EfficientNet-B7 encoder, Adam, log-cosh Dice loss.
  - Class weights [0.05, 0.20, 0.75] for background, interior and boundary.
  - Augmentations: channel shuffling (input-order invariance), random brightness, random resize (scale robustness).
  - Designed to reduce tiling artefacts in large-scale deployment.
  — [FTW global map paper, Methods](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- Benchmark scope: 21 countries. Compared standard decoders (U-Net, DeepLabV3+, FCN, UPerNet, SegFormer, DPT), GFM encoders (Clay, TerraFM, DINOv3, TerraMind, CROMA, DeCUR, DOFA, Prithvi, SatLAS, SoftCon, Galileo) and custom models (DECODE/FracTAL ResUNet, SAM/SAM2, Mask2Former, DelineateAnything). Precomputed FTW GFM embeddings are on source.coop (`mvrl/ftw-inference-gfm`). A Colab tutorial exists (`fieldsoftheworld/iclr2026-ml4rs-tutorial`, notebook 1.0-agriculture-monitoring-with-ftw.ipynb). — [ftw-prue README](https://github.com/fieldsoftheworld/ftw-prue)
- The authors "release all models and model-derived field boundary datasets for five countries". — [PRUE abstract / MSR page](https://www.microsoft.com/en-us/research/publication/prue-a-practical-recipe-for-field-boundary-segmentation-at-scale/) (search extract). FTW's site lists country maps for Japan, Mexico, Rwanda, South Africa and Switzerland (search extract, [Taylor Geospatial FTW page](https://taylorgeospatial.org/innovation/fields-of-the-world/)).

#### C. FTW global field-boundary map 2024 + 2025 (Microsoft AI for Good, ASU, WashU, Taylor Geospatial et al.; released April 2026)
- Robinson, Muhawenayo, Khanal, Fang, Corley, Tárano, Estes, Marcus, Jacobs, Kerner, Becker-Reshef, Lavista Ferres, "The first global agricultural field boundary map at 10 m resolution", April 23, 2026 (arXiv 2605.11055).
  - Output: 3.17 billion polygons (1.62 B in 2024, 1.55 B in 2025) across 241 countries and territories.
  - Model: PRUE U-Net with EfficientNet-B7, trained **only on the CC-BY subset of FTW**.
  - Inputs: per-pixel median RGBN (B02, B03, B04, B08) mosaics for planting and harvest windows per MGRS tile, from the WorldCereal crop calendar; scenes with <20% cloud plus an SCL mask.
  - Inference: 256×256 patches with 25% overlap and Gaussian-weighted stitching on 256 A10G GPUs, then connected components of the field-interior class.
  — [FTW global map paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- Released under CC-BY at https://source.coop/ftw/global-data as three products: the confidence-thresholded default map, the full unfiltered map, and a 500 m confidence raster. The default threshold is conf ≥ 0.4, which keeps 864 M fields in 2024 and 844 M in 2025. — [FTW global map paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- Also distributed as Zarr, GeoParquet and PMTiles through an Earthmover marketplace listing (search extract). — [Earthmover listing](https://app.earthmover.io/marketplace/69e935646f290ce55a5b5c7a)

#### D. Fields of the Planet (FTP): PlanetScope 3 m companion, not Sentinel-2
- arXiv 2607.04449 (July 2026). 133,168 co-registered PlanetScope patch-window targets across the 24 FTW countries.
  - At matched recipes, 3 m imagery raises panoptic quality from 21.0 (S2 10 m) to 35.5.
  - Quality on fields under 0.5 ha rises from 5.8 to 15.7.
  - Matched-boundary error falls from 18.6 m to 7.4 m.
  - Weights: `ftp-b3` and `ftp-b7` (U-Net, 8 channels, 3 classes), CC-BY-NC-4.0.
  — [FTP arXiv](https://arxiv.org/abs/2607.04449) (search extract); [HF fieldsoftheworld/ftp-b7](https://huggingface.co/fieldsoftheworld/ftp-b7) (search extract); [model_registry.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/model_registry.py)

#### E. Delineate Anything (DelAny) v1, Flow, v2: resolution-agnostic instance segmentation (YOLO11-seg)
- Authors: Lavreniuk, Kussul, Shelestov, Yailymov, Salii, Kuzin, Szantoi (v1); v2 adds C. J. L.-X. Wang.
  - v1: "Delineate Anything: Resolution-Agnostic Field Boundary Delineation on Satellite Imagery", ECAI 2025, arXiv 2504.02534.
  - Flow: arXiv 2511.13417 (Nov 2025).
  - v2: "Delineate Anything v2: A Global Foundation Model for Field Delineation", accepted to ECCV 2026 (Workshops per the bibtex), arXiv 2607.19069.
  - Code: github.com/Lavreniuk/Delineate-Anything, **AGPL-3.0**, built on Ultralytics.
  — [Delineate-Anything README](https://github.com/Lavreniuk/Delineate-Anything)
- Architecture: YOLOv11 instance segmentation (v1 "large" = yolo11x, 125 MB; "small" = yolo11n, 17.6 MB). The output is polygons directly, with no intermediate raster. — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines); [model_registry.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/model_registry.py)
- Weights:
  - HF `MykolaL/DelineateAnything`: files `DelineateAnything.pt` and `DelineateAnythingv2.pt`.
  - TorchGeo mirrors: `hf.co/torchgeo/delineate-anything` (`delineate_anything_rgb_yolo11x-88ede029.pt`) and `hf.co/torchgeo/delineate-anything-s` (`delineate_anything_s_rgb_yolo11n-b879d643.pt`).
  - The FTW registry lists the licence as AGPL-3.
  — [Delineate-Anything README](https://github.com/Lavreniuk/Delineate-Anything); [model_registry.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/model_registry.py)
- Input:
  - Single date, 3 bands (RGB). The config takes `bands: [3, 2, 1]` (GDAL 1-based) and states that "RGB generally performs best".
  - `min`/`max` normalisation is estimated automatically if null.
  - FP16 inference (`use_half: true`, CPU forced to FP32).
  - Default `minimal_confidence` 0.15; tile overlap `tile_step` 0.5.
  - The pipeline super-resolves S2 internally ("256 px after super-resolution for Sentinel-2"; tiles every 1280 m for S2).
  - Default filters: minimum field area 2500 m², plus a "giant filter" made for 2.5–10 m imagery.
  - An optional LULC mask can remove or clip polygons.
  — [conf_sample.yaml](https://github.com/Lavreniuk/Delineate-Anything/blob/main/conf_sample.yaml); [delineation_config_guide.md](https://github.com/Lavreniuk/Delineate-Anything/blob/main/delineation_config_guide.md)
- In ftw-tools: `ftw inference instance-segmentation-all <S2 item> --model {DelineateAnything|DelineateAnything-S|DelineateAnythingV2}` (default V2), resize_factor 2, patch 256. Overlapping-patch duplicates are merged by IoU and containment thresholds. — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- Post-processing (Flow/v2): structured post-processing, merging, vectorisation and simplification, plus tile-seam and region-seam merging (on by default). — [DelAny Flow arXiv](https://arxiv.org/abs/2511.13417) (search extract); [delineation_config_guide.md](https://github.com/Lavreniuk/Delineate-Anything/blob/main/delineation_config_guide.md)
- Colab demo, HF Space demo (`hugging-apps/delineate-anything-v2`), ESA APEx map explorer (explorer.delineate-anything.apex.esa.int), and a browser tool (autobounds.com). Integrated into TorchGeo on 2025-07-07. — [Delineate-Anything README](https://github.com/Lavreniuk/Delineate-Anything)

#### F. DECODE / FracTAL ResUNet / ResUNet-a (CSIRO: Waldner, Diakogiannis)
- ResUNet-a (Diakogiannis, Waldner, Caccetta, Wu, ISPRS JPRS 2020) introduced multi-task outputs: extent, boundary, distance transform (and colour reconstruction). — [arXiv 1904.00592](https://arxiv.org/abs/1904.00592)
- "Deep learning on edge" (Waldner & Diakogiannis, RSE 2020, DOI 10.1016/j.rse.2020.111741):
  - ResUNet-a on Sentinel-2 monthly composites in South Africa predicts field, boundary and distance, then post-processes into closed polygons.
  - Reported to generalise to single-date imagery, to coarser resolutions and other sensors, and to other dates and locations. One search extract lists secondary sites in Argentina, Australia, Canada, Russia and Ukraine; this was not verified against the full text.
  — [arXiv 1910.12023](https://arxiv.org/abs/1910.12023) (search extract)
- DECODE ("Detect, Consolidate, Delineate", Waldner et al., Remote Sensing 13(11):2197, 2021):
  - FracTAL ResUNet plus hierarchical watershed segmentation for instances.
  - Australian grains zone: pixel accuracy 0.87, field-based accuracy up to 0.88.
  - A model trained in South Africa "transferred well to unseen Australian landscapes".
  - Code: MXNet (<2.0) in github.com/waldnerf/decode, "We do not provide pre-trained weights". Licence: CSIRO BSD/MIT, with the condition that modifications made available are under BSD-3.
  — [DECODE README](https://github.com/waldnerf/decode)
- A PyTorch DECODE re-implementation is included in ftw-prue (`decode/`) and was benchmarked there. — [ftw-prue README](https://github.com/fieldsoftheworld/ftw-prue)
- Applications:
  - CSIRO ePaddocks (national Australian product), cited in the tfcl paper. — [tfcl README](https://github.com/feevos/tfcl)
  - Mozambique national mapping from SPOT using DECODE (Rufin et al., ERL 2026). — [FTW global map paper refs](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
  - India transfer (Wang, Waldner & Lobell, Remote Sensing 14(22):5738, 2022): FracTAL ResUNet trained in France and applied in India. Transfer plus weak supervision cut the labels needed by 5–10×, and 10,000 manual Indian field boundaries were released on Zenodo (10.5281/zenodo.7315090). No code or weights repository was found. — [MDPI](https://www.mdpi.com/2072-4292/14/22/5738) (search extract)

#### G. PTAViT3D / PTAViT3D-CA ("Tackling fluffy clouds", Diakogiannis et al., CSIRO; RSE 2026)
- A 3D Vision Transformer on S2 and/or S1 time series that tolerates cloud-contaminated inputs. The CA variant fuses S1 and S2 with cross-attention.
  - Validated on ePaddocks (Australia), FTW, PASTIS and AI4SmallFarms.
  - Repo: github.com/feevos/tfcl, CSIRO Open Source Software Licence v1.0 (MIT/BSD-style with the BSD-3 condition). Demo forward and training notebooks; Docker image `fdiakogiannis/trchprosthesis_requirements:24.07-py3`.
  - The paper abstract says "Our code and models are publicly available", but the README shows no explicit weight download link.
  — [tfcl README](https://github.com/feevos/tfcl); [RSE article](https://www.sciencedirect.com/science/article/pii/S0034425726004657); [arXiv 2409.13568](https://arxiv.org/abs/2409.13568)
- The arXiv v3 (Sep 2026) also fine-tunes Delineate Anything on PASTIS. It reports that S2 gives a stronger boundary signal than S1. — [arXiv 2409.13568](https://arxiv.org/abs/2409.13568) (search extract)

#### H. AI4Boundaries (JRC: d'Andrimont et al. 2023) and models trained on it
- Dataset (ESSD 15:317–329, 2023):
  - S2 10 m monthly composites (7,831 samples of 256×256 px) and 1 m orthophotos.
  - Labels from 2019 GSAA in Austria, Catalonia, France, Luxembourg, the Netherlands, Slovenia and Sweden; 2.5 M parcels covering 47,105 km² in the released samples.
  - `pip install git+https://github.com/waldnerf/ai4boundaries` (MIT) downloads the data.
  — [ai4boundaries README](https://github.com/waldnerf/ai4boundaries)
- No official JRC baseline weights were found. Search did not surface the official AI4B baseline table either.
- fkroeber/ai4b (master's thesis; the README has no LICENSE file) trained two models on an enhanced AI4Boundaries: an ML-driven OBIA model (hierarchical watershed + Bayesian optimisation + random forest) and an enhanced Mask R-CNN (attention, oriented boxes, mask scoring) on multitemporal S2. "The final configurations and weights for the trained models can be downloaded" from a OneDrive link. — [fkroeber/ai4b README](https://github.com/fkroeber/ai4b)
- SAM fine-tuned on AI4Boundaries: Scribano, Govi, Bertellini, Parisi, Franchini, Bertogna, "Segment Anything for Satellite Imagery: A Strong Baseline and a Regional Dataset for Automatic Field Delineation", ICIAP 2025, arXiv 2506.16318. Trained on AI4B RGB S2 and tested on the new ERAgriSeg (Emilia-Romagna) dataset. Only the dataset repo (github.com/cscribano/ERAS-dataset) was found; no confirmed model code or weights. — [arXiv 2506.16318](https://arxiv.org/abs/2506.16318) (search extract)

#### I. cultionet (J. Graesser)
- A U-Net variant (UNet3+-style deep supervision, Psi-Net multi-stream outputs, ResUNet-a residual blocks with dilated neighbourhood attention, transformer time-series embeddings, Tanimoto loss) on S2 time series. Targets are crop/non-crop plus a boundary distance transform (`bdist`).
- Version 2.0.0b. Apache-2.0. Training uses user-supplied grid and polygon GeoPackages (1 km × 1 km grids tested).
- **No published pretrained weights** in the README; users train their own (`cultionet train`, checkpoint at `ckpt/last.ckpt`).
— [cultionet README](https://github.com/jgrss/cultionet)

#### J. Sentinel Hub / Sinergise "field-delineation" (NIVA, eo-learn), plus Planet EU product
- Repo sentinel-hub/field-delineation: MIT, **archived**.
  - ResUNet-a (eo-flow), trained k-fold on GSAA of Lithuania and Castilla y León.
  - Input: S2 B-G-R-NIR multitemporal; extent and boundary probabilities are merged temporally, then vectorised.
  - Requires paid Sentinel Hub Batch API plus AWS S3.
  - No weights published; a paid service exists on EuroDataCube.
  — [field-delineation README](https://github.com/sentinel-hub/field-delineation)
- Planet/Sinergise published an EU field-boundary vector dataset from S2 May–June 2022. It is on source.coop/planet/eu-field-boundaries (parquet 14.6 GB, PMTiles) and Zenodo (Oct 2024). The search extract did not state a licence. — [source.coop planet/eu-field-boundaries](https://source.coop/planet/eu-field-boundaries) (search extract); [Planet Pulse](https://www.planet.com/pulse/generating-51-million-field-boundaries-across-europe/) (search extract)

#### K. Google ALU (Agricultural Landscape Understanding)
- A multi-class panoptic segmentation model served through the Agricultural Understanding API (`lookupLandscape`). It covers fields, trees and water bodies, with up to 15 years of history refreshed every 6–12 months. Initially India-focused; on 2026-08-26 Google DeepMind announced expansion to several African countries.
- **No public model weights found**; API/hosted outputs only.
— [Google developers ALU page](https://developers.google.com/agricultural-understanding/landscape-understanding) (search extract); [DeepMind Agricultural Understanding](https://deepmind.google/science/agricultural-understanding/) (search extract)

#### L. NASA Harvest / Radiant Earth Rwanda field boundary challenge (Zindi, 2023)
- Imagery: PlanetScope/NICFI monthly basemaps (not Sentinel-2) in Rwanda.
- Gold solution: "Spatio-Temporal Attention-based Unet", a modified Unet++ with a `regnetv_040` encoder and temporal attention pooling. It takes 6 fixed months (2021_03, 04, 08, 10, 11, 12). Docker image `radiantearth/model_nasa_rwanda_field_boundary_competition_gold:1`. Training needed an A100 80GB.
- Licence conflict: the repo LICENSE file is CC-BY-NC-4.0, while `docs/index.md` says CC-BY-4.0.
— [gold repo README](https://github.com/radiantearth/model_nasa_rwanda_field_boundary_competition_gold); [Zindi](https://zindi.africa/competitions/nasa-harvest-field-boundary-detection-challenge) (search extract)

#### M. U-TAE + PaPs (Garnot & Landrieu, ICCV 2021): panoptic parcels from S2 time series
- Trained on PASTIS (2,433 S2 time-series patches, metropolitan France). Panoptic PQ 40.4; semantic mIoU 63.1.
- Weights on Zenodo (record 5172301 for panoptic, 5172293 for semantic). MIT licence.
- `torch_scatter` is needed for the panoptic module.
— [utae-paps README](https://github.com/VSainteuf/utae-paps)

#### N. SAM / SAM2 / segment-geospatial and geospatial foundation models
- **SAM (zero-shot) on seasonal S2 composites, Canadian Prairies** (Univ. of Saskatchewan, MethodsX 2026, doi 10.1016/j.mex.2026.103794): a GEE → SAM → ArcGIS Pro workflow over 32 Mha of cultivated land, IoU 0.86 vs manual segmentation. The output dataset of 656,082 polygons is on Mendeley Data (2y568rt76w). No code repository was found. — [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12857264/) (search extract); [Mendeley dataset](https://data.mendeley.com/datasets/2y568rt76w) (search extract)
- **segment-geospatial (samgeo) / geoai** (Qiusheng Wu, MIT): generic SAM/SAM2/SAM3 and Mask R-CNN tooling. agribound reports that GeoAI Mask R-CNN has "no published field weights; needs fine-tuning or `checkpoint_path`". — [agribound README](https://github.com/montimaj/agribound); [geoai README](https://github.com/opengeos/geoai)
- **DelAny v1 paper vs SAM:** on the FBIS-22M test set SAM scored 0.339 / 0.197 and SAM2 0.382 / 0.235 (mAP@0.5 / mAP@0.5:0.95), against 0.720 / 0.477 for DelAny. — [DelAny arXiv 2504.02534](https://arxiv.org/abs/2504.02534) (search extract)
- **GFMs on FTW (PRUE benchmark):** Clay, TerraFM, DINOv3, TerraMind, CROMA, DeCUR, DOFA, Prithvi, SatLAS, SoftCon and Galileo were tested with decoders. PRUE beat the best GFM (Clay) by 9 points IoU and 11 points object F1. The authors attribute this to coarse embeddings struggling with localisation. — [PRUE MSR page](https://www.microsoft.com/en-us/research/publication/prue-a-practical-recipe-for-field-boundary-segmentation-at-scale/) (search extract); [ftw-prue README](https://github.com/fieldsoftheworld/ftw-prue)
- No public field-boundary-fine-tuned checkpoints were found for Prithvi, Clay, TerraMind, SatlasPretrain, DOFA or AnySat. The TerraTorch/HF search turned up base models and other tasks only. — [TerraTorch](https://github.com/torchgeo/terratorch) (search extract)
- **AgriFM** (Li, Liang et al., RSE 2026; Video-Swin-based multi-source MODIS/Landsat/S2 foundation model): field-boundary delineation is listed as a downstream task. Pretrained backbone weights are on OneDrive/GLASS, Apache-2.0. Whether a fine-tuned boundary head is released was not confirmed. — [AgriFM README](https://github.com/flyakon/AgriFM)
- **agribound** (montimaj, v1.0.1, Apache-2.0, Python ≥3.12, Zenodo 10.5281/zenodo.19229665):
  - One pipeline: GEE composite → optional fine-tune → delineate → optional SAM2 refinement → LULC crop filter → export.
  - Engines: Delineate-Anything (default `large_v2`), FTW (`FTW_PRUE_EFNET_B5`), GeoAI Mask R-CNN, DINOv3 (SAT-493M) + DPT, Prithvi-EO-2.0-300M-TL (TerraTorch), embedding clustering (Google Satellite Embedding / AlphaEarth, TESSERA), and ensembles.
  - FTW (needs `lightning<2.6`) and Prithvi (needs `lightning>=2.6`) require two separate environments.
  - Version 1.0.0 fixed 0.1.x defects, including FTW season windows that were copies of the annual composite and a silent Delineate-Anything fallback with swapped red/blue channels.
  — [agribound README](https://github.com/montimaj/agribound)

#### O. Other items (papers or datasets without confirmed open weights)
- NASA Harvest Ukraine boundaries (Sadeh et al., Scientific Data 12:833, 2025):
  - Not a DL model: time-series edge detection (NDVI + PCA) on Planet 3 m bi-weekly composites in GEE, 14 composites from September to August, segmented per oblast.
  - More than 5 M fields for 2021–2023; median IoU 0.77 against 12,331 GNSS combine-harvester boundaries.
  - Data on the NASA Harvest Portal and Zenodo; regional files are CC-BY per the portal.
  — [Nature Sci Data](https://www.nature.com/articles/s41597-025-05190-7) (search extract); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12095754/) (search extract)
- "Early-season delineation of agricultural fields using a fully convolutional multi-task network" (Science of Remote Sensing 2025): ResUNet-a d7 on a single S2 Level-3A early-summer composite. Code or weights not checked. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666017225000628) (search extract)
- Lower Saxony comparison of Mask R-CNN, U-Net, FracTAL ResUNet and optimised multiresolution segmentation on S2 (PFG 2023). — [Springer](https://link.springer.com/article/10.1007/s41064-023-00247-x) (search extract)
- AAFC "Automatically Detected Field Boundaries in Canada, 2023": an open government dataset. Its method and weights were not checked. — [geo.ca record](https://app.geo.ca/en-ca/map-browser/record/b3dba669-6611-4cb1-b4be-6c317b9908ad) (search extract)

### Inferences
- For Colab (free T4 or CPU), the lowest-friction reproducible options are:
  - `pip install ftw-tools` (2.0.0b5 pre-release, or 1.4.3 stable), then `ftw inference all` with a PRUE checkpoint. Patches can be reduced; the default is CPU.
  - The official Delineate Anything Colab notebook or `ftw inference instance-segmentation-all`.
  - These are single small U-Nets or YOLO models, so a 16 GB T4 should suffice; FP16 is supported by DelAny. This is an inference from model sizes (125 MB for DelAny-large) and the stated <1 GB VRAM per image tile in the DelAny guide, not a tested benchmark.
- The FTW CLI accepts years 2015–2025 only. Mapping 2026 imagery needs either a newer ftw-tools release or manual stacking of the 8-band input (`ftw inference download` / `run` with STAC items).
- Normalisation must match the checkpoint. Using /3000 vs /10000 incorrectly, or ignoring the S2 processing-baseline ≥04.00 BOA offset (−1000 DN since 2022), will shift inputs. The EarthSearch `sentinel-2-c1-l2a` collection (FTW default) is harmonised. Harmonisation is general S2 knowledge and was not verified in these sources.
- DelAny needs only one cloud-free RGB date, so it is easier to feed in cloudy Volga-region springs. FTW/PRUE needs two seasonal dates; the single-window v2 model is the fallback.

### Gaps
- Could not fetch HF model cards (MykolaL/DelineateAnything, torchgeo/ftw, isaaccorley/ftw-prue) directly. The licence and file names come from the FTW registry and TorchGeo source, not the cards themselves.
- No explicit weight-download URLs were found for PTAViT3D (tfcl) despite the abstract's claim. No code was found for the ERAgriSeg SAM baseline or the Canadian SAM workflow.
- Exact per-model PRUE table (IoU and object F1 for each of the 18 models) not retrieved.

## Key question 2: What data were they trained on (countries/climate/landscape)? Which include Eastern Europe / steppe / large mechanised fields similar to Russia (Ukraine, Kazakhstan, Baltic, Finland, Poland, etc.)?

### Takeaway
No open model was trained on Russian or Kazakh fields, and the FTW authors state explicitly that "Russia and Central Asia" are absent from FTW training. The closest training coverage is:
- **FTW/PRUE** (Baltic states Estonia, Latvia and Lithuania; Finland; Slovakia; Croatia; Austria; plus Western Europe). No Ukraine, Poland or Russia.
- **Delineate Anything** (Ukrainian authors; the Flow and v2 papers produced a complete 2024 Ukraine map from Sentinel-2, which is the best landscape analogue to the chernozem steppe of Samara, Tatarstan and Donbas). Whether Ukraine is inside FBIS-22M/73M training data was not confirmed.
- **ResUNet-a "Deep learning on edge"**: South Africa-trained, reportedly tested on Russia/Ukraine sites. No weights.

### Cited Findings
- FTW geographic scope: 25 regions (Europe-heavy, plus Brazil, Cambodia, India, Kenya, Rwanda, South Africa and Vietnam). Eastern-European/Baltic members are Estonia, Latvia, Lithuania, Finland, Slovakia, Croatia and Slovenia. Poland, Ukraine, Russia, Belarus and Kazakhstan are absent. — [settings.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/settings.py)
- The authors' own limitations section: "Geographic coverage is heavily European (17 of 24 countries) ... North America, China, Russia and Central Asia, Australia, the Middle East, and most of North Africa are entirely absent from training, as are paddy rice outside Cambodia and Vietnam, semi-arid and arid irrigated agriculture, terraced highland smallholder systems, and boreal cropping." — [FTW global map paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- FTW training scope: **annual crops only**, explicitly excluding permanent grassland, pasture, orchards, vineyards, fallow and forestry. Pasture, perennial and fallow areas are therefore missed or turn into false positives. — [FTW global map paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- The CC-BY subset used for the global PRUE model provides 1.6 million field polygons from 24 countries, per the global paper (that sentence describes FTW overall). — [FTW global map paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- Delineate Anything training data:
  - FBIS-22M has 672,909 image patches (0.25–10 m) with 22,926,427 field instance masks. Sources include Sentinel-2, Planet, Maxar, Pleiades and others.
  - The FBIS-22M test splits "primarily covered only Europe".
  — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines); [Delineate-Anything README](https://github.com/Lavreniuk/Delineate-Anything); [DelAny arXiv 2504.02534](https://arxiv.org/abs/2504.02534) (search extract)
- FBIS-73M (v2): 73 M instances, 61 countries, 0.25–10 m, curated with a "resolution-specific curation pipeline that solves the parcel-versus-field mismatch". The HF dataset reports 1,478,096 patches and 74,015,889 masks, with train, test and 100-country zero-shot splits. Training data is long-tailed according to open cadastral availability. — [Delineate-Anything README](https://github.com/Lavreniuk/Delineate-Anything); [HF FBIS-73M](https://huggingface.co/datasets/MykolaL/FBIS-73M) (search extract)
- FBIS-73M also covers some evaluation sites that third parties treat as test sites: per agribound, it covers the Madera square and 87% of the Úbeda square, so DelAny v2 scores there "are not zero-shot". — [agribound README](https://github.com/montimaj/agribound)
- Ukraine applications by the DelAny team:
  - Flow: complete 2024 Ukraine map (603,000 km²) from Sentinel-2 in under 6 hours on one workstation; 3.75 M fields at 5 m and 5.15 M at 2.5 m, vs 2.66 M (Sinergise) and 1.69 M (NASA Harvest).
  - v2: Ukraine in 5.4 h on one RTX 5070 Ti 16 GB.
  — [DelAny Flow arXiv](https://arxiv.org/abs/2511.13417) (search extract); [Delineate-Anything README](https://github.com/Lavreniuk/Delineate-Anything)
- The DelAny "giant filter" notes that on 2.5 m input "over very large fields (for example Ukraine) it may also remove some real fields". This is a directly relevant warning for large mechanised fields. — [delineation_config_guide.md](https://github.com/Lavreniuk/Delineate-Anything/blob/main/delineation_config_guide.md)
- Sentinel Hub ResUNet-a was trained on Lithuania and Castilla y León GSAA. — [field-delineation README](https://github.com/sentinel-hub/field-delineation)
- AI4Boundaries covers Austria, Catalonia, France, Luxembourg, the Netherlands, Slovenia and Sweden (2019 GSAA). — [ai4boundaries README](https://github.com/waldnerf/ai4boundaries)
- DECODE / FracTAL was trained in the Australian grains zone and South Africa; both are large mechanised dryland grain systems. — [DECODE README](https://github.com/waldnerf/decode)
- "Deep learning on edge" (South Africa training) is described as generalising to other locations; one search extract names Russia and Ukraine secondary sites (unverified). — [arXiv 1910.12023](https://arxiv.org/abs/1910.12023) (search extract)
- No DL field-boundary study or dataset specific to Kazakhstan or Russia was found in English or Russian searches. Results returned crop classification and yield studies for Kazakhstan only. — [search: MDPI Agronomy Kazakhstan crop classification](https://www.mdpi.com/2073-4395/15/9/2040) (search extract)
- The Ukraine reference data available openly is the NASA Harvest 2021–2023 boundaries (Planet 3 m, edge-detection method, CC-BY regional files). It could serve as training labels for steppe-type fields. — [Nature Sci Data](https://www.nature.com/articles/s41597-025-05190-7) (search extract); [Harvest Portal](https://data.harvestportal.org/dataset/ukraine-field-boundary-delineation-2022) (search extract)

### Inferences
- For Samara, Tatarstan and Donbas (forest-steppe and steppe chernozem, very large mechanised annual-crop fields, shelterbelts, winter and spring cereals, sunflower), the most landscape-similar training signal among open models is DelAny (Ukraine operational maps) and FTW's Baltic/Slovak/Austrian annual-crop fields.
- The FTW 10 m U-Net reliably separates fields when boundaries are wide (roads, shelterbelts), but tends to split large fields where in-field heterogeneity exists. The global paper notes that internal crop, tillage or soil differences can produce multiple polygons. Russian forest-zone margins (Tatarstan north) may suffer the boreal-clearing and wet-meadow false positives seen in Finland.
- Fine-tuning on local labels is likely needed for production quality. Public label sources near the target area are the NASA Harvest Ukraine boundaries and DelAny Ukraine outputs (pseudo-labels); no Russian LPIS exists publicly. This is an inference; no source evaluates on Russia.

### Gaps
- Whether FBIS-22M/73M contains Ukrainian, Russian, Kazakh or Polish cadastral labels: the per-country list was not retrieved.
- Licence and download location of the DelAny Ukraine 2024 vector product: a "Field Boundaries (Delineate Anything)" source.coop listing (source.coop/khvzix) surfaced in search with "4.35 million fields at 5 m" for Ukraine 2024, but this could not be verified.
- Exact Russia/Ukraine test-site results of Waldner & Diakogiannis 2020 not retrieved.

## Key question 3: What accuracy did they report, especially on out-of-domain countries?

### Takeaway
In-domain FTW numbers look high: PRUE reaches 76% pixel IoU and 47% object F1 on the FTW test set, and the global map gets F1 0.89 for Austria and 0.88 for Latvia at pixel level. Object-level, out-of-domain performance is much weaker: Finland F1 0.74, collapsing in Lapland; over-fragmentation in Zambia; Ethiopian fields filtered out. Delineate Anything v2 is the only model with a 100-country manual zero-shot benchmark: mAP@0.5 0.559 overall and 0.612 for Europe (v1: 0.275). An independent agribound test of DelAny v2 on Sentinel-2 10 m gave object F1 of only 0.34 (New Mexico). Ten-metre data caps object-level quality: PlanetScope 3 m raises panoptic quality from 21.0 to 35.5 at matched recipe.

### Cited Findings
- **PRUE (FTW test):** 76% IoU and 47% object F1, which the authors describe as +6 and +9 over the previous FTW baseline. A U-Net beats instance-based models and GFMs; it is +9 IoU and +11 object F1 over the best GFM (Clay). PRUE also proposes deployment metrics: translation consistency, input-order sensitivity, preprocessing spec, and spatial scale. — [PRUE MSR page / abstract](https://www.microsoft.com/en-us/research/publication/prue-a-practical-recipe-for-field-boundary-segmentation-at-scale/) (search extract)
- A search extract of the ftw-baselines release notes reports pixel IoU 0.76 and object recall 0.31 for the CC-BY PRUE EfficientNet-B3 on FTW Full. Unverified; release page not fetched. — [ftw-baselines releases](https://github.com/fieldsoftheworld/ftw-baselines/releases) (search extract)
- **FTW paper (2024/25):** models pre-trained on FTW show better zero-shot and fine-tuning performance in held-out countries than models not pre-trained on diverse data. One summary example: India zero-shot pixel recall of 0.14 vs 0.16 for a fine-tuned AI4Boundaries-style model. Per-country zero-shot tables were not retrieved. — [arXiv 2409.16252](https://arxiv.org/abs/2409.16252) (search extract)
- **FTW global map (PRUE-B7 CC-BY):**
  - Pixel recall against FTW ground truth: mean 0.852 across 24 countries, above 0.90 in 14 countries. Brazil 0.970, Lithuania 0.955, France 0.953; Luxembourg 0.327, Corsica 0.576, Cambodia 0.694. Portugal (Azores) scored 0.
  - Full-country LPIS evaluation (2024, unfiltered):

    | Country | Precision | Recall | F1 | IoU |
    |---|---|---|---|---|
    | Austria | 0.900 | 0.884 | 0.892 | 0.805 |
    | Latvia | 0.870 | 0.898 | 0.884 | 0.792 |
    | Finland | 0.650 | 0.860 | 0.740 | 0.588 |

  - Finland F1 is 0.83 at 60° N but below 0.13 in Lapland (66° N), from over-prediction on boreal forest clearings, bogs and wet meadows.
  - Zambia: median polygon 0.07 ha vs 0.31 ha in a reference product, with fractal dimension 1.05, i.e. pixel-grid boundaries and over-fragmentation of smallholder fields.
  - The confidence layer has AUC 0.82 (5-fold) and a leave-one-country-out mean of 0.84 (range 0.64 Brazil to 0.95 Slovakia). The paper reports no object-level accuracy.
  — [FTW global map paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- **FTP (3 m vs 10 m):**
  - Panoptic quality 21.0 at 10 m S2 vs 35.5 at 3 m PlanetScope.
  - Fields under 0.5 ha: 5.8 vs 15.7.
  - Boundary error: 18.6 m vs 7.4 m.
  - PRUE+ augmentation raises object F1 from 29.1 to 37.4 on countries absent from training.
  — [FTP arXiv 2607.04449](https://arxiv.org/abs/2607.04449) (search extract)
- **Delineate Anything:**

  | Model | Test set | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
  |---|---|---|---|---|---|
  | DelAny v1 | FBIS-22M (mostly Europe) | 0.720 | 0.477 | – | – |
  | DelAny-S | FBIS-22M (mostly Europe) | 0.632 | 0.383 | – | – |
  | DelAny v1 | 100-country manual benchmark | 0.275 | 0.103 | 0.345 | 0.454 |
  | DelAny v2 | 100-country manual benchmark | 0.559 | 0.278 | 0.639 | 0.525 |

  - Regional mAP@0.5 for v2: Europe 0.612, Africa 0.584, Asia & Oceania 0.440, Latin America 0.563, North America 0.618. v1: 0.332, 0.251, 0.161, 0.314, 0.317.
  - Latency 25 ms per 512×512 patch (A100). On FBIS-22M, SAM scored 0.339 and SAM2 0.382 mAP@0.5.
  — [Delineate-Anything README](https://github.com/Lavreniuk/Delineate-Anything); [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines); [DelAny arXiv](https://arxiv.org/abs/2504.02534) (search extract)
- The v2 paper says it could not quantitatively benchmark against other open pipelines because their weights are not public, and uses visual comparison instead. It claims existing layers, including FTW, "often collapse boundaries, omitting smallholder fields". This is a qualitative judgment. — [DelAny v2 arXiv 2607.19069](https://arxiv.org/abs/2607.19069) (search extract)
- **Independent tests (agribound 1.0.1):**
  - DelAny v2 as released, against 944 NMOSE reference polygons in San Juan County, NM (object F1 at IoU ≥ 0.5): 0.15 at 30 m Landsat, **0.34 at 10 m Sentinel-2**, 0.33 at 6 m SPOT, 0.43 at 1 m NAIP; recall rises from 0.08 to 0.44.
  - DINOv3 fine-tuned + SAM2, in-sample F1: 0.38 on S2 vs 0.59 on NAIP.
  - On oil-palm smallholders (Papua New Guinea), DelAny v2 drew 1 polygon among 302 parcels.
  — [agribound README](https://github.com/montimaj/agribound)
- **DECODE:** Australia pixel accuracy 0.87 and field-based accuracy up to 0.88. The South Africa-trained model transferred well to Australia. — [DECODE README](https://github.com/waldnerf/decode)
- **SAM on S2 (Canadian Prairies):** IoU 0.86 vs manual segmentation. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12857264/) (search extract)
- **U-TAE + PaPs (PASTIS, France):** PQ 40.4 (later SOTA entries around 52.6 PQ per a leaderboard extract). — [utae-paps README](https://github.com/VSainteuf/utae-paps); [leaderboard extract](https://vercel.hyper.ai/en/sota/tasks/panoptic-segmentation/benchmark/panoptic-segmentation-on-pastis) (search extract)
- **NASA Harvest Ukraine (non-DL):** median IoU 0.77 against 12,331 GNSS boundaries. — [Nature Sci Data](https://www.nature.com/articles/s41597-025-05190-7) (search extract)

### Inferences
- Metrics across families are not comparable: pixel IoU/F1 (FTW), mAP on instance masks (DelAny), PQ (U-TAE/FTP), and median field IoU (NASA Harvest). The only like-for-like external comparisons found are the PRUE 18-model benchmark on FTW and the agribound site tests.
- For large mechanised fields, pixel-level F1 of 0.85–0.89 (Austria/Latvia-like) is a plausible upper bound for out-of-the-box FTW/PRUE in Russian steppe, if crop-calendar windows are correct. Object-level quality is unknown and likely lower. This is extrapolation; there is no Russian validation.

### Gaps
- No published accuracy for any open model on Russian, Kazakh, Ukrainian (with ground truth) or Polish fields, except DelAny's European aggregate and the Latvia/Finland/Lithuania numbers above.
- The full PRUE per-country and per-model tables and the FTW-paper zero-shot tables were not retrieved.

## Key question 4: Licence restrictions (commercial use, CC-BY-NC, AGPL)

### Takeaway
For commercial or clean-licence use, take the **FTW CC-BY-4.0 checkpoints** (v1 CCBY 2/3-class, v3.1 PRUE B3/B5/B7 CCBY) and the **CC-BY global map**. The "FULL" FTW and PRUE v3 checkpoints are mixed or non-commercial (trained on data including CC-BY-NC-SA). The FTP PlanetScope models are CC-BY-NC-4.0. Delineate Anything (code and weights) is **AGPL-3.0**, inherited from Ultralytics: usable commercially, but derivative or networked-service source-disclosure obligations apply. CSIRO DECODE/tfcl code carries a BSD-3 condition on contributions. Google ALU is API-only.

### Cited Findings
- FTW code: MIT. — [ftw-baselines LICENSE/README](https://github.com/fieldsoftheworld/ftw-baselines)
- FTW weights by registry licence field:

  | Licence | Checkpoints |
  |---|---|
  | "Mixed Open Licenses" (data includes CC-BY-NC-SA and non-CC open licences) | FTW v1 FULL, v2 (single and multi window), v3 PRUE B3/B5/B7 |
  | "CC-BY-4.0" | v1 CCBY 2/3-class, v3.1 PRUE CCBY B3/B5/B7 |
  | "CC-BY-NC-4.0" | FTP B3/B7 |
  | "AGPL-3" | DelineateAnything, DelineateAnything-S, DelineateAnythingV2 |

  — [model_registry.py](https://github.com/fieldsoftheworld/ftw-baselines/blob/main/ftw_tools/inference/model_registry.py)
- README advice: "Consider using CC-BY FTW Trained Checkpoints ... for Commercial Purpose. For Non-Commercial Purpose and Academic purpose, you can use the FULL FTW Trained Checkpoints". — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- TorchGeo labels the FULL v1 and PRUE (non-CCBY) weights `'license': 'non-commercial'`, and the CCBY ones `'CC-BY-4.0'`. — [TorchGeo unet.py](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/unet.py)
- FTW global field-boundary map: CC-BY, in fiboa GeoParquet. Use note: "It is not a cadastral or land-tenure product". — [FTW global map paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- ftw-prue code: MIT. — [ftw-prue README](https://github.com/fieldsoftheworld/ftw-prue)
- Delineate Anything: "This project is licensed under the AGPL-3.0 License", and "This code is based on Ultralytics". — [Delineate-Anything README](https://github.com/Lavreniuk/Delineate-Anything)
- The fbis-73m-polygons community conversion lists "MIT" next to the citation; the original FBIS-73M licence is not confirmed. — [HF isaaccorley/fbis-73m-polygons](https://huggingface.co/datasets/isaaccorley/fbis-73m-polygons) (search extract)
- DECODE: "CSIRO BSTD/MIT LICENSE", with modifications made available to be under BSD-3. No weights. — [DECODE README](https://github.com/waldnerf/decode)
- tfcl / PTAViT3D: CSIRO Open Source Software Licence v1.0 (MIT/BSD with the BSD-3 condition). — [tfcl LICENSE/README](https://github.com/feevos/tfcl)
- Other licences:

  | Project | Licence | Source |
  |---|---|---|
  | cultionet | Apache-2.0 | [cultionet LICENSE](https://github.com/jgrss/cultionet) |
  | sentinel-hub field-delineation | MIT (archived; needs paid Sentinel Hub) | [field-delineation](https://github.com/sentinel-hub/field-delineation) |
  | ai4boundaries downloader | MIT (the dataset licence itself was not retrieved) | [ai4boundaries](https://github.com/waldnerf/ai4boundaries) |
  | utae-paps | MIT | [utae-paps](https://github.com/VSainteuf/utae-paps) |
  | agribound | Apache-2.0 | [agribound](https://github.com/montimaj/agribound) |
  | AgriFM | Apache-2.0 | [AgriFM](https://github.com/flyakon/AgriFM) |
  | samgeo / geoai | MIT | [segment-geospatial](https://github.com/opengeos/segment-geospatial); [geoai](https://github.com/opengeos/geoai) |

- Radiant/NASA Harvest Rwanda gold model: the LICENSE file says CC-BY-NC-4.0 while `docs/index.md` says CC-BY-4.0. This is a conflict to resolve with the maintainers. — [gold repo](https://github.com/radiantearth/model_nasa_rwanda_field_boundary_competition_gold)
- Google ALU: hosted API, no weights found. — [Google developers](https://developers.google.com/agricultural-understanding/landscape-understanding) (search extract)

### Inferences
- For a Russian commercial or agency pipeline:
  - **PRUE CC-BY (v3.1)** is the cleanest licence choice: CC-BY-4.0 weights with attribution only.
  - **DelAny under AGPL** is fine for internal or research use. Offering it as a network service, or distributing modified code, triggers AGPL source-sharing obligations. Whether AGPL applies to derived outputs (polygons) is a legal question, not answered in the sources.
  - The FULL/v3 non-CC-BY FTW checkpoints should be treated as non-commercial.

### Gaps
- No HF model-card text was read for the DelAny weights, AI4Boundaries dataset licence (likely CC-BY per JRC norms, but unverified), the Planet EU field-boundaries licence, or the DelAny Ukraine vector product.
