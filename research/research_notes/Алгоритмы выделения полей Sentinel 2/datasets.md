# Public labelled datasets for field-boundary delineation with Sentinel-2, and how Russia-like they are

Research note, compiled 2026-10-26. Scope: open datasets and label sources usable to train or evaluate Sentinel-2 field-boundary models for Samara, Tatarstan and Donbas. The project's own ground truth is about 136 hand-labelled polygons tagged sown, fallow or unused.

Method caveat: in this session WebFetch could reach only GitHub and PyPI. Zenodo, arXiv, Source Cooperative, Hugging Face, nature.com and the Russian portals failed DNS or were blocked by the proxy. Facts from those hosts therefore come from search-engine summaries of the pages. Where possible they were cross-checked against GitHub READMEs and dataset loader code (torchgeo 0.10.1 sdist from PyPI, the ftw-baselines README, the ftw-datasets-list repo) or the full text of the FTW global-map PDF, which was downloaded and parsed. Items marked "unverified" were not confirmed against a primary page.

---

## Q1. What public labelled field-boundary datasets exist, and what do they contain?

### Takeaway
Three open resources have Sentinel-2-scale field instances at scale:
- **FTW:** 24 countries (25 regions), about 1.6 M parcels, 70,462 bi-temporal Sentinel-2 chips.
- **AI4Boundaries:** 7 EU countries, 7,831 samples, about 2.5 M parcels.
- **FBIS-22M / FBIS-73M (Delineate Anything):** 22.9 M and 73 M instances, multi-resolution 0.25–10 m, mostly Europe.

PASTIS, Sen4AgriNet, ZueriCrop, BreizhCrops and EuroCrops(ML) are mainly crop-type datasets. EuroCrops and the national LPIS/GSAA releases are vector labels only. No benchmark dataset with paired imagery covers Russia, Ukraine, Kazakhstan, Belarus or Moldova. The open polygons that do exist in that zone are GloCAB (hand-digitised, includes Russia and Ukraine) and several model-derived national Ukraine layers.

### Cited Findings

**Fields of The World (FTW): benchmark**
- 70,462 patches across 24 countries. Each country has train/val/test splits. Images are 4-channel 256×256 GeoTIFFs. Targets are 2-class and 3-class semantic masks (background, field, field-boundary) and instance masks. Kenya, Rwanda, Brazil and India have presence-only labels (class 3 = unlabeled). — [torchgeo 0.10.1 `FieldsOfTheWorld` (ftw.py)](https://pypi.org/project/torchgeo/0.10.1/); [FTW paper arXiv 2409.16252](https://arxiv.org/abs/2409.16252)
- The input is two aligned Sentinel-2 L2A scenes, Window A (planting) and Window B (harvest), with bands B04, B03, B02, B08 each. The windows were chosen from the USDA crop calendar. — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- Per-country parcels, chips and licences, from the FTW datasets list:

  | Country | Year | Parcels | Chips | Licence |
  |---|---|---|---|---|
  | Austria | 2021 | 196,101 | 6,686 | CC-BY-4.0 |
  | Belgium | 2021 | 63,431 | 1,941 | open, no restrictions |
  | Brazil | 2020 | 1,854 | 1,607 | CC-BY-4.0 |
  | Cambodia | 2021 | 318,088 | 344 | CC-BY-4.0 |
  | Corsica | 2021 | 5,360 | 2,472 | CC-BY-2.0 |
  | Croatia | 2023 | 157,481 | 3,482 | "Open Data" |
  | Denmark | 2021 | 37,677 | 3,560 | CC0 |
  | **Estonia** | 2021 | 26,695 | 6,713 | "CC-3.0" (as written) |
  | **Finland** | 2021 | 57,323 | 5,665 | CC-BY-4.0 |
  | France | 2020 | 55,342 | 3,744 | Licence Ouverte |
  | Germany | 2018/19 | 4,598 | 686 | DL-DE/BY-2-0 |
  | India | 2016 | 10,013 | 2,002 | CC-BY-4.0 |
  | Kenya | 2022 | 874 | 391 | GPL-2.0+ |
  | **Latvia** | 2021 | 44,964 | 6,938 | **CC-BY-NC-4.0** |
  | **Lithuania** | 2021 | 61,424 | 5,258 | **non-commercial only** |
  | Luxembourg | 2022 | 29,018 | 808 | CC0 |
  | Netherlands | 2022 | 43,169 | 3,879 | CC0 |
  | Portugal | 2021 | 5,040 | 86 | CC-BY-NC-4.0 |
  | Rwanda | 2021 | 1,532 | 70 | CC-BY-4.0 |
  | **Slovakia** | 2021 | 14,242 | 4,073 | CC0 |
  | Slovenia | 2021 | 67,488 | 2,177 | CC-BY-4.0 |
  | South Africa | 2018 | 6,568 | 747 | CC-BY-NC-SA-4.0 |
  | Spain | 2020 | 258,465 | 2,440 | CC-BY-4.0 |
  | Sweden | 2021 | 39,718 | 4,760 | open, no restrictions |
  | Vietnam | 2021 | 120,913 | 288 | CC-BY-4.0 |

  — [fieldsoftheworld/ftw-datasets-list](https://github.com/fieldsoftheworld/ftw-datasets-list)
- FTW has **no Ukraine, Russia, Kazakhstan, Belarus, Moldova, Poland or Romania**. The closest members are the Baltic states, Finland, Slovakia and Austria. — [ftw-datasets-list](https://github.com/fieldsoftheworld/ftw-datasets-list)
- There is no single licence. Each source keeps its own, and the Hugging Face card counts 25 regions (24 countries plus Corsica). — [Voxel51/fields-of-the-world HF card (via search)](https://huggingface.co/datasets/Voxel51/fields-of-the-world)
- FTW is hosted at `source.coop/kerner-lab/fields-of-the-world`, with per-country zips under `data.source.coop/kerner-lab/fields-of-the-world-archive/`. — [torchgeo ftw.py](https://pypi.org/project/torchgeo/0.10.1/); [FTW org](https://github.com/fieldsoftheworld)

**FTW global field-boundary map (2024, 2025): a derived product, not a training set**
- The map has 3.17 billion polygons: 1.62 B for 2024 and 1.55 B for 2025, across 241 countries and territories. The minimum polygon is 4 px (about 0.04 ha). — [Kerner/Robinson et al., FTW Global Map v1.1 PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf); [arXiv 2605.11055](https://arxiv.org/abs/2605.11055)
  - The Source Cooperative listing gives different figures: 195 countries, 1.63 B and 1.58 B polygons. — [source.coop/ftw/global-data (via search)](https://source.coop/ftw/global-data)
- Model and inputs:
  - The model is PRUE, a U-Net with an EfficientNet-B7 encoder. It was trained only on the CC-BY subset of FTW, with 8-channel input (RGBN × 2 seasons).
  - Inputs are per-tile median composites for a planting window and a harvest window. The windows come from the WorldCereal crop calendar (SOS/EOS), using scenes with less than 20% cloud and an SCL mask.
  - The output is 3 classes at 10 m.

  — [FTW Global Map PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- Accuracy against national LPIS, unfiltered, seasonal crops only:

  | Country | F1 | Precision | Recall |
  |---|---|---|---|
  | Austria | 0.892 | 0.900 | 0.884 |
  | Latvia | 0.884 | 0.870 | 0.898 |
  | Finland | 0.740 | 0.650 | 0.860 |

  In Finland, accuracy fell with latitude, from F1 0.83 at 60°N to below 0.13 in Lapland, because the model over-predicted fields on forest clearings, bogs and wet meadows. — [FTW Global Map PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- Stated limitations:
  - "North America, China, Russia and Central Asia … are entirely absent from training," as is "boreal cropping".
  - The model inherits FTW's scope of annual crops only. Permanent grassland, pasture, orchards and **fallow** were excluded from training and evaluation.
  - A 500 m confidence layer accompanies the map; the recommended threshold is conf ≥ 0.4.
  - The map is not a cadastral product.
  - It is released under **CC-BY** at source.coop/ftw/global-data.

  — [FTW Global Map PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- Formats are Zarr, GeoParquet and PMTiles. The model inputs (Sentinel-2 median composites) are published as COG and Zarr v3. — [source.coop/ftw/global-data docs (via search)](https://source.coop/ftw/global-data/docs)

**AI4Boundaries (JRC, ESSD 2023)**
- 7,831 samples:

  | Region | Samples |
  |---|---|
  | Austria | 2,091 |
  | Catalonia | 652 |
  | France | 2,078 |
  | Luxembourg | 132 |
  | Netherlands | 1,157 |
  | Slovenia | 301 |
  | Sweden | 1,420 |

  — [d'Andrimont et al., ESSD 15, 317 (2023)](https://essd.copernicus.org/articles/15/317/2023/)
- Imagery: Sentinel-2 10 m monthly composites for March–August 2019 (R, G, B, NIR), in 256×256 chips, plus 1 m orthophotos in 512×512 chips. — [ESSD 2023](https://essd.copernicus.org/articles/15/317/2023/)
- Labels: open 2019 GSAA parcels, about 2.5 M parcels over 47,105 km². Hosted in the JRC Open Data Catalogue. — [ESSD 2023](https://essd.copernicus.org/articles/15/317/2023/); [JRC repository](https://publications.jrc.ec.europa.eu/repository/handle/JRC130529)
- Label-mask layers: my recollection is extent, boundary, distance and enumeration/instance masks — **unverified in this session**.
- Dataset licence is unclear. The paper is CC BY 4.0, and a citing page shows CC BY-SA 4.0. — [ESSD/ResearchGate (via search)](https://www.researchgate.net/publication/367221722_AI4Boundaries_an_open_AI-ready_dataset_to_map_field_boundaries_with_Sentinel-2_and_aerial_photography)

**Delineate Anything: FBIS-22M and FBIS-73M (Lavreniuk et al., Ukrainian-led team)**
- **FBIS-22M:** 672,909 image patches at 0.25–10 m with 22,926,427 instance masks. Sources include Sentinel-2, PlanetScope, Pleiades and Maxar. Its test splits "primarily covered only Europe." The v1 paper is listed CC BY-NC-SA 4.0. — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines); [Delineate-Anything GitHub](https://github.com/Lavreniuk/Delineate-Anything); [arXiv 2511.13417 (via search)](https://arxiv.org/abs/2511.13417)
- **FBIS-73M:** 73 M instances across 61 countries, at 0.25–10 m. It is hosted at [HF MykolaL/FBIS-73M](https://huggingface.co/datasets/MykolaL/FBIS-73M) together with a separate 100-country zero-shot benchmark of 400 images (4 per country). A third-party mirror describes it as about 1.48 M chips with YOLO instance-segmentation labels. — [Delineate-Anything GitHub](https://github.com/Lavreniuk/Delineate-Anything); [arXiv 2607.19069](https://arxiv.org/abs/2607.19069); [isaaccorley/fbis-73m-polygons (via search)](https://huggingface.co/datasets/isaaccorley/fbis-73m-polygons)
- Model and code:
  - The model is YOLO-based (Ultralytics) with RGB input. The code is **AGPL-3.0**. Weights are DelineateAnything.pt and DelineateAnythingv2.pt, 125 MB each.
  - mAP@0.5 on the global 100-country benchmark is 0.275 for v1 and 0.559 for v2.
  - The v2 paper was accepted at ECCV 2026.

  — [Delineate-Anything GitHub](https://github.com/Lavreniuk/Delineate-Anything)

**Crop-type and panoptic datasets (secondary for boundary work)**
- **PASTIS / PASTIS-R** (France):
  - 2,433 patches of 128×128 px at 10 m, with 10 Sentinel-2 L2A bands.
  - Time series are of variable length. PASTIS-R adds about 70 ascending and 70 descending Sentinel-1 observations per patch.
  - 124,422 parcels, 18 crop classes, semantic and panoptic labels.
  - Zenodo sizes: PASTIS 29 GB zipped, PASTIS-R 54 GB, pixel-set version 27 GB.

  — [pastis-benchmark README](https://github.com/VSainteuf/pastis-benchmark); [torchgeo pastis.py](https://pypi.org/project/torchgeo/0.10.1/)
- **Sen4AgriNet (S4A)** (Catalonia and France):
  - 42.5 M parcels: about 2.5 M in Catalonia (2016–2020) and about 40 M in France (2016–2019).
  - netCDF patches with all 13 Sentinel-2 bands plus `labels` and `parcels` layers.
  - Downloads via Dropbox, Google Drive or [HF orion-ai-lab/S4A](https://huggingface.co/datasets/orion-ai-lab/S4A).
  - The source LPIS is ODC-By; the HF loader states MIT.

  — [S4A README](https://github.com/Orion-AI-Lab/S4A); [Sykas et al. arXiv 2204.00951 (via search)](https://arxiv.org/abs/2204.00951)
- **ZueriCrop** (Switzerland): 24×24 px time series at 10 m with 9 bands. About 28 k time series of 142 images each (116 k images). Instance and crop-hierarchy labels. Downloads from ETH polybox, with an HF mirror used by torchgeo. — [torchgeo zuericrop.py](https://pypi.org/project/torchgeo/0.10.1/); [ms-convSTAR README](https://github.com/0zgur0/multi-stage-convSTAR-network)
- **BreizhCrops** (Brittany): parcel-level Sentinel-2 time series (L1C and L2A) with no image chips. About 580 k parcels in v1, about 610 k observations in the later version. 9 classes, labels from the IGN RPG. — [Rußwurm et al., ISPRS 2020](https://isprs-archives.copernicus.org/articles/XLIII-B2-2020/1545/2020/); [arXiv 1905.11893 (via search)](https://arxiv.org/abs/1905.11893)
- **EuroCropsML:** 706,683 parcel time series from 2021 (Estonia 175,906; Latvia 431,143; Portugal 99,634). Each is a per-parcel median of 13 Sentinel-2 L1C bands, with 176 classes. Zenodo 10683255. — [EuroCropsML docs](https://eurocropsml.readthedocs.io/en/latest/)

**Vector-only aggregations and national LPIS/GSAA**
- **EuroCrops:** harmonised declared parcels from 16 countries, with crop names mapped to the HCAT taxonomy. Vector only.
  - Countries: AT 2021, BE-VLG 2021, DE-LS and DE-NRW 2021, DK 2019, EE 2021, ES-NA 2020, FR 2018, HR 2020, LT 2021, LV 2021, NL 2021, PT 2021, RO (no year), SE 2021, SI 2021, SK 2021.
  - Licence **CC BY-SA 4.0**, distributed on Zenodo 8229128 (v11). EuroCrops 2.0 is hosted at JRC.

  — [EuroCrops GitHub](https://github.com/maja601/EuroCrops)
- **Open field-boundary sources, by country:** the FTW harmonized catalog (fiboa format) lists these collections:
  - **Austria:** 2018–2025.
  - **Czechia:** 2019–2026.
  - **Slovakia:** 2018–2026.
  - **Poland:** 2025 and 2026.
  - **Lithuania:** 2024–2025.
  - **Latvia:** 2015–2025.
  - **Estonia:** 2016–2024.
  - **Finland:** 2020–2025.
  - **Romania:** `ec_ro`, the EuroCrops copy.
  - **Bulgaria:** 2021–2025.
  - **Croatia:** 2011–2024.
  - **Denmark:** 2008–2026.
  - **Sweden:** 2015–2025.
  - **Slovenia:** 2019–2024.
  - Belgium (Flanders and Wallonia), the Netherlands, Luxembourg, France, Portugal, Ireland, Spain (national and regional), several German Länder, Italian regions and Swiss cantons.
  - Outside Europe: **US (USDA, 2017–2024)**, California, **Canadian Prairies "ca_prairies" (2024)**, Japan, New Zealand and Brazil (CC-BY-NC).
  - **No entries for Ukraine, Russia or Kazakhstan.**

  — [harmonized-field-data-catalog datasets.yaml](https://github.com/fieldsoftheworld/harmonized-field-data-catalog/blob/main/datasets.yaml)
- **Poland:**
  - ARiMR publishes declared parcels with crop for 2025 and 2026 (about 10.7 M parcels per year) over WFS. It also publishes LPIS reference and cadastral parcels (about 37.1 M) over WFS/WMS and as weekly voivodeship shapefiles.
  - No account is needed, but **no licence is stated** (the data is tagged only "Publiczne dane ARiMR").

  — [fiboa data-survey PL.md](https://github.com/fiboa/data-survey/blob/main/data/PL.md)
- **Romania:**
  - The only surveyed open polygon set is a CC0 INSPIRE land-cover layer for 7 Danube counties (Mehedinți, Dolj, Olt, Teleorman, Giurgiu, Călărași, Constanța). It comes from a 2007–2013 cross-border project and has no stated year.
  - It is not LPIS.

  — [fiboa data-survey RO.md](https://github.com/fiboa/data-survey/blob/main/data/RO.md)
- **Coverage gaps in the fiboa survey:** it has no entry for Hungary, Canada, Ukraine, Russia or Kazakhstan. — [fiboa/data-survey](https://github.com/fiboa/data-survey)

**Other regions**
- **AI4SmallFarms** (Vietnam and Cambodia): 439,001 field polygons in 62 tiles of about 5×5 km, with Sentinel-2 composites. CC BY 4.0, DANS DOI 10.17026/dans-xy6-ngg6 (Persello et al. 2023, IEEE GRSL). — [Univ. Twente record](https://research.utwente.nl/en/publications/ai4smallfarms-a-data-set-for-crop-field-delineation-in-southeast-/); [source.coop ftw/ai4sf-ml README (via search)](https://data.source.coop/ftw/ai4sf-ml/README.md)
- **South Africa:**
  - Western Cape field boundaries (Radiant MLHub) were annotated on 2016 aerial imagery, with 4,151 training labels, licensed CC BY-NC-SA 4.0. — [arXiv 2404.00179 (via search)](https://arxiv.org/pdf/2404.00179)
  - Kerner Lab's "Fieldscapes" South Africa (2018) is also CC-BY-NC-SA-4.0. — [beta.source.coop fieldscapes-southafrica (via search)](https://beta.source.coop/kerner-lab/fieldscapes-southafrica)
  - The crop-type competition data now sits on Source Cooperative (`radiantearth/south-africa-crops-competition`). — [torchgeo south_africa_crop_type.py](https://pypi.org/project/torchgeo/0.10.1/)
- **Rwanda** field-boundary competition: 1,532 fields in 70 chips of 256×256, using **Planet NICFI** imagery (not Sentinel-2). — [torchgeo rwanda_field_boundary.py](https://pypi.org/project/torchgeo/0.10.1/)
- **AgriFieldNet** (India): 7,081 fields in 1,217 Sentinel-2 chips of 256×256. These are crop-type labels and are only partially labelled. — [torchgeo agrifieldnet.py](https://pypi.org/project/torchgeo/0.10.1/)
- **US:**
  - The FSA Common Land Unit (CLU) is confidential.
  - NASS **Crop Sequence Boundaries (CSB)** are public, non-confidential field polygons. They are derived from historic Cropland Data Layers plus road and rail networks, with no ownership information. — [Hunt et al., Stat. J. IAOS (via search)](https://content.iospress.com/articles/statistical-journal-of-the-iaos/sji230078); [GEE community catalog CSB](https://gee-community-catalog.org/projects/csb/)
- **Canada:**
  - AAFC "Automatically Detected Field Boundaries in Canada, 2023" was produced by OneSoil's proprietary model on Sentinel-2. It is published as GeoPackage/FGDB and listed under OGL-Canada on one record, but terms are inconsistent across mirrors. — [open.canada.ca record](https://open.canada.ca/data/en/dataset/b3dba669-6611-4cb1-b4be-6c317b9908ad); [donnees.iriu.ca resource (licence ca-ogl-lgo)](https://donnees.iriu.ca/en/dataset/b3dba669-6611-4cb1-b4be-6c317b9908ad/resource/6a8388de-50e8-459e-994a-021e90bac19e)
  - Ha et al. (2026, Data in Brief) published a Prairie-wide (AB, SK, MB) field-boundary set derived with SAM from Sentinel-2 L2A at 10 m, as Shapefile and GeoParquet. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340926002441); [Mendeley Data 2y568rt76w](https://data.mendeley.com/datasets/2y568rt76w)
- **Sentinel Hub / EDC "Field Delineation"** is an on-demand, paid processing service rather than a dataset. Output is CC BY 4.0, the model was trained in Europe, and data is available from 2016. — [eurodatacube public-collections field-delineation.yaml](https://github.com/eurodatacube/public-collections/blob/main/collections/field-delineation.yaml)

### Inferences
- For boundary segmentation specifically, only FTW, AI4Boundaries, FBIS, AI4SmallFarms and PASTIS (via its instance masks) provide image chips with instance labels. The rest give crop-type time series or vectors.
- FTW is the de facto standard because it has a CLI, a torchgeo loader, per-country downloads and pretrained checkpoints. Its Baltic, Finnish and Slovak subsets are the closest fit for Russia.

### Gaps
- FTW total download size and per-country zip sizes were not retrieved (source.coop unreachable).
- The AI4Boundaries dataset licence and exact mask layers were not confirmed.
- FBIS-73M country list and licence were not confirmed. Whether it includes Ukraine, Russia or Kazakhstan training chips is unknown; the per-country tables are on the Hugging Face card and project page.
- The Sen4AgriNet total size and patch count (often cited as about 225 k patches of 366×366) were not verified.
- DACIA5, a Sentinel-1/2 crop identification dataset that may cover Romania, was found but not inspected. — [Big Earth Data 2025](https://www.tandfonline.com/doi/full/10.1080/20964471.2025.2512685)
- FieldSeg reference boundaries (Zenodo 14630397): region not determined.

---

## Q2. Which datasets ship Sentinel-2 imagery paired with labels, and which are labels only? How easy are they to load in Google Colab?

### Takeaway
**Ready-to-train (imagery and masks paired):**
- FTW: bi-temporal RGBN, 256×256.
- AI4Boundaries: 6 monthly RGBN composites, 256×256, plus 1 m orthophotos.
- PASTIS/PASTIS-R: dense time series, 128×128.
- Sen4AgriNet: 13 bands, netCDF.
- ZueriCrop: 24×24 time series.
- AI4SmallFarms: Sentinel-2 composites.
- FBIS-22M/73M: multi-source RGB chips.

**Labels only (you must fetch Sentinel-2 yourself):**
- EuroCrops, national LPIS/GSAA (via fiboa).
- USDA CSB and Canadian AAFC/OneSoil and Prairie SAM sets.
- GloCAB.
- All the Ukraine national products.
- The FTW global map, which ships predictions plus its input composites.

FTW is the easiest to use in Colab: `pip install ftw-tools`, then `ftw data download --countries=...`, or torchgeo `FieldsOfTheWorld(download=True)`.

### Cited Findings
- **FTW** in Colab:
  - Download with `ftw data download --countries latvia,estonia,finland`. Use `--no-unpack` and `ftw data unpack` to separate download from extraction, and `-o` to set the output folder.
  - The package is `ftw-tools` (pyproject 2.0.0b5 on GitHub; 1.4.3 on PyPI at time of check), MIT-licensed. The CLI also runs inference, polygonisation (output GeoParquet/fiboa, FGB or GPKG), LULC filtering and Delineate Anything.

  — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines); [ftw-tools on PyPI](https://pypi.org/project/ftw-tools/)
- **FTW in torchgeo:**
  - Class `FieldsOfTheWorld(root, countries=[...], target='2-class'|'3-class'|'instance', download=True)`.
  - Per-country zips come from `data.source.coop/kerner-lab/fields-of-the-world-archive/`. Each contains `s2_images/window_a`, `s2_images/window_b`, `label_masks/...` and `chips_<country>.parquet`.

  — [torchgeo 0.10.1 ftw.py](https://pypi.org/project/torchgeo/0.10.1/)
- **FTW pretrained checkpoints:** "CC-BY" checkpoints are for commercial use. "FULL" checkpoints, which include NC-licensed countries, are for non-commercial or academic use. — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- **FTW global map** is cloud-native: Zarr, GeoParquet and PMTiles for predictions, and COG and Zarr v3 for the Sentinel-2 input composites, on Source Cooperative. — [source.coop/ftw/global-data docs (via search)](https://source.coop/ftw/global-data/docs)
- **PASTIS** is on Zenodo: 29 GB zipped, 54 GB for PASTIS-R. torchgeo `PASTIS` downloads PASTIS-R.zip from Zenodo 5735646. — [pastis-benchmark README](https://github.com/VSainteuf/pastis-benchmark); [torchgeo pastis.py](https://pypi.org/project/torchgeo/0.10.1/)
- **Sen4AgriNet:** netCDF patches via Dropbox, Google Drive or HF Hub, with a PyTorch `PatchesDataset` and COCO-style split files. — [S4A README](https://github.com/Orion-AI-Lab/S4A)
- **ZueriCrop** loads through torchgeo from an HF mirror (`hf.co/datasets/isaaccorley/zuericrop`). — [torchgeo zuericrop.py](https://pypi.org/project/torchgeo/0.10.1/)
- **EuroCrops** loads through torchgeo `EuroCrops` (a VectorDataset) from Zenodo 8229128. — [torchgeo eurocrops.py](https://pypi.org/project/torchgeo/0.10.1/)
- **AI4Boundaries:** distributed via the JRC Open Data Catalogue. — [ESSD 2023](https://essd.copernicus.org/articles/15/317/2023/)
- **FBIS-73M:** on the Hugging Face Hub. Delineate Anything has a Colab demo and an HF Space. — [Delineate-Anything GitHub](https://github.com/Lavreniuk/Delineate-Anything)
- **GloCAB:** a single 17.1 MB zipped shapefile on Zenodo 10479122. — [Zenodo (via search)](https://zenodo.org/records/10479122)
- **Ukraine NASA Harvest boundaries:** one ZIP per oblast, about 435 MiB to 1 GiB each, on Harvest Portal and Zenodo. — [Harvest Portal 2023 resource (via search)](https://data.harvestportal.org/sq/dataset/ukraine-field-boundary-delineation-2023/resource/6b10228d-8782-45cb-837e-ae85d76399c1)
- **Ukraine 2023 parcel raster** (Li et al. 2026): single-band GeoTIFF in EPSG:3857 on Zenodo 17183516. — [Zenodo (via search)](https://zenodo.org/records/17183516)
- **Poland ARiMR WFS:** use WFS 1.1.0 for the declared-crop layers; pages hold up to 50 k features (`startIndex`). Shapefile ZIPs are in EPSG:2180. — [fiboa data-survey PL.md](https://github.com/fiboa/data-survey/blob/main/data/PL.md)

### Inferences
**Colab practicality**
- **Easiest:** FTW (Baltic, Finnish or Slovak subsets are small per-country zips), GloCAB (17 MB) and the global FTW map. The map can be read by bounding box over HTTPS with DuckDB or GeoParquet without downloading the whole dataset; in Samara or Tatarstan it serves as pseudo-labels or a baseline.
- **Medium:** AI4Boundaries, EuroCrops per country, and NASA Harvest oblast ZIPs (Donetsk/Luhansk only).
- **Heavy:** PASTIS-R (54 GB) and full FBIS-73M. These need Google Drive or GCS staging and streaming via `huggingface_hub`.

**Fetching your own Sentinel-2 for vector-only labels**
- Pull imagery from Planetary Computer or Earth Search STAC, or GEE, and match FTW's format: two seasons, RGBN, 256×256.
- For Russia, choose the window dates by hand; in FTW's own pipeline they come from the WorldCereal calendar.
- Use a May window and a July–August window so that winter crops, spring crops, fallow and stubble separate.

### Gaps
- Total download sizes for FTW, AI4Boundaries, Sen4AgriNet and FBIS-73M could not be confirmed.
- The exact Source Cooperative S3 paths and partitioning for global-map GeoParquet were not inspected (host unreachable).

---

## Q3. What licensing limits apply?

### Takeaway
The commercial-use blockers are:
- FTW's Latvia, Lithuania, Portugal and South Africa subsets (NC).
- FBIS-22M (CC BY-NC-SA per the v1 paper).
- The Delineate Anything code (AGPL-3.0).
- The South Africa Western Cape labels (CC BY-NC-SA).

These are clean for research and commercial use with attribution:
- FTW's CC-BY/CC0 subset: Austria, Finland, Slovakia, Denmark, Netherlands, Luxembourg, Slovenia, Spain and others.
- The FTW global map (CC-BY).
- AI4SmallFarms (CC BY 4.0).
- NASA Harvest Ukraine (CC BY per mirrored listings).

EuroCrops is CC BY-SA, so derivatives must share alike. Russian state sources (НСПД, ЕФИС ЗСН, VEGA) carry no open licence.

### Cited Findings
- **FTW per-country labels:**
  - Latvia CC-BY-NC-4.0, Lithuania non-commercial only, Portugal CC-BY-NC-4.0, South Africa CC-BY-NC-SA-4.0.
  - Estonia is "CC-3.0", with the variant unclear.
  - Finland and Austria are CC-BY-4.0; Slovakia is CC0.

  — [ftw-datasets-list](https://github.com/fieldsoftheworld/ftw-datasets-list)
- **FTW models:** CC-BY-only checkpoints are recommended for commercial use, and FULL checkpoints for non-commercial use. FTW code is MIT. — [ftw-baselines README](https://github.com/fieldsoftheworld/ftw-baselines)
- **FTW global map:** CC-BY. — [FTW Global Map PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- **Delineate Anything:**
  - Code is AGPL-3.0, with no separate weights licence stated. — [Delineate-Anything GitHub](https://github.com/Lavreniuk/Delineate-Anything)
  - FBIS-22M (v1 paper) is CC BY-NC-SA 4.0. The FBIS-73M licence is not confirmed; a mirror claims MIT for its derived polygons. — [search summary of arXiv/HF](https://huggingface.co/datasets/MykolaL/FBIS-73M)
- **EuroCrops:** CC BY-SA 4.0. Users must also cite the country sources. — [EuroCrops GitHub](https://github.com/maja601/EuroCrops)
- **Sen4AgriNet:** source LPIS under ODC-By; the HF loader says MIT. — [search summary](https://www.huggingface.co/datasets/orion-ai-lab/S4A/blob/main/S4A.py)
- **AI4SmallFarms:** CC BY 4.0. — [Univ. Twente record (via search)](https://research.utwente.nl/en/publications/ai4smallfarms-a-data-set-for-crop-field-delineation-in-southeast-/)
- **South Africa Western Cape (MLHub):** CC BY-NC-SA 4.0. — [arXiv 2404.00179 (via search)](https://arxiv.org/pdf/2404.00179)
- **Poland ARiMR:** no licence stated. — [fiboa PL.md](https://github.com/fiboa/data-survey/blob/main/data/PL.md)
- **Romania cross-border land cover:** CC0. — [fiboa RO.md](https://github.com/fiboa/data-survey/blob/main/data/RO.md)
- **Canada AAFC/OneSoil 2023:** OGL-Canada on one record; "AI Training Allowed" and "Redistribution Allowed" are blank on another copy. — [donnees.iriu.ca (via search)](https://donnees.iriu.ca/en/dataset/b3dba669-6611-4cb1-b4be-6c317b9908ad/resource/6a8388de-50e8-459e-994a-021e90bac19e)
- **NASA Harvest Ukraine boundaries 2021–2023:** CC BY on Harvest Portal listings. — [Harvest Portal (via search)](https://data.harvestportal.org/sq/dataset/ukraine-field-boundary-delineation-2023/resource/6b10228d-8782-45cb-837e-ae85d76399c1)
- **Rosreestr:** a 2012 letter states that public cadastral maps are objects of copyright. — [КонсультантПлюс подборка](https://www.consultant.ru/law/podborki/dannye_publichnoj_kadastrovoj_karty/)
- **VEGA-Science:** used under a cooperation agreement with ЦКП «ИКИ-Мониторинг», not an open-data licence. — [sci-vega.ru agreement template](http://sci-vega.ru/dogovor_vega_sci.pdf)

### Inferences
- **Research or thesis use:** all of the above is usable, citing each source.
- **Commercial product:** train on the FTW CC-BY subset plus the FTW global map and the CC-BY Ukraine layers. Avoid FULL FTW checkpoints, FBIS-22M and the Latvia/Lithuania subsets. Treat Delineate Anything's AGPL as viral if you run it as a network service.
- **Russian cadastral/НСПД geometry:** do not use it as redistributable labels.

### Gaps
- Licences not confirmed: GloCAB (Zenodo), FBIS-73M, the AI4Boundaries data, the Li et al. 2026 Ukraine raster, the Ha et al. 2026 Prairie set, and the Estonia "CC-3.0" variant.

---

## Q4. What exists for Russia and its neighbours (Ukraine, Kazakhstan, Belarus, Moldova), and what Russian sources are actually open?

### Takeaway
- **Russia:** GloCAB is the only openly downloadable, hand-labelled field-polygon set found that includes Russian territory. It has 4–5 Russian regions digitised on 20 m Sentinel-2 SWIR composites from 2016–2020 and is small. Beyond that, the FTW global map (CC-BY) gives model predictions for Russia, but Russia was absent from its training data and fallow is excluded.
- **Ukraine:** well covered by open, *model-derived* national layers: NASA Harvest 2021–2023 (PlanetScope, CC BY, over 5 M fields), Li et al. 2023 (10 m raster) and DelAnyFlow 2024. These are good pseudo-labels for Donbas.
- **Kazakhstan, Belarus, Moldova:** no open labelled field datasets were found.
- **Russian state sources:** none publishes open field contours. ЕФИС ЗСН requires ЕСИА login for rights-holders. The Public Cadastral Map has moved to НСПД, and its parcels are legal plots, not fields. VEGA-Science access goes through an agreement. OSM completeness in Samara and Tatarstan is unmeasured.

### Cited Findings

**Russia: open labelled polygons**
- **GloCAB** (Hall, Argueta, Giglio, *Data in Brief* 2024, DOI 10.1016/j.dib.2024.110739):
  - 190,832 manually digitised cropland field boundaries in 22 regions of 74–38,000 km², in Brazil, **Ukraine**, USA, Canada and **Russia**.
  - Fields were drawn over 20 m Sentinel-2 imagery (B12/B8A/B4 composite), with a mapping date range of 2016–2020 per region.
  - Each region was QC'd by a senior analyst. Fields too small to identify were grouped and flagged "NoArea".
  - Zenodo 10479122, v1.0, 17.1 MB zip.

  — [PMC article (via search)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11293513); [Zenodo record (via search)](https://zenodo.org/records/10479122); [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340924007066)
- The GloCAB burned-area ESSD preprint lists Russia reference areas A–D plus a Russia_E validation region. Its table values (46.1, 54.3, 113.9, 45.6) are unlabeled and may not be field counts. — [ESSD preprint essd-2023-191](https://essd.copernicus.org/preprints/essd-2023-191/essd-2023-191.pdf)

**Russia: model-derived coverage**
- The FTW global map covers 241 countries and territories for 2024 and 2025, under CC-BY. Its authors state that "Russia and Central Asia" are "entirely absent from training" and that fallow and pasture are out of scope. — [FTW Global Map PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)

**Ukraine**
- **NASA Harvest** (Sadeh et al. 2025, *Scientific Data* 12:833):
  - In-season field boundaries for 2021–2023 from 3 m PlanetScope bi-weekly time series, totalling over 5 M fields.
  - Validated against **12,331 field boundaries from GNSS-equipped farm machinery**.
  - Zenodo records 12770018 (2021), 12786399 (2022) and 12791903 (2023), plus Harvest Portal. One ZIP per oblast, licensed CC BY.

  — [Sci Data article (via search)](https://www.nature.com/articles/s41597-025-05190-7); [Zenodo 2023 (via search)](https://zenodo.org/records/12791903); [Harvest Portal (via search)](https://data.harvestportal.org/sq/dataset/ukraine-field-boundary-delineation-2023/resource/6b10228d-8782-45cb-837e-ae85d76399c1)
- **Li et al. 2026** (*Scientific Data*): a 10 m national parcel raster for Ukraine 2023 from Sentinel-2 with HBGNet 2.0, reporting OA 92.86%, F1 91.03% and IoU 85.70%. Zenodo 17183516 holds GeoTIFFs only; the training vector labels may not be public. — [Sci Data (via search)](https://www.nature.com/articles/s41597-026-07837-5); [Zenodo (via search)](https://zenodo.org/records/17183516)
- **DelAnyFlow:** a Ukraine 2024 layer from Sentinel-2 covering 603,000 km² in under 6 h. It reports about 3.75 M fields at 5 m and 5.15 M at 2.5 m, against 2.66 M from Sinergise and 1.69 M from NASA Harvest (operational products). The authors say national vector outputs are available on their project page. — [arXiv 2511.13417 (via search)](https://arxiv.org/abs/2511.13417); [Delineate-Anything GitHub](https://github.com/Lavreniuk/Delineate-Anything)

**Kazakhstan**
- No public labelled field dataset was found. A 2025 *Agronomy* study in Turkestan Region digitised its own boundaries, with no public download. — [Agronomy 15(9):2040](https://doi.org/10.3390/agronomy15092040)
- The Kazakh Ministry of Agriculture has orders on electronic soil, geobotanical and land-cadastre maps, but no open field-contour service was found. — [FAOLEX/adilet (via search)](https://leap.unep.org/en/countries/kz/national-legislation/order-no-186-minister-agriculture-validating-methodology-creation)

**Belarus and Moldova**
- No specific datasets were found. Only the global FTW map covers them. — [search summary](https://arxiv.org/pdf/2605.11055)

**ЕФИС ЗСН (ЕФГИС ЗСН, Минсельхоз)**
- The system stores location, condition and actual use of agricultural land, including field contours within each field, municipality and region. — [Пермский край, ЕФИС ЗСН](https://agro.permkrai.ru/deyatelnost/efis-zsn)
- Producers log in via ЕСИА (Госуслуги) and work in «Госмониторинг» → field registry. — [Минсельхоз Пензенской обл., FAQ](https://mcx.pnzreg.ru/news/tsifrovizatsiya-apk/6386/); [direct.farm instruction](https://direct.farm/content/e7e/e7e4bbbdbb844e66979f45c0524053ac1704646.pdf)
- A rights-holder must submit an application to add a field. — [direct.farm](https://direct.farm/post/yefis-zsn-25882)
- The legal basis is ПП РФ № 154 of 02.02.2023 (state register of agricultural lands as a subsystem) and 101-ФЗ. New rules for rights-holders took effect on 1 March 2026. — [rshzm.ru regulatory docs](http://rshzm.ru/regulatorydocs); [finkont.ru](https://www.finkont.ru/blog/obiazannosty-pravoobladateley-selhozzemel-kak-rabotat-s-efgis-zsn/)
- No public, no-login map or open-data export of field contours was found. Thematic layers are described as available to "authorised users". — [АГВУ УрГАУ 2019](https://agvu.urgau.ru/images/Agricultural_Journal/05_2019/16.pdf); [search summaries](https://svoefermerstvo.ru/svoemedia/articles/efis-zsn-aktual-naja-i-dostovernaja-informacija-o-zemljah-sel-hoznaznachenija)

**Regional portals**
- **Samara:**
  - The regional geoportal has a «Карта ГИС АПК Самарской области» section. — [geoportal.samregion.ru/apk](https://geoportal.samregion.ru/apk/)
  - apk.geosamara.ru has a "Самарская область" group (created 20.04.2023) with layers for EGRN agricultural land, **ЕФИС ЗСН (2,262,794.21 ha)**, agricultural zones per ПЗЗ and satellite-analysis results. Whether field contours are public or downloadable is **unconfirmed**. — [apk.geosamara.ru/groups/12 (via search)](https://apk.geosamara.ru/groups/12/)
- **Krasnodar:** fields are digitised in the regional system maps.krasnodar.ru. — [kaicc.ru](https://www.kaicc.ru/content/ocifrovka-poley-rgis-mapskrasnodarru)
- **Tatarstan:** no open agro-GIS with field contours was found.

**Rosreestr / НСПД**
- Rosreestr retired the Public Cadastral Map and moved its data to НСПД (nspd.gov.ru, operated by Rosreestr). — [Yandex Realty journal](https://realty.yandex.ru/journal/post/nspd-publichnaya-kadastrovaya-karta/); [cadastre.ru](https://cadastre.ru/news/1107)
- One source says access is via a Госуслуги account and that some functions may be paid. — [gosuslugi municipal news](https://shhigrovskij-r38.gosweb.gosuslugi.ru/dlya-zhiteley/novosti-i-reportazhi/novosti_658.html)
- Layers include land parcels, municipal boundaries, protected zones and utilities. No agricultural-field layer was found. — [Yandex Realty journal](https://realty.yandex.ru/journal/post/nspd-publichnaya-kadastrovaya-karta/)
- **Cadastral parcels are not fields:**
  - A field can be one parcel, several parcels or parts of several. Cadastral, production and technological boundaries differ, and registry coordinates are often inaccurate. — [agrarii.com](https://agrarii.com/elektronnaya-karta-poley-pochemu-narisovat-granitsy-nedostatochno/); [agrarii.com](https://agrarii.com/starye-kadastrovye-granitsy-i-fakticheskaya-obrabotka-pochemu-tsifry-ne-sovpadayut/)
  - Multi-contour parcels ("единое землепользование", allocations from land shares) persist on agricultural land. — [rkc56.ru](https://rkc56.ru/articles/7418)

**IKI RAN VEGA-Science**
- The service added a map of cultivated lands for 2019–2024 (news item 2025-12-25) and a winter-crop map for 2025–26 (2026-04-21). — [sci-vega.ru](http://sci-vega.ru/)
- VEGA-Pro offers arable-land and winter-crop maps since 2001 and used-land maps for some regions. — [pro-vega.ru products](http://pro-vega.ru/products.sht)
- IKI's older map of used arable land of Russia is MODIS-based. — [smiswww.iki.rssi.ru](http://smiswww.iki.rssi.ru/default.aspx?page=317)
- Access is governed by a cooperation agreement. No open bulk download was confirmed. — [VEGA-Science agreement](http://sci-vega.ru/dogovor_vega_sci.pdf); [АлтГАУ news](https://www.asau.ru/novosti-i-sobytiya/nauchnye-meropriyatiya/8430-altajskij-gau-i-institut-kosmicheskikh-issledovanij-ran-dogovorilis-o-sovmestnoj-rabote-po-ispolzovaniyu-sistemy-sputnikovogo-monitoringa-vega-science-dlya-resheniya-zadach-selskogo-khozyajstva-v-regione)

**OpenStreetMap**
- `landuse=farmland` is defined as tillage land for annual crops, and the RU wiki recommends drawing the polygon around the field boundary. — [OSM wiki RU:Tag:landuse=farmland](https://wiki.openstreetmap.org/wiki/RU:Tag:landuse=farmland)
- No study of OSM farmland completeness or accuracy in Russia, Samara or Tatarstan was found. OSM is generally weaker in rural areas. — [ScienceDirect: OSM land-cover completeness patterns](https://www.sciencedirect.com/science/article/abs/pii/S0143622822001138)

### Inferences
**Labels for Donbas**
- The NASA Harvest Donetsk/Luhansk oblast files (2021–2023, CC BY) and the DelAnyFlow 2024 layer can be cut out and paired with Sentinel-2 composites to make thousands of Russia-steppe-like training chips.
- Expect label noise: they are PlanetScope or model-derived, and war-affected fields in 2022–2024 add abandonment dynamics.

**Labels for Samara and Tatarstan**
- No open ground truth exists beyond OSM, GloCAB (if a Russian region falls nearby) and the FTW global map.
- The realistic path is pseudo-labels from the FTW global map or Delineate Anything v2, corrected by hand. Measure OSM farmland coverage with an Overpass query per district before relying on it.
- The 136-polygon ground truth should be the held-out test set.

**Fallow and abandoned land (залежь)**
- No public boundary dataset labels fallow or abandoned land explicitly. FTW and its global map deliberately exclude fallow.
- The user's sown/fallow/unused attribute is therefore unique. A separate land-use classifier is needed on top of boundary delineation.

### Gaps
- GloCAB's exact Russian regions (oblasts), field counts per region and licence need the Zenodo readme.
- НСПД's current access regime (anonymous viewing vs Госуслуги login) and any machine-readable API terms are not confirmed from an official Rosreestr source.
- Whether ЕФИС ЗСН contours are ever exposed publicly, for example through regional ГИС АПК such as Samara's, is not verified.
- OSM farmland completeness in Samara and Tatarstan was not measured; Overpass was unreachable from this session.
- Whether FBIS-73M or the 100-country benchmark includes Russian or Kazakh chips is unknown.

---

## Q5. Which dataset or region gives the best "Russia-like" training and evaluation data?

### Takeaway
No single open dataset is Russia-like and also ships paired Sentinel-2. The best combination:
1. **Ukraine-derived labels:** NASA Harvest 2021–2023 (CC BY), Li et al. 2023 and DelAnyFlow 2024, all paired with your own Sentinel-2 composites. These match Donbas almost exactly: steppe, chernozem, shelterbelt grid, large fields, winter wheat/sunflower/maize.
2. **GloCAB Russia and Ukraine regions:** hand-digitised on Sentinel-2. Small, but the only truly Russian labels.
3. **FTW Baltic, Finland and Slovakia subsets:** ready-made bi-temporal Sentinel-2 pairs with a continental or boreal-transition climate, the closest analogue for Tatarstan's forest-steppe. Fields are smaller and there are no shelterbelts.
4. **Canadian Prairies and US CSB:** structurally similar (large mechanised fields, chernozemic soils, shelterbelts), but the labels are model-derived and come without imagery.

The FTW global map (CC-BY) is a strong zero-shot baseline and pseudo-label source for Samara and Tatarstan. Its authors flag Russia as out-of-distribution and fallow as excluded.

### Cited Findings
- FTW geography is "heavily European (17 of 24 countries)". Russia, Central Asia, North America and boreal cropping are absent from training. — [FTW Global Map PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- FTW global-map F1 is 0.89 in Austria and 0.88 in Latvia against full national LPIS, but 0.74 in Finland. Boreal clearings, bogs and wet meadows were over-predicted. — [FTW Global Map PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- FTW Baltic, Finnish and Slovak volumes:

  | Country | Parcels | Chips | Licence |
  |---|---|---|---|
  | Estonia | 26,695 | 6,713 | — |
  | Latvia | 44,964 | 6,938 | NC |
  | Lithuania | 61,424 | 5,258 | NC |
  | Finland | 57,323 | 5,665 | CC-BY |
  | Slovakia | 14,242 | 4,073 | CC0 |
  | Austria | 196,101 | 6,686 | CC-BY |

  — [ftw-datasets-list](https://github.com/fieldsoftheworld/ftw-datasets-list)
- AI4Boundaries countries are Austria, Catalonia, France, Luxembourg, Netherlands, Slovenia and Sweden; it has no Eastern European or steppe sites. — [ESSD 2023](https://essd.copernicus.org/articles/15/317/2023/)
- The NASA Harvest Ukraine boundaries were validated against 12,331 GNSS machinery-derived field boundaries, which makes them comparatively trustworthy pseudo-labels. — [Sci Data 2025 (via search)](https://www.nature.com/articles/s41597-025-05190-7)
- The Delineate Anything v2 authors note that FBIS-22M was concentrated in Europe. FBIS-73M expands to 61 countries. Their Ukraine national run demonstrates performance in a Russia-like landscape. — [arXiv 2607.19069 (via search)](https://arxiv.org/abs/2607.19069); [Delineate-Anything GitHub](https://github.com/Lavreniuk/Delineate-Anything)
- GloCAB was digitised on 20 m Sentinel-2 SWIR composites, so its boundaries are coarser than 10 m RGBN FTW labels. Fields too small to identify were grouped as "NoArea". — [PMC GloCAB (via search)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11293513)
- Canadian Prairie boundaries (Ha et al. 2026) were SAM-derived from Sentinel-2 at 10 m. The AAFC 2023 layer is from OneSoil's proprietary ML. Both are model outputs, not hand labels. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340926002441); [open.canada.ca](https://open.canada.ca/data/en/dataset/b3dba669-6611-4cb1-b4be-6c317b9908ad)
- USDA CSB polygons are derived from CDL crop rotations and road/rail networks, not from farmer declarations. — [Stat. J. IAOS (via search)](https://content.iospress.com/articles/statistical-journal-of-the-iaos/sji230078)
- LPIS parcels (Poland ARiMR, Baltic, Finland) are declared management parcels. Poland's 2025–2026 data adds crop type for about 10.7 M parcels per year. — [fiboa PL.md](https://github.com/fiboa/data-survey/blob/main/data/PL.md)

### Inferences
The scores below are qualitative judgements from general agro-geographic knowledge (field size, climate, soils, shelterbelts, crop mix, fallow). They are not sourced measurements; verify field-size distributions empirically.

| Candidate | Similarity to Samara/Tatarstan (forest-steppe/steppe, Dfb) | Similarity to Donbas (steppe, Dfa/Dfb, chernozem, shelterbelts) | Paired S2? | Label quality | Licence | Verdict |
|---|---|---|---|---|---|---|
| Ukraine NASA Harvest 2021–23 / DelAnyFlow 2024 / Li 2023 | High | **Very high** (same landscape) | No (fetch S2) | Model-derived; NASA Harvest GNSS-validated | CC BY (NASA Harvest); others to check | **Best Russia-like training source** for steppe, as pseudo-labels |
| GloCAB Russia and Ukraine regions | **High** (actual Russian fields) | High | No (vectors only, 20 m) | Hand-digitised, QC'd; coarse | Unconfirmed | Best **evaluation** add-on to the 136 polygons; too small to train on alone |
| FTW Latvia, Lithuania, Estonia, Finland | Medium (climate and season similar to Tatarstan; smaller fields, more forest/grass mosaic; no steppe) | Low–medium | **Yes** | LPIS-quality | LV/LT NC; FI CC-BY | Best **ready-made** pretraining set; fine-tune afterwards |
| FTW Slovakia, Austria (also AI4B Austria) | Medium-low (Danubian lowland large fields, otherwise Alpine) | Medium (Pannonian plain chernozems) | **Yes** | LPIS | CC0 / CC-BY | Useful extra CC-licensed pretraining |
| Poland ARiMR 2025–26 | Medium-low (fragmented east, larger west) | Low–medium | No | Declared parcels with crop | No licence stated | Large label pool; needs your own imagery |
| Canadian Prairies (AAFC/OneSoil, Ha 2026), US CSB | Medium-high (large mechanised fields, continental, chernozemic soils, shelterbelts) | High | No | Model- or CDL-derived | OGL-Canada (to check); public | Good structural analogue; label-noise risk |
| AI4Boundaries (FR, NL, SE, SI, LU, Catalonia) | Low–medium | Low | **Yes** (6 monthly composites) | GSAA | Unclear | Secondary |
| PASTIS, S4A, ZueriCrop, BreizhCrops, AI4SmallFarms, Rwanda, South Africa | Low | Low | Yes or no | Various | Various (some NC) | Not recommended for the Russia target |
| FTW global map 2024/25 | Covers Samara, Tatarstan, Donbas directly | Same | Inputs published | Model prediction; Russia out-of-distribution; fallow excluded | CC-BY | **Zero-shot baseline and pseudo-label source** |

**Recommended pipeline**
1. Pretrain on, or start from, FTW CC-BY checkpoints. Add the Baltic, Finnish and Slovak subsets for continental and northern seasons.
2. Fine-tune on Sentinel-2 chips built in FTW format (two windows, RGBN, 256×256) over Donetsk, Luhansk and adjacent Ukrainian oblasts, using NASA Harvest or DelAnyFlow polygons as labels.
3. Add pseudo-labels from the FTW global map or Delineate Anything v2 for Samara and Tatarstan, filtered by confidence and corrected by hand.
4. Evaluate on the 136 hand-labelled polygons plus any GloCAB Russian regions.
5. Report fallow and unused land separately, because all public sets treat them as non-field or exclude them.

Shelterbelts, about 10–30 m wide, are only 1–3 Sentinel-2 pixels, so a boundary class weighted as in FTW (boundary weight 0.75) suits them. Of the open sources, only Ukrainian, Canadian and US data contain this pattern in volume.

### Gaps
- No quantitative comparison of field-size distributions between Samara, Tatarstan or Donbas and FTW countries was found. Computing the median field area of the 136 ground-truth polygons against FTW, NASA Harvest and GloCAB would settle it.
- Published accuracy of the FTW global map or Delineate Anything v2 in Russia, Kazakhstan or Ukraine was not found, beyond DelAnyFlow's field counts.
- Whether the FBIS-73M training set or the 100-country benchmark contains Russian or Kazakh samples, which would make DA v2 better for Russia than FTW, is unverified.
