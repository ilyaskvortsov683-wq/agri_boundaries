# Field boundary delineation / parcel mapping in Russia and CIS (Ukraine, Kazakhstan, Belarus), with emphasis on Sentinel-2

*How the research was done (affects reliability): in this session direct page fetching was blocked for almost every host (arXiv, elibrary, cyberleninka, computeroptics.ru, d33.infospace.ru, MDPI, Nature, ResearchGate, company sites). I read two primary documents in full: the FTW global-map paper (Microsoft-hosted PDF) and the Delineate-Anything GitHub README (raw.githubusercontent). Everything else comes from search-engine result summaries of the cited pages. Treat numbers marked "per search summary" as needing a check against the full text. Current to late October 2026.*

---

## Q1. Which Russian studies (ИКИ РАН, universities) address automatic delineation of arable field boundaries, and what did they use?

### Takeaway
Very little peer-reviewed Russian work targets **field-boundary (parcel) delineation** specifically. Most Russian research maps **arable land / cropland extent**, not individual field polygons. ИКИ РАН (Space Research Institute) uses classical time-series segmentation and semi-automatic, seed-based contouring inside its VEGA systems. The clearest recent deep-learning study on Sentinel-2 is from **Samara University** (Bavrina & Agafonov, 2025, UPerNet + Swin-V2, Samara oblast). It segments arable land, not individual fields, and its accuracy figures could not be retrieved.

### Cited Findings
**ИКИ РАН (Bartalev, Lupyan, Plotnikov, Kolbudaev, Tolpin, Denisov)**
- Plotnikov D.E., Kolbudaev P.A., Bartalev S.A. "Identification of dynamically homogeneous areas with time series segmentation of remote sensing data", *Computer Optics* (Компьютерная оптика), vol. 42, 2018. This is ИКИ's classical, non-neural method for segmenting satellite time series into homogeneous (field-like) units. Reference lists give issue 3, pp. 447–456. I did not retrieve the abstract or accuracy. — [ScienceDirect "cited by" listing via search](https://www.sciencedirect.com/science/article/abs/pii/S0167865511004119); issue/pages from search summary of [Computer Optics 2022 reference list](https://computeroptics.ru/eng/KO/Annot/KO46-5/460516e.html)
- An ИКИ РАН/HSE review of agricultural monitoring methods lists "выделение границ полей методом сегментации данных ДЗЗ" (field-boundary extraction by segmenting remote sensing data) as a Plotnikov and Kolbudaev technology — [HSE mirror of ИКИ review (search summary)](https://inagres.hse.ru/mirror/pubs/share/direct/286823205)
- Related ИКИ work on cropland mask (not boundaries): Plotnikov, Kolbudaev, Bartalev, Lupyan, "Automated annual cropland mapping from reconstructed time series of Landsat data", *Современные проблемы ДЗЗ из космоса* (СПДЗЗ), 2018, 15(2): 112–127 — [conference/journal index (search summary)](https://ikiconf.txldev.ru/files/books/2022/9535.htm). Earlier arable-land mapping used MODIS with the LAGMA locally adaptive classification: Bartalev, Egorov, Lupyan, Plotnikov, Uvarov, *Computer Optics* 2011, 35(1): 103–116 — [search summary citing Computer Optics 45(4)](https://www.computeroptics.ru/KO/Annot/KO45-4/4504ED.html); [ResearchGate: Mapping of arable land in Russia using MODIS and LAGMA](https://www.researchgate.net/publication/289496422_Mapping_of_arable_land_in_Russia_using_multi-year_time_series_of_MODIS_data_and_the_LAGMA_classification_technique)
- VEGA-Science field contouring is **semi-automatic**. The operator places points where field boundaries should be extracted. The system then segments a series of images automatically and vectorises the result, and the polygon is corrected by hand if needed (Denisov et al., 2021) — [ResearchGate: Vega-Science system: design features, main capabilities and usage experience (search summary)](https://www.researchgate.net/publication/357760327_Vega-Science_system_design_features_main_capabilities_and_usage_experience)
- VEGA-PRO, ИКИ's agricultural service: Denisov, Lupyan, Tolpin, Troshko, "Спутниковый сервис «ВЕГА-PRO» для сельскохозяйственного мониторинга", *Управление рисками в АПК*, 2021, no. 39, pp. 22–36, DOI 10.53988/24136573-2021-01-02. The service relies on fully automatic processing of incoming satellite data — [ИКИ smiswww page / pro-vega.ru (search summary)](http://smiswww.iki.rssi.ru/default.aspx?page=444); [pro-vega.ru](http://pro-vega.ru/)
- An ИКИ lab annual report says high-resolution data are used in the system mainly to determine field boundaries and to correct information derived from medium-resolution data — [ИКИ РАН lab 563 report](http://www.iki.rssi.ru/annual/2009/R24c-09.htm)

**Samara University (Самарский университет им. Королёва)**
- Bavrina A.Yu., Agafonov A.A., "Распознавание пахотных земель на территории Самарской области по спутниковым снимкам для решения задач землепользования" (Recognition of arable land in Samara oblast from satellite images for land-use tasks), *Computer Optics*, 2025, 49(6): 1002–1011, DOI 10.18287/2412-6179-CO-1754:
  - Deep learning for extracting **arable-land boundaries** from both single-date and multi-temporal Sentinel-2.
  - Best result: **UPerNet with a Swin Transformer v2** encoder.
  - Inputs: composites over the five months of most active field vegetation change, cut into 256×256 tiles.
  - The output vector arable-land layer is used to detect illegal ploughing of specially protected natural areas.
  - Accuracy metrics were not retrieved. — [PDF (computeroptics.ru)](https://computeroptics.ru/KO/PDF/KO49-6/490616.pdf); [Agafonov staff page](https://ssau.ru/english/staff/364272302-agafonov-anton-a/publ)
- Samara University and the company «Самис» (SAMIS) launched an automatic neural-network service that identifies and ranks **abandoned/unused agricultural parcels**. It compares cadastre data with satellite imagery. The university reports about **95% accuracy** for abandonment detection, says the service was tested in Samara oblast, and says it suits about 50 regions (press release, Dec 2025) — [ТАСС Наука](https://nauka.tass.ru/nauka/25812761); [sovainfo.ru](https://sovainfo.ru/news/samarskie-uchenye-razrabotali-neyrosetevoy-servis-dlya-vyyavleniya-i-klassifikatsii-zabroshennykh-se/); [RuNews24, 04.12.2025](https://runews24.ru/samara/04/12/2025/uchenyie-samaryi-razrabotali-nejroservis-dlya-vyyavleniya-zabroshennyix-selxozugodij)
- Regional context: Samara oblast says **197.9 thousand ha** of unused land have returned to agricultural use since 2018 thanks to remote-monitoring systems — [CityTraffic, 2026](https://citytraffic.ru/2026/4/403081/v-samarskoi-oblasti-spytnikovie-tehnolog)
- A. Yu. Denisova (Samara University) has Sentinel-2 work in Samara oblast on tree species and on superpixel segmentation for change detection, not on fields — [Denisova staff page](https://ssau.ru/staff/380087762-denisova-anna-yurevna)

**Other Russian studies**
- Ermolaev N.R. et al., "Использование нейронных сетей и облачных сервисов для выделения геометрий сельскохозяйственных полей на основе данных ДЗЗ" (Using neural networks and cloud services to extract agricultural field geometries from remote sensing data), *Исследование Земли из космоса*, 2023, no. 6, pp. 98–105, DOI 10.31857/S0205961423060039:
  - Region: Stavropol krai.
  - Data and method: Sentinel-2, Google Earth Engine plus Kaggle for deep learning.
  - Training masks were rasterised from existing vector field data; each training square had to contain field boundaries.
  - Class threshold set empirically at 0.62.
  - Accuracy and authors' institution not retrieved. — [sciencejournals.ru](https://sciencejournals.ru/view-article/?j=iszem&y=2023&v=0&n=6&a=IsZem2306003Ermolaev)
- "Разработка нейросетевых моделей для автоматического мониторинга состояния сельскохозяйственных земель на основе спутниковых снимков" (Developing neural-network models for automatic monitoring of agricultural land from satellite images), *Успехи современного естествознания*. The network delineates field boundaries by **instance segmentation**, justified by the need to update cadastral boundaries. Authors, region and metrics not retrieved — [natural-sciences.ru id=38034](https://natural-sciences.ru/article/view?id=38034)
- A 2024 SGUGiT (Novosibirsk) conference paper on arable-land mapping in Khabarovsk krai uses Sentinel-2 NDVI time series with ML. It notes that federal agricultural-land databases hold incomplete arable-land data — [GEOSIB 2024 (search summary)](https://geosib.sgugit.ru/upload/geosibir/sborniki/2024/tom-4-1/020-025.pdf)
- Kazan Federal University (KFU): no KFU field-boundary paper was found. The Tatarstan-specific findings are operational (see Q3).
- Skoltech (Illarionova, Burnaev et al.): Sentinel-2 work on buildings (super-resolution to 2.5 m), cloud domain adaptation and forests. **No Skoltech field-boundary paper was found** — [MDPI RS 15(9):2347](https://www.mdpi.com/2072-4292/15/9/2347); [ResearchGate profile](https://www.researchgate.net/profile/Svetlana-Illarionova)

**English-language studies with Russian test sites**
- Waldner & Diakogiannis (2020), *Remote Sensing of Environment*, "Deep learning on edge" (ResUNet-a, field extent + boundary + distance multitask). The model generalised to secondary sites including **Russia and Ukraine**, plus Argentina, Australia and Canada — [arXiv 1910.12023](https://arxiv.org/pdf/1910.12023); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0034425720301115)
- Prishchepov et al. used Landsat to map farmland abandonment in European Russia (Ryazan/Mordovia, more than 67,000 km²; determinants study over 5 oblasts). This maps abandonment, not parcels. A Landsat-based estimate gave 37% abandonment for European Russia versus 22.5% from MODIS — [RSE 2013 Oksky/Mordovsky](https://www.sciencedirect.com/science/article/abs/pii/S0034425713000394); [Spatial Economics 2013](http://spatial-economics.com/eng/arkhiv-nomerov/2013/38-2013-3/570-prishchepov-av-mller-d-dubinin-m-baumann-m-radeloff-vc-determinants-of-the-spatial-distribution-of-abandoned-agricultural-lands-in-the-european-part-of-russia)

### Inferences
- The Russian state research school (ИКИ РАН) has favoured **classical, time-series-based segmentation with operator seeding**, built into operational systems (VEGA, ЕФИС ЗСН module). The deep-learning work is newer and mostly university-based (Samara, Stavropol case), and it targets the arable/non-arable mask more than instance-level fields.
- Samara University (Bavrina & Agafonov; Samara University–SAMIS) is the closest Russian research group to the end user's Samara oblast use case. It is a natural contact for regional reference data.

### Gaps
- Accuracy metrics (IoU/F1) for Bavrina & Agafonov 2025, Plotnikov et al. 2018 and Ermolaev et al. 2023 could not be retrieved: full texts were unreachable from this environment.
- I could not confirm whether ИКИ has a fully automatic neural field-contour product. No such publication surfaced.
- No papers were found from КФУ, ИПУ РАН, Cognitive Technologies, Skoltech or МГУ on field-boundary delineation specifically.
- Mentions found but not verified: arXiv 2506.16318 (SAM for field delineation with a "regional dataset ERAS") — its region was not determined; PMC12857264 (SAM on seasonal Sentinel-2) — region unknown.

---

## Q2. What exists for Ukraine, Kazakhstan and Belarus (including Donbas)?

### Takeaway
**Ukraine is by far the best-served CIS country.** Two open, national-scale field-boundary products exist: NASA Harvest/UMD for 2021–2023 (PlanetScope, CC-BY-4.0), and the Kussul/Shelestov/Lavreniuk "Delineate Anything" family (open AGPL code and weights; Ukraine-wide 2024 map from Sentinel-2). There is also war-impact and abandonment work covering Donetsk oblast. Kazakhstan has operational state systems and one crop-classification paper that compared hand-digitised and automatic boundaries. Belarus has essentially no published field-boundary DL work beyond being OneSoil's origin.

### Cited Findings
**Ukraine — NASA Harvest / UMD**
- Sadeh Y., Wagner J., Nair S.S. et al., "National-Scale In-Season Field Boundaries of Ukraine Using Remote Sensing", *Scientific Data* 12, 833 (2025), doi:10.1038/s41597-025-05190-7:
  - Covers 2021–2023, more than **5 million fields**.
  - Built from **Planet bi-weekly 3 m composites**, because Sentinel-2 cloud cover limits in-season mapping. 8–11 composites per oblast were usable.
  - Validated against **12,331 field boundaries from GNSS-equipped farm machinery**. Reference data came from agroholdings (Kernel, IMC, Frendt) covering 11 oblasts.
  - Headline accuracy numbers not retrieved. — [Nature](https://www.nature.com/articles/s41597-025-05190-7); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12095754/); [NASA Harvest news](https://www.nasaharvest.org/news/mapping-ukraines-fields-from-space-a-new-tool-for-food-security)
- The data are one shapefile per oblast per year on the Harvest Portal and Zenodo, licence **CC-BY-4.0** (the paper itself appears as CC BY-NC-ND) — [Zenodo 2023](https://zenodo.org/records/12791903); [Zenodo 2022](https://zenodo.org/records/12786399); [Zenodo 2021](https://zenodo.org/records/12770018); [Harvest Portal 2023](https://data.harvestportal.org/dataset/ukraine-field-boundary-delineation-2023)

**Ukraine — Space Research Institute NASU-SSAU / KPI (Kussul, Shelestov, Lavreniuk) with ESA**
- *Delineate Anything* (DelAny, arXiv 2504.02534, ECAI 2025):
  - Treats field delineation as **instance segmentation** (YOLO/Ultralytics-based).
  - FBIS-22M dataset: 672,909 patches at 0.25–10 m with 22.9 M field instances; test splits mostly in Europe.
  - Integrated into TorchGeo.
  - [GitHub README (read directly)](https://github.com/Lavreniuk/Delineate-Anything); [HF paper page](https://huggingface.co/papers/2504.02534)
- *Delineate Anything Flow* (arXiv 2511.13417, Nov 2025):
  - Produced a **Ukraine-wide 2024 field map from Sentinel-2**: 3.75 M fields at 5 m and 5.15 M at 2.5 m (super-resolved/interpolated inputs).
  - For comparison: 2.66 M fields from Sinergise Solutions and 1.69 M from NASA Harvest.
  - The gain is largest for smallholder fields of 0.25–1 ha.
  - About 25 pixels per field is the practical lower limit; Sentinel-2's minimum detectable field is about 0.5 ha.
  - [arXiv PDF (search summary)](https://arxiv.org/pdf/2511.13417); [review summary](https://www.themoonlight.io/en/review/delineate-anything-flow-fast-country-level-field-boundary-detection-from-any-source)
- *Delineate Anything v2* (ECCV 2026, arXiv 2607.19069). Authors: Lavreniuk, Kussul, Shelestov, Salii, Kuzin, Wang, Szantoi.
  - Trained on **FBIS-73M**: 73 M instances, **61 countries**, 0.25–10 m.
  - Maps all of Ukraine (603,000 km²) in 5.4 h on one RTX 5070 Ti.
  - Global 100-country benchmark: mAP@0.5 = 0.559, precision 0.639, recall 0.525. Europe 0.612, Asia & Oceania 0.440 (v1: 0.275 overall).
  - **Licence AGPL-3.0**. Weights on Hugging Face; ESA APEX "Map Explorer" link; Colab and HF demos; browser tool "Autobounds".
  - [GitHub README (read directly)](https://github.com/Lavreniuk/Delineate-Anything); [FBIS-73M on HF](https://huggingface.co/datasets/MykolaL/FBIS-73M); [arXiv 2607.19069](https://arxiv.org/abs/2607.19069)
- Kussul, Shelestov et al. (2025, *Int. J. Applied Earth Observation & Geoinformation*):
  - First annual **10 m land-use maps of Ukraine for 2016–2024** from Sentinel-1/2 with ML.
  - Cropland fell by **11% nationwide and about 22% in occupied areas**.
  - [ScienceDirect S1569843225001980](https://www.sciencedirect.com/science/article/pii/S1569843225001980)
- Kussul, Shumilo, Shelestov, Wagner (ESA LPS 2025) — field-level damage:
  - Sentinel-2 field polygons classified as shelled / burned / abandoned / ploughed.
  - About **12,000 km² of fields with artillery craters**, roughly 60% of cultivated land in the analysed (war-affected) area.
  - Occupied-area boundaries taken from ISW.
  - [LPS25 presentation PDF](https://lps25.esa.int/lps25-presentations/presentations/1622/_1622.pdf)
- Akymenko K., Sergieieva K., Kavats Yu., Kovrov O. (Dnipro University of Technology), "Information System for Abandoned Arable Land Detection From Sentinel-2 Images", CEUR-WS vol. 4048 (2025):
  - Compares NDVI time-series peaks in GEE against a reference year.
  - Tested in **Dnipropetrovsk and Donetsk** oblasts.
  - Accuracy up to **92.5%, F1 = 0.898**, including in combat areas.
  - [CEUR-WS PDF](https://ceur-ws.org/Vol-4048/paper24.pdf)
- A 2025 study of cultivation, abandonment, fallowing and recultivation versus conflict intensity notes that since 2022 official statistics cover only government-controlled territory — [ScienceDirect S2666017225001324](https://www.sciencedirect.com/science/article/pii/S2666017225001324)

**Kazakhstan**
- *Agronomy* 2025, 15(9): 2040, "Automatic Classification of Agricultural Crops Using Sentinel-2 Data in the Rainfed Zone of Southern Kazakhstan". Classification within **hand-digitised field boundaries** matched official statistics better than a fully automatic approach. The automatic approach is scalable but slightly less accurate — [doi:10.3390/agronomy15092040](https://doi.org/10.3390/agronomy15092040)
- Kazakhstan Gharysh Sapary's **AgroSpace** gives each farmer a personal map account with an **individual field number**. Confirmed data flow into the state subsidy system (ЕГИСС). The JerInSpectr monitoring system produced 18,000+ cards covering 8.3 Mha (188 districts, 17 oblasts) in 2026 — [AgroSpace](https://agrospace.gharysh.kz/); [24.kz](https://24.kz/ru/news/economyc/800777-sputniki-sledyat-za-polyami-ii-pomogaet-fermeram-kak-menyaetsya-selskoe-khozyajstvo-kazakhstana); [inbusiness.kz](https://inbusiness.kz/ru/last/zemlyu-uvidyat-iz-kosmosa-a-urozhaj-sprognoziruet-ii-kak-menyaetsya-apk-kazahstana)
- OneSoil says it is confident in its field-boundary quality for Europe, the US, Canada, Argentina, Brazil, South Africa, Australia **and Kazakhstan**. Russia and Ukraine are not on that list — [OneSoil B2B (search summary)](https://b2b.onesoil.ai/)

**Belarus**
- No BSU/UIIP field-boundary DL publications were found. A 2020 BSUIR conference paper used a neural network on seasonal NDVI to detect unused land, not boundaries — [BSUIR PDF](https://www.bsuir.by/m/12_100229_1_144999.pdf)
- OneSoil was founded by Belarusians (the original 2018 map covered Europe and the USA) — [vc.ru](https://vc.ru/offline/49421-tochnoe-zemledelie-belorusskie-razrabotchiki-nanesli-na-kartu-vse-polya-v-evrope-i-ssha); [bel.biz](https://bel.biz/news/startap-onesoil-sozdal-kartu-fermerskih-polej-v-evrope-i-ssha-i-predstavil-eyo-na-producthunt/)

### Inferences
- For **Donbas**, the most directly usable open resources are:
  - the Ukrainian national products: NASA Harvest per-oblast shapefiles 2021–2023, and a DelAnyFlow/DelAny-v2 national map, if downloadable;
  - the Delineate Anything v2 model run on fresh Sentinel-2.
- War damage (craters, burning, abandonment) and occupation-era land-use change mean pre-2022 boundaries may be stale. Time-series abandonment checks (the Akymenko et al. method) are a useful complement.
- Delineate Anything's AGPL-3.0 licence matters for any proprietary or networked deployment.

### Gaps
- Not confirmed whether the NASA Harvest per-oblast files include **Donetsk and Luhansk** (occupied or front-line parts) or Crimea.
- Not confirmed whether the DelAnyFlow/DelAny-v2 Ukraine 2024 vector map is publicly downloadable ("national-scale vector products are public" per the HF page summary, not verified), nor its licence.
- Not confirmed whether **FBIS-73M's 61 countries include Russia or Kazakhstan**: the HF dataset card was unreachable.
- Headline F1/IoU values for Sadeh et al. 2025 and DelAnyFlow were not retrieved.

---

## Q3. Which commercial/operational services map field boundaries over Russia, and is there free data or an API?

### Takeaway
Russia's authoritative field/agricultural-land contours live in the **state ЕФИС ЗСН** (Minselkhoz) and the regional GIS-АПК systems (e.g., Tatarstan's «Агрополия»). They are compiled from farmer submissions, cadastre, satellite and aerial imagery, and contractor digitising. The process is largely manual or semi-automatic, and the data are **not openly downloadable**. Russian commercial farm-management platforms (ExactFarming, АгроСигнал, ScanEx, КБ «Панорама», ЦПС) offer field mapping, mostly as paid services. None was found to publish a free bulk field-boundary dataset for Russia. OneSoil is the only international provider with a field-boundaries API, and Russia is not on its "confident quality" list.

### Cited Findings
**ЕФИС ЗСН (Единая федеральная информационная система о землях сельхозназначения, Minselkhoz) and the federal map-scheme**
- Its remote-sensing module was developed by ООО «ИКИЗ» together with ИКИ РАН. It analyses each **already-entered vector field contour** with Terra/MODIS and Landsat-8/OLI imagery. User reports say contours were often drawn manually by specialists from satellite images, or copied from the public cadastral map — [agrodag.ru "ЕФИС ЗСН практика пользования" (search summary)](https://agrodag.ru/efis-zsn-praktika-polzovaniya/); [Arkhangelsk competence centre](http://xn--29-1lc1c.xn--p1ai/news/edinaya-federalnaya-informacionnaya-sistema-o-zemlyah-selskohozyajstvennogo-naznacheniya-efis-zsn); [pro-vega.ru "Аналитик ДЗЗ ЕФИС ЗСН"](http://pro-vega.ru/efis/)
- Minselkhoz is building a "digital map of all agricultural land, down to the crop on each field". Planned status: 15 crop types recognised in 23 regions, all regions by 2026. A technical specification requires the contractor to introduce AI into ЕФИС ЗСН — [своё.рф (Своё фермерство)](https://xn--e1alid.xn--p1ai/journal/publication/efis-zsn-pozvolit-raspoznavat-selkhozkultury-na-territorii-vseiy-rossii-s-2026-goda); [TAdviser](https://www.tadviser.ru/index.php/%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82:%D0%9C%D0%B8%D0%BD%D0%B8%D1%81%D1%82%D0%B5%D1%80%D1%81%D1%82%D0%B2%D0%BE_%D1%81%D0%B5%D0%BB%D1%8C%D1%81%D0%BA%D0%BE%D0%B3%D0%BE_%D1%85%D0%BE%D0%B7%D1%8F%D0%B9%D1%81%D1%82%D0%B2%D0%B0_(%D0%9C%D0%B8%D0%BD%D1%81%D0%B5%D0%BB%D1%8C%D1%85%D0%BE%D0%B7_%D0%A0%D0%BE%D1%81%D1%81%D0%B8%D0%B8)(%D0%98%D1%81%D0%BA%D1%83%D1%81%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%B8%D0%BD%D1%82%D0%B5%D0%BB%D0%BB%D0%B5%D0%BA%D1%82_(%D0%98%D0%98,_Artificial_intelligence,_AI))
- **Legal status:** agricultural-land (сельхозугодья) boundaries are currently held only in ЕФИС ЗСН, not in the land registry (ЕГРН). A new law brings them into ЕГРН from **1 July 2026**, and Minselkhoz must establish boundaries of all сельхозугодья by **1 Jan 2028** — [polygon-journal](https://polygon-journal.ru/article/614); [Garant/Rosreestr digest Q4 2025](https://www.garant.ru/products/ipo/prime/doc/413445884/)
- Federal map-scheme of agricultural land (федеральная карта-схема земель сельхозназначения):
  - Started 2022; **36 regions covered in 2 years**; completion targeted for end of 2025.
  - Contours determined from archives plus remote sensing.
  - Pilots: Kaliningrad, Moscow, Belgorod oblasts and **Tatarstan**.
  - Contractors included **Geoscan** (Leningrad and Novgorod oblasts). Geoscan cross-checked ЕФИС ЗСН, ЕГРН, ФГИС ТП, ГФДЗ and Рослесинфорг data against satellite and aerial RGB/NDVI imagery and delivered vector-raster layers.
  - No source mentions neural networks for this work.
  - [Geoscan blog](https://www.geoscan.ru/ru/blog/cifrovaya-karta-dlya-ministerstva-selskogo-khozyaystva-rf); [МГУ ЕЦПБ](https://ecfs.msu.ru/news/edinaya-federalnaya-karta-sxema-zemel-selxoznaznacheniya); [Kommersant](https://www.kommersant.ru/doc/7230600); [itrussia.media, 21.08.2026](https://itrussia.media/ru/article/neyroseti-i-sputniki-pomogut-vernut-rossii-milliony-gektarov-zemli-21-08-2026)
- Practical ЕФИС ЗСН quirk: a field crossing a district boundary must be split into two objects — [Agroinvest (search summary)](https://www.agroinvest.com/news/v-poryadke-sila)

**Tatarstan**
- The regional GIS-АПК / «Агрополия» platform (8 projects):
  - Farmers submit field contour, crop and rotation through personal cabinets.
  - The ministry verifies field data using satellite imagery with **AI / machine-vision elements** to confirm ploughing.
  - The pilot in Pestrechinsky and Sabinsky districts reconstructed each field's use history back to 1985.
  - Headline: "96% пашни Татарстана оцифровано" (96% of Tatarstan's arable land digitised).
  - Access is for government and farmers; no open download was found.
  - [business-gazeta.ru](https://www.business-gazeta.ru/article/406008); [glavagronom.ru](https://glavagronom.ru/articles/cifrovaya-transformaciya-regionalnoy-selhoz-gis-kak-uskorit-i-uprostit-rabotu-s-geodannymi-o-polyah); [Innogeotech](https://innogeotech.ru/news/news-all-35/); [YouTube «Агрополия»](https://www.youtube.com/watch?v=GgjNLlyLbQY)

**Russian commercial platforms**
- **ExactFarming** (listed on the «Своё фермерство» farmer platform): its "AI-оценщик" determines boundaries of cultivated land, crop types and five-year yield. It also offers "мониторинг границ поля" (current cultivated boundaries) — [betaren.ru](https://betaren.ru/news/virtualnye-polya-kak-razvivayutsya-tsifrovye-programmy-agrosoprovozhdeniya/?journal_id=9033); [svoefermerstvo.ru](https://svoefermerstvo.ru/services/exact-farming)
- **АгроСигнал**: sells "картирование полей и электронные карты полей", replacing document boundaries with actual contours. The method (GPS survey, imagery or manual) is not disclosed — [agrosignal.com](https://agrosignal.com/resheniya/kartirovanie/)
- **ScanEx (СКАНЭКС)**: GeoMixer-based agro services include updating field maps. ScanEx Image Processor has an automatic vectorisation function; no field-specific automatic delineation product was confirmed — [scanex.ru agriculture](https://www.scanex.ru/thematic/economic/selskoe-khozyaystvo/); [ScanEx Image Processor](https://www.scanex.ru/software/image-processing/scanex-image-processor/)
- **КБ «Панорама» — Panorama Vision**: automatic recognition and vectorisation that outputs vector maps with contours of recognised objects, **including agricultural field boundaries**, from 2–10 m RGB/multispectral imagery (commercial software) — [gisinfo.ru](https://gisinfo.ru/products/panvision.htm)
- **ГК «Центр Программ Систем»**: a geo-service combining GIS, satellite imagery and AI. It analyses 10 m imagery and NDVI against past years for remote land control by regional ministries — [itrussia.media](https://itrussia.media/ru/article/neyroseti-i-sputniki-pomogut-vernut-rossii-milliony-gektarov-zemli-21-08-2026)
- Field-map contractors (e.g., sasagro, Geomir) describe building electronic field maps from GPS ground surveys, UAV imagery (to about 20 cm) and 0.8–2 m satellite imagery — [geomir.ru](https://www.geomir.ru/publikatsii/elektronnye-karty-poley/); [sasagro.ru](https://sasagro.ru/digital-cartography)

**International services**
- **OneSoil**:
  - Field Boundaries API with automatically detected boundaries for **59 countries**; the "data on some regions is available free".
  - The app shows boundaries in 57 countries.
  - Quality-confident list: Europe, US, Canada, Argentina, Brazil, South Africa, Australia, **Kazakhstan**. For other countries OneSoil starts with a paid proof-of-concept.
  - Boundary downloads in the Scouting app are a **subscription** feature. Built on open Sentinel-1/2.
  - [b2b.onesoil.ai field-boundaries-api](https://b2b.onesoil.ai/ru/field-boundaries-api); [App Store listing](https://apps.apple.com/us/app/onesoil-scouting-farming-tool/id1458134040); [OneSoil blog](https://blog.onesoil.ai/en/onesoil-map)
- **Varda Global FieldID**: published coverage covers EU countries, the US, Brazil and Japan (27.8 M boundaries added in 2024), with an OAuth2 API. **No evidence of Russia/Ukraine/Kazakhstan coverage** — [Varda](https://www.varda.ag/global-field-id); [developer docs](https://developer.varda.ag/docs/fid); [X post](https://x.com/Varda_FieldData/status/1813136727831130252)
- **Google ALU/AMED**: India, Malaysia, Vietnam, Indonesia, Japan, plus six African countries added in Aug 2026. **Not Russia/CIS**; API access for trusted testers, no open dataset — [newspaceeconomy.ca](https://newspaceeconomy.ca/2026/09/25/how-could-googles-agricultural-landscape-understanding-reshape-agricultural-intelligence-from-space/); [agri.withgoogle.com](https://agri.withgoogle.com/)
- **Planet (Sinergise) Field Delineation**: a commercial Sentinel-2 deep-learning marker that can be fine-tuned on local data. Sinergise produced a Ukraine layer (2.66 M fields, cited in DelAnyFlow) — [Planet docs](https://docs.planet.com/data/area-monitoring/markers/field-delineation/); [Sentinel Hub blog](https://medium.com/sentinel-hub/automatic-field-delineation-new-release-1c2938399f0)

### Inferences
- For Samara and Tatarstan, ЕФИС ЗСН and the regional GIS-АПК hold the most authoritative contours. They are suitable as **training/validation labels if access can be obtained** (e.g., via the regional ministry or a farmer's own cabinet), but they are not open data.
- Western commercial services (Varda, Planet, Google) appear not to serve Russia. OneSoil's Russian coverage is unclear and not quality-assured.

### Gaps
- No primary source describes **how ЕФИС ЗСН contours were generated technically** (algorithm, imagery, share of manual digitising) or their positional accuracy.
- Not verified: Cropwise (Syngenta), Smart Agro, Kosmosnimki-Agro, Sber's role in ЕФИС ЗСН, and whether any of them auto-delineates fields over Russia. Nothing specific was found.
- Not verified whether OneSoil's 59-country set includes Russia, Ukraine or Belarus, or whether the company still operates in Russia.
- No information found on ЕФИС ЗСН digitisation status in the occupied Donetsk/Luhansk territories.

---

## Q4. Which global products cover Russia, and what do they actually provide there?

### Takeaway
The only **open, wall-to-wall field-boundary product covering Russia** is the **Fields of The World (FTW) Global map** (Microsoft AI for Good, Taylor Geospatial, ASU, WashU and others; 2024 and 2025; Sentinel-2 at 10 m; CC-BY). Its model saw **no Russian or Central Asian training data**, so quality there is unvalidated and the 500 m confidence layer should be used. Delineate Anything v2 is the main open **model** to run yourself. Other global layers (WorldCereal, GLAD cropland, FSU arable/abandoned maps) give cropland masks, not parcels.

### Cited Findings
- **FTW Global field boundary map** (Robinson, Muhawenayo, Khanal, Fang, Corley, Tárano, Estes, Marcus, Jacobs, Kerner, Becker-Reshef, Lavista Ferres; arXiv 2605.11055, 2026):
  - **3.17 billion polygons across 241 countries and territories** for 2024 and 2025.
  - Model: the PRUE model (U-Net with EfficientNet-B7, 3-class interior/boundary/background) run on two cloud-free Sentinel-2 L2A mosaics per year (planting and harvest, timed with the WorldCereal crop calendar).
  - Trained on the CC-BY subset of FTW: 1.6 M polygons, 24 countries, bi-temporal RGBN.
  - Released under **CC-BY** at source.coop/ftw/global-data, with code at github.com/fieldsoftheworld/ftw-baselines.
  - [FTW Global Map PDF (read in full)](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf); [source.coop](https://source.coop/ftw/global-data)
- FTW limitations stated by the authors:
  - Training is heavily European (17 of 24 countries). **"North America, China, Russia and Central Asia… are entirely absent from training"**, as are boreal cropping and semi-arid irrigated systems.
  - Annual crops only (pasture, orchards and perennials missed).
  - Polygons are "remote-sensing field units", **not cadastral parcels**.
  - [FTW Global Map PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- FTW accuracy:
  - Mean pixel recall 0.852 over 24 countries (14 countries above 0.90).
  - Full-country F1: Austria 0.89, Latvia 0.88, **Finland 0.74**. In Finland performance degraded with latitude, from F1 0.83 at about 60°N to below 0.13 in Lapland, where boreal forest was over-predicted.
  - 500 m confidence layer AUC about 0.82–0.84.
  - [FTW Global Map PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- Source.coop lists about 1.63 B (2024) and 1.58 B (2025) polygons across 195 countries/territories, which differs from the paper's 241 — [search summary of source.coop](https://source.coop/ftw/global-data)
- PRUE recipe paper (arXiv 2603.27101): U-Net + EfficientNet with log-cosh Dice loss for large-scale field segmentation — [arXiv](https://arxiv.org/pdf/2603.27101)
- **Delineate Anything v2**: open weights (AGPL-3.0) and a resolution-agnostic instance-segmentation model; see Q2. Asia & Oceania mAP@0.5 = 0.44 and Europe 0.61; Russia is not reported separately — [GitHub README](https://github.com/Lavreniuk/Delineate-Anything)
- **ESA WorldCereal 2021**:
  - Global 10 m temporary-cropland extent, crop types (maize, winter and spring cereals) and irrigation, processed per **106 agro-ecological zones**.
  - Spring cereals covered in only 21 AEZs.
  - ~237 GB on Zenodo. Cropland and crop-type masks only, **no boundaries**.
  - [Zenodo 7875105](https://zenodo.org/records/7875105); [BELSPO news](https://eo.belspo.be/news/worldcereal-releases-global-seasonal-crop-type-maps)
- **UMD GLAD global cropland** (Potapov et al.): 30 m Landsat cropland extent for 2003, 2007, 2011, 2015 and 2019. **Russia had the largest net cropland reduction 2000–2019**, with abandonment concentrated in the Nechernozemie. A cropland mask, no boundaries — [GLAD croplands](https://glad.umd.edu/dataset/croplands); [ResearchGate](https://www.researchgate.net/publication/357292941_Global_maps_of_cropland_extent_and_change_show_accelerated_cropland_expansion_in_the_twenty-first_century)
- "Spatial distribution of arable and abandoned land across former Soviet Union countries", *Scientific Data* (2018): an open FSU arable/abandoned dataset. Details not retrieved — [Nature sdata201856](https://www.nature.com/articles/sdata201856)
- Greenpeace Russia "Карта неиспользуемых сельхозземель" (map of unused agricultural land, 2018): Landsat/GEE-based abandoned-land mapping. It estimates 4,307 thousand ha of сельхозугодья (incl. 3,274 thousand ha arable) suitable for forest — [maps.greenpeace.org](https://maps.greenpeace.org/maps/aal/); [business-gazeta](https://www.business-gazeta.ru/news/425877)
- Open training sets for boundaries are all outside Russia: AI4Boundaries (Sentinel-2 + aerial, about 2.5 M parcels, EU), FTW (24 countries), FBIS-22M/73M — [ESSD AI4Boundaries](https://essd.copernicus.org/articles/15/317/2023/)

### Inferences
- For Samara and Tatarstan, FTW Global 2024/2025 is a **free baseline layer** to download and audit immediately. The Finland latitude result suggests that in Tatarstan (about 55–56°N) forest–field mosaics may be over-predicted, so the FTW confidence layer and a cropland mask (WorldCereal/GLAD) should be used to filter.
- Samara (about 52–54°N, steppe/forest-steppe with large fields) is likely closer to FTW's well-performing European conditions. This is unverified.

### Gaps
- No published validation of FTW Global or DelAny-v2 **inside Russia**. FTW's confidence layer is model-based and was not validated against Russian ground truth.
- I did not confirm whether WorldCereal's 2021 AEZ products cover every Russian AEZ with crop types (cropland extent is described as global).

---

## Q5. Which methods were shown to work in Russian conditions (large fields, shelterbelts, fallows, abandoned land)?

### Takeaway
Published evidence specific to Russia is thin. Three things have been shown:
1. ИКИ's **multi-temporal segmentation** and semi-automatic seeded contouring work operationally.
2. **Multi-temporal Sentinel-2 composites with transformer segmentation** (UPerNet + Swin-V2) worked best for arable boundaries in Samara oblast.
3. **Time-series (multi-year) analysis** is the established way to separate abandoned land and fallows from active fields (ИКИ, Prishchepov, Samara–SAMIS, Dnipro).

The war-zone Ukrainian and Kazakh work points the same way: multi-date inputs plus instance segmentation, with abandonment handled as a separate time-series step.

### Cited Findings
- Multi-temporal input beats single-date for arable-boundary extraction in Samara oblast: five-month vegetation-change composites; UPerNet + Swin-V2 best — [Bavrina & Agafonov 2025](https://computeroptics.ru/KO/PDF/KO49-6/490616.pdf)
- ИКИ segments **time series** of remote sensing data into "dynamically homogeneous" areas, and uses seasonal reconstructed Landsat series for cropland masks — [Plotnikov et al. 2018 (cited by)](https://www.sciencedirect.com/science/article/abs/pii/S0167865511004119); [ИКИ conference index](https://ikiconf.txldev.ru/files/books/2022/9535.htm)
- Semi-automatic seeded segmentation of image series, followed by vectorisation and manual correction, is the operational ИКИ approach in VEGA-Science — [Vega-Science paper (search summary)](https://www.researchgate.net/publication/357760327_Vega-Science_system_design_features_main_capabilities_and_usage_experience)
- Abandoned or unused land:
  - Samara University–SAMIS: cadastre plus imagery, about 95% accuracy — [ТАСС](https://nauka.tass.ru/nauka/25812761)
  - Dnipro, NDVI peak comparison: F1 0.898 in Donetsk/Dnipropetrovsk — [CEUR](https://ceur-ws.org/Vol-4048/paper24.pdf)
  - Abandonment rates in European Russia depend strongly on sensor resolution (Landsat 37% vs MODIS 22.5%) — [Prishchepov et al. (search summary)](http://spatial-economics.com/eng/arkhiv-nomerov/2013/38-2013-3/570-prishchepov-av-mller-d-dubinin-m-baumann-m-radeloff-vc-determinants-of-the-spatial-distribution-of-abandoned-agricultural-lands-in-the-european-part-of-russia)
- Resolution limits: about 25 pixels per field is the practical lower bound, and Sentinel-2 detects fields down to about 0.5 ha — [DelAnyFlow (search summary)](https://arxiv.org/pdf/2511.13417)
- Instance segmentation (DelAny) outperformed semantic boundary approaches on country-scale Ukraine, especially for 0.25–1 ha fields. Bi-temporal (planting + harvest) input with a 3-class U-Net is the FTW global recipe — [GitHub](https://github.com/Lavreniuk/Delineate-Anything); [FTW PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- A small GitHub prototype for an area around 53.3–53.6°N, 38.2–38.95°E (central European Russia) used classical Random Forest pixel classification (vegetation vs bare soil) on Sentinel-1/2, Sobel edges on B08 thresholded at the 88th percentile, and morphology. It reports **mIoU 0.793 / macro-F1 0.881 for pixel classes, not boundary accuracy** — [hartid/satellite-boundaries README (read directly)](https://github.com/hartid/satellite-boundaries)

### Inferences
- Russian arable fields in Samara and Tatarstan are typically large (tens to hundreds of ha) and separated by roads and **shelterbelts (лесополосы)**. These are well above Sentinel-2's about 0.5 ha detection limit, so resolution is unlikely to be the binding constraint. The harder problems are likely:
  - (a) adjacent fields under the same crop with no visible separator, which multi-date input helps with;
  - (b) shelterbelts and woody strips being merged into fields or splitting them;
  - (c) fallows and abandoned land that look like grassland.

  This is inference from the cited limits; no source tested shelterbelts explicitly.
- A pragmatic recipe consistent with the evidence:
  1. Use multi-date Sentinel-2 (spring/planting plus summer/harvest) to run, or fine-tune, an open model (FTW/PRUE U-Net or Delineate Anything v2).
  2. Filter with a cropland or abandonment mask derived from multi-year NDVI time series.
  3. Use ЕФИС ЗСН/regional contours, if obtainable, for fine-tuning and validation.

### Gaps
- No study was found that quantifies boundary errors caused by **shelterbelts** or treats them explicitly in Russia or Ukraine.
- No Russian study comparing classical (ИКИ segmentation) and neural approaches on the same area was found.

---

## Q6. Is there any open Russian field-boundary dataset or open code?

### Takeaway
**No open, Russia-specific field-boundary dataset was found**: the state ЕФИС ЗСН and regional GIS-АПК are closed, and commercial products are paid. The open options covering Russia are global: **FTW Global 2024/2025 polygons (CC-BY)** and **Delineate Anything v2 weights/code (AGPL-3.0)**, plus FTW baselines code. For Ukraine (Donbas), open per-oblast boundary data exist (NASA Harvest 2021–2023, CC-BY-4.0). No open Russian-authored field-delineation code of research quality was found; the only Russian-language GitHub repo found is a small hobby prototype.

### Cited Findings
- FTW Global (CC-BY), covering Russia — [FTW PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf); [source.coop](https://source.coop/ftw/global-data)
- FTW training data and code: source.coop/kerner-lab/fields-of-the-world; github.com/fieldsoftheworld/ftw-baselines — [FTW PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- Delineate Anything v1/v2: code AGPL-3.0, weights on HF (DelineateAnything.pt / DelineateAnythingv2.pt, 125 MB; small "S" variant 17.6 MB for v1), FBIS-73M dataset on HF, TorchGeo integration — [GitHub README](https://github.com/Lavreniuk/Delineate-Anything)
- FBIS-22M licence was flagged as **unspecified** by the FTW datasets-list community — [GitHub issue](https://github.com/fieldsoftheworld/ftw-datasets-list/issues/6)
- NASA Harvest Ukraine field boundaries 2021–2023, CC-BY-4.0 — [Zenodo](https://zenodo.org/records/12791903)
- hartid/satellite-boundaries: a Russian-language prototype (RF + Sobel + morphology) with an output GPKG of 1,100 polygons for one scene; no licence was observed in the README — [GitHub](https://github.com/hartid/satellite-boundaries)
- ЕФИС ЗСН boundaries are held in a state information system; farmers edit their own fields through a cabinet or requests to municipalities. No public download was found — [своё.рф](https://xn--e1alid.xn--p1ai/journal/publication/peredavat-dannye-v-efgis-zsn-budut-8-ministerstv-i-vedomstv); [aclsx.ru instructions](https://aclsx.ru/efgis_zsn)

### Inferences
- Once agricultural-land boundaries enter **ЕГРН (from 1 July 2026)**, some contour information may become visible through the public cadastral map. These would be legal land-use units, not observed field units, and bulk download is restricted in practice. This is speculative and should be monitored.
- For the end user, the realistic open path for Russia is: start from FTW Global and/or Delineate Anything v2, then fine-tune with locally digitised labels (Samara, Tatarstan) and with Ukrainian open labels for Donbas.

### Gaps
- GitHub-wide search was unavailable in this environment, so smaller Russian repos (e.g., hackathon solutions, ODS/Kaggle competitions on Russian fields) may exist but were not found.
- Not determined whether any Russian agricultural-field benchmark dataset has been published (e.g., on Kaggle or Hugging Face).

---

## Q7. What accuracy levels are reported for Russia (and the region)?

### Takeaway
**No peer-reviewed boundary-level accuracy (object F1/IoU) for field delineation in Russia was retrieved.** The figures available are:
- Russia: about 95% (press release) for abandoned-parcel detection in Samara.
- Ukraine and Donbas: F1 0.898 for abandoned-land detection; 92.5% overall accuracy.
- Global or proxy results relevant to Russia: FTW mean pixel recall 0.85 globally, and F1 0.74 in Finland (a boreal analogue for northern Russia); DelAny-v2 global mAP@0.5 0.559.
- Kazakhstan: automatic boundaries were "slightly less accurate" than hand-digitised ones for crop statistics.

### Cited Findings
- Samara University–SAMIS abandoned-land service: about 95% detection accuracy (press release; no metric definition) — [ТАСС](https://nauka.tass.ru/nauka/25812761)
- Bavrina & Agafonov 2025 (Samara, Sentinel-2): numeric results not retrieved; "best quality with UPerNet + Swin-V2" — [Computer Optics](https://computeroptics.ru/KO/PDF/KO49-6/490616.pdf)
- Dnipro University of Technology, Donetsk/Dnipropetrovsk: accuracy up to 92.5%, F1 0.898 (cultivated vs abandoned) — [CEUR-WS](https://ceur-ws.org/Vol-4048/paper24.pdf)
- FTW Global: pixel recall 0.852 (mean, 24 countries); F1 Austria 0.89, Latvia 0.88, Finland 0.74 (precision 0.65, recall 0.86); Russia not validated — [FTW PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/04/FTW_Global_Map_v1.1.pdf)
- Delineate Anything v2: global mAP@0.5 0.559 (precision 0.639, recall 0.525); Europe 0.612; Asia & Oceania 0.440 — [GitHub README](https://github.com/Lavreniuk/Delineate-Anything)
- DelAnyFlow, Ukraine: field counts (3.75 M at 5 m, 5.15 M at 2.5 m) versus Sinergise 2.66 M and NASA Harvest 1.69 M, but no F1 retrieved — [arXiv 2511.13417 (search summary)](https://arxiv.org/pdf/2511.13417)
- Southern Kazakhstan: hand-digitised boundaries gave better agreement with official crop statistics than automatic ones — [Agronomy 15(9):2040](https://doi.org/10.3390/agronomy15092040)
- Russian GitHub prototype (central Russia): pixel mIoU 0.793 for vegetation/bare-soil classes, not a boundary metric — [hartid/satellite-boundaries](https://github.com/hartid/satellite-boundaries)

### Inferences
- Without Russian validation, expected quality for Samara and Tatarstan can only be bracketed by analogues:
  - European lowland agriculture: F1 about 0.88–0.89 (Austria/Latvia, FTW).
  - Mixed forest–field, higher latitude: F1 about 0.74 (Finland, FTW).

  The end user should plan an **independent validation sample** (for example, a few hundred fields digitised from high-resolution imagery, or ЕФИС/farm GNSS contours) in each target region.

### Gaps
- Full-text metrics from the Russian papers (Computer Optics 2018/2025; Исследование Земли из космоса 2023; Успехи современного естествознания) were not accessible.
- No object-level accuracy statistics for ЕФИС ЗСН or the federal map-scheme contours were found.
