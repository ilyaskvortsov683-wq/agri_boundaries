
# %% 15_stage5_refine
if STAGES[5]:
    f0 = get_input(5)
    crs0 = f0.crs
    if crs0.is_projected:
        WORK_CRS = crs0
    else:
        lon0 = f0.to_crs(4326).total_bounds[[0, 2]].mean()
        WORK_CRS = rasterio.crs.CRS.from_epsg(32600 + int((lon0 + 180) // 6) + 1)
    fw = f0.to_crs(WORK_CRS)
    sel = np.ones(len(fw), bool) if REFINE_KINDS is None else np.array(fw['kind'].isin(REFINE_KINDS).values, dtype=bool)
    if REFINE_AOI_M and RUN_MODE == 'test':
        cx, cy = [(a + b) / 2 for a, b in zip(fw.total_bounds[:2], fw.total_bounds[2:])]
        aoi = shp_box(cx - REFINE_AOI_M / 2, cy - REFINE_AOI_M / 2, cx + REFINE_AOI_M / 2, cy + REFINE_AOI_M / 2)
        sel = sel & np.asarray(fw.geometry.within(aoi).values)
    sub = fw[sel].copy()
    print(f'Этап 5: полигонов к уточнению {len(sub)} из {len(fw)}')
    z, size, tx0, tx1, ty0, ty1 = basemap_tile_range(tuple(sub.total_bounds))
    n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    print(f'zoom {z}: ~{n_tiles} тайлов (≈{n_tiles * 20 / 1024:.0f} МБ при 20 КБ/тайл)')
    assert n_tiles <= REFINE_MAX_TILES, f'{n_tiles} тайлов > REFINE_MAX_TILES — уменьшите область/zoom или поднимите лимит'
    ref = refine_fields(sub, basemap_rgb, res=REFINE_RES_M, block_px=REFINE_BLOCK_PX, margin_px=REFINE_MARGIN_PX,
                        passes=REFINE_PASSES, compactness=REFINE_COMPACT, prior=REFINE_PRIOR,
                        max_area_change=REFINE_MAX_AREA_CHANGE)
    full = pd.concat([fw[~sel].assign(refined=False), ref], ignore_index=True)
    full = gpd.GeoDataFrame(full, geometry='geometry', crs=WORK_CRS).to_crs(CRS)
    FIELDS = add_area(renumber(full))
    save_stage(5, FIELDS)
    # до/после на подложке (центральный участок)
    b = sub.total_bounds
    cxm, cym = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    half = 900
    px = int(2 * half / REFINE_RES_M)
    Tv = Affine(REFINE_RES_M, 0, cxm - half, 0, -REFINE_RES_M, cym + half)
    img = basemap_rgb(Tv, (px, px))
    if img is not None:
        fig, axs = plt.subplots(1, 2, figsize=(18, 9))
        for ax, g_, ttl in zip(axs, (sub, ref), ('до (10 м)', 'после уточнения по подложке')):
            ax.imshow(img, extent=(cxm - half, cxm + half, cym - half, cym + half))
            g_.cx[cxm - half:cxm + half, cym - half:cym + half].boundary.plot(ax=ax, color='yellow', linewidth=0.8)
            ax.set_title(ttl); ax.set_axis_off()
        plt.tight_layout(); plt.savefig(f'{OUT_DIR}/refine_before_after.png', dpi=80); plt.show()
else:
    print('Этап 5 пропущен (STAGES[5]=0)')

# %% 16_stage6_after_refine
# Классификация после уточнения границ (этап 5) — отдельный проход по полям. Если этапы 4 и 6 уже выполнены одним проходом
# (FUSE_4_6), здесь ничего не делается. Если этап 4 выключен, а 6 включён — классификация идёт только здесь.
if STAGES[6] and not FUSE_4_6:
    _, cl, _, SPLIT = field_pass(get_input(6), spots=False, classify=True)
    FIELDS = cl
    save_class_outputs(cl, SPLIT)
elif not STAGES[6]:
    print('Этап 6 пропущен (STAGES[6]=0)')

# %% 17_final
if FIELDS is not None:
    final_path = Path(OUT_DIR) / f'fields_final_{RUN_MODE}.gpkg'
    FIELDS.to_file(final_path, driver='GPKG', layer='fields')
    print('итог:', final_path, '|', len(FIELDS), 'полей |', FIELDS.kind.value_counts().to_dict() if 'kind' in FIELDS.columns else '')
    fig, ax = plt.subplots(figsize=(10, 10))
    win = compare_window(1800)
    rgb = np.clip(np.moveaxis(src.read([1, 2, 3], window=win, out_shape=(3, 1200, 1200)), 0, -1) / 255., 0, 1)
    wt_ = rasterio.windows.transform(win, T)
    xa, ya = wt_ * (0, 0); xb, yb = wt_ * (win.width, win.height)
    ax.imshow(rgb, extent=(xa, xb, yb, ya))
    colors = {'main': 'yellow', 'small': 'cyan', 'yellow': 'orange', 'input': 'white'}
    for k_, c_ in colors.items():
        s_ = FIELDS[FIELDS.kind == k_] if 'kind' in FIELDS.columns else FIELDS.iloc[0:0]
        if len(s_):
            s_.cx[min(xa, xb):max(xa, xb), min(ya, yb):max(ya, yb)].boundary.plot(ax=ax, color=c_, linewidth=0.8, label=k_)
    ax.legend(); ax.set_axis_off(); plt.tight_layout(); plt.savefig(Path(OUT_DIR) / f'overview_{RUN_MODE}.png', dpi=80); plt.show()
