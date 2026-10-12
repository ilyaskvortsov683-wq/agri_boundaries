import sys
exec(open('gt_eval.py').read())
f = gpd.read_file(sys.argv[1]); out = sys.argv[2]
win = tuple(int(v) for v in sys.argv[3:7]) if len(sys.argv) > 6 else (0,0,W,H)
x0,y0,x1,y1 = win
rgb = np.clip(FEATS[y0:y1, x0:x1] / 255, 0, 1)
fig, ax = plt.subplots(figsize=(15, 15*(y1-y0)/(x1-x0)))
ax.imshow(rgb, extent=(x0,x1,y1,y0), interpolation='nearest')
cols = {'main':'yellow','small':'cyan','yellow':'red','input':'white'}
for k,c in cols.items():
    for gm in to_px(f[f.kind==k]):
        for pl in (gm.geoms if gm.geom_type=='MultiPolygon' else [gm]):
            ax.plot(*pl.exterior.xy, color=c, lw=1.4)
            for r in pl.interiors: ax.plot(*r.xy, color='white', lw=0.7)
for gm in GT_PX:
    for pl in (gm.geoms if gm.geom_type=='MultiPolygon' else [gm]):
        ax.plot(*pl.exterior.xy, color='lime', lw=0.8, ls='--')
ax.set_xlim(x0,x1); ax.set_ylim(y1,y0); ax.set_axis_off(); plt.tight_layout(); plt.savefig(out, dpi=65)
