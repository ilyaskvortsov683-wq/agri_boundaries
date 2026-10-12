exec(open('split_proto.py').read().split("P0 = dict")[0])
P = dict(SC=1.3, SA=4, E_MIN=6.3, R_MIN=3.15, E_MAX=9.1, HYS_LO=0.6, EDGE_PX=2, LINE_MIN_PX=30, LINE_DIL=2, PIECE_MIN_HA=8, MIN_HA=40)
for k in ['W1','W2','W3','W4','W5','W6']:
    f = gpd.read_file(f'v3run/out_P{k}/fields_after_stage1_full.gpkg')
    f['area_ha'] = [x.area*px_area_ha for x in to_px(f)]
    e0 = evaluate(f); out, ns = split_layer(f, P); e1 = evaluate(out)
    print(f"{k} | raw n {e0['n']} IoU {e0['IoU']} merged {e0['merged']} frag {e0['frag']} bF {e0['bF']} pxF1 {e0['pxF1']} || split n {e1['n']} IoU {e1['IoU']} IoU70 {e1['IoU70']} merged {e1['merged']} frag {e1['frag']} rec {e1['recall']} bF {e1['bF']} pxF1 {e1['pxF1']}", flush=True)
