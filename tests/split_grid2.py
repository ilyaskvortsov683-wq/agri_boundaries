exec(open('split_proto.py').read().split("P0 = dict")[0])
P0 = dict(SC=1.3, SA=4, E_MIN=9, R_MIN=4.5, E_MAX=13, HYS_LO=0.6, EDGE_PX=2, LINE_MIN_PX=30, LINE_DIL=1, PIECE_MIN_HA=8, MIN_HA=40)
f = gpd.read_file('v3run/out_F1/fields_after_stage1_full.gpkg')
f['area_ha'] = [x.area*px_area_ha for x in to_px(f)]
for s in (0.5, 0.6, 0.7, 0.8):
  for pm in (5, 8, 12):
    for ld in (1, 2):
      for sa in (4, 6):
        ov = dict(E_MIN=9*s, R_MIN=4.5*s, E_MAX=13*s, PIECE_MIN_HA=pm, LINE_DIL=ld, SA=sa)
        P = {**P0, **ov}; out, ns = split_layer(f, P); e = evaluate(out)
        print(f"s{s} pm{pm} ld{ld} sa{sa} | n {e['n']} IoU {e['IoU']} IoU70 {e['IoU70']} frag {e['frag']} merged {e['merged']} rec {e['recall']} bF {e['bF']} pxF1 {e['pxF1']}", flush=True)
