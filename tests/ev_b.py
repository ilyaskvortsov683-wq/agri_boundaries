import sys
exec(open('ev_s23.py').read().split("if __name__")[0])
R = sys.argv[1]
for k in ('1','1b','2','3'):
    f = gpd.read_file(R+f'fields_after_stage{k}_full.gpkg')
    e = evaluate(f[f.kind.isin(['main','input'])]) if k in ('1','1b') else None
    if e: print(k, 'main only:', {a: (float(b) if not isinstance(b,int) else b) for a,b in e.items() if a in ('n','IoU','IoU70','recall','prec','frag','merged','pxF1','bF')})
    if k in ('2','3'):
        print(k, 'all n', len(f), 'small GT (n,det.3,det.5,meanIoU):', [round(float(x),3) for x in eval_small(f)], 'new small/yellow prec:', prec_new(f[f.kind=='small']), prec_new(f[f.kind=='yellow']))
