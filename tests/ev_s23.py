import sys
exec(open('gt_eval.py').read())
def eval_small(fl, kinds=('small','main'), lo=1.0, hi=15.5):
    """Совпадение с эталонными розовыми/тёмными полями lo..hi га: detection IoU>=0.3, и точность новых полигонов"""
    pid = idmap(fl); npp = np.bincount(pid.ravel(), minlength=len(fl)+1)
    ids = np.flatnonzero((_gt.cls == 'pd') & (_gt.ha >= lo) & (_gt.ha < hi)) + 1
    ov = np.zeros((len(fl)+1, len(_gt)+1), np.int64); m = (pid>0)&(GT_ID>0); np.add.at(ov, (pid[m], GT_ID[m]), 1)
    det3 = det5 = 0; ious = []
    for k in ids:
        v = ov[1:, k]; j = int(v.argmax())+1 if v.max()>0 else 0
        if j == 0: ious.append(0); continue
        iou = ov[j,k]/(_npx[k-1]+npp[j]-ov[j,k]); ious.append(iou); det3 += iou>=0.3; det5 += iou>=0.5
    return len(ids), det3, det5, float(np.mean(ious))
def prec_new(new):
    if len(new)==0: return 0,0,0
    pid = idmap(new); inside = np.bincount(pid[GT_ANY].ravel(), minlength=len(new)+1)[1:]; tot = np.bincount(pid.ravel(), minlength=len(new)+1)[1:]
    fr = inside/np.maximum(tot,1)
    return len(new), round(float((fr>=0.5).mean()),3), round(float(inside.sum()/tot.sum()),3)
if __name__ == '__main__':
    R = (sys.argv[1] if len(sys.argv)>1 else 'v3run/out/')
    for k in (1,2,3):
        f = gpd.read_file(R+f'fields_after_stage{k}_full.gpkg')
        print('stage', k, 'n', len(f), 'small GT (n, det>=.3, det>=.5, meanIoU):', eval_small(f))
    f3 = gpd.read_file(R+'fields_after_stage3_full.gpkg')
    for kd in ('small','yellow'):
        print(kd, 'новые: (n, доля полигонов ≥50% внутри эталона, доля площади внутри эталона):', prec_new(f3[f3.kind==kd]))
