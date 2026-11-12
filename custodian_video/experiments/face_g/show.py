import json, sys
for k in sys.argv[1:]:
    d = json.load(open(f'out/{k}.id3.json')); m = json.load(open(f'out/{k}.motion.json')); b = json.load(open(f'out/{k}.boundary.json'))
    for v in d:
        if v == 'frames': continue
        r = d[v]; mo = m.get(v, {})
        print(k, v, ' '.join(f"{x}:{r[x]['reid_rate']*100:.1f}%/{r[x]['mean']:.3f}/{r[x]['max']:.3f}" for x in r),
              '| mouth %.3f blink %.3f yaw %.3f flick %.3f' % tuple(mo.get(x) or 0 for x in ['corr_mouth_open', 'corr_eyeBlink', 'corr_yaw', 'flicker_ratio_vs_original']),
              '| seam', b.get(v, {}).get('dE_flat'))
