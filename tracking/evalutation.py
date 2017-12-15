import json
import numpy as np
import matplotlib.pyplot as plt

from gen_config import gen_config
from run_tracker import run_mdnet, calc_iou_graph

otb_path="/home/wpark/OTB/"

threshold = None
cum_iou = []

with open("%s/otb50.txt"%otb_path) as f:
    seq_list = f.readlines()

    i = 0
    for seq in seq_list:
        if i >= 1:
            break
        i += 1
        seq = seq.rstrip()
        img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(otb_path, seq, False, False)

        # David(300:770), Football1(1:74), Freeman3(1:460), Freeman4(1:283).
        if seq == "David":
            gt = gt[299:770]
        if seq == "Football1":
            gt = gt[0:74]
        if seq == "Freeman3":
            gt = gt[0:460]
        if seq == "Freeman4":
            gt = gt[0:283]

        result, result_bb, fps, iou = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)

        x, y = calc_iou_graph(iou)
        threshold = x
        cum_iou.append(y)

        # Save result
        res = {}
        res['res'] = result_bb.round().tolist()
        res['type'] = 'rect'
        res['fps'] = fps
        json.dump(res, open(result_path, 'w'), indent=2)

cum_iou = np.array(cum_iou)
cum_iou = np.mean(cum_iou, 0)
plt.plot(threshold, cum_iou)
plt.show()
