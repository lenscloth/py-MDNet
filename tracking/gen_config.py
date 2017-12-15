import os
import numpy as np

def gen_config(seq_home, seq_name, save_fig, display):

    if seq_name != '':
        # generate config from a sequence name

        save_home = '../result_fig'
        result_home = '../result'

        img_dir = os.path.join(seq_home, seq_name, 'img')
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir,x) for x in img_list]

        with open(gt_path, "r") as f:
            first_line = f.readline()
            if "," in first_line:
                gt = np.loadtxt(gt_path,delimiter=',')
            else:
                gt = np.loadtxt(gt_path)

        init_bbox = gt[0]
        
        savefig_dir = os.path.join(save_home,seq_name)
        result_dir = os.path.join(result_home,seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir,'result.json')
        
    if save_fig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, display, result_path
