import numpy as np
import cv2 as cv
import torch
from metrics import calculate_dc,calculate_jc
import cv2
from losses import DiceLoss
from losses import FocalLoss
from user_interact_sim import user_interact_sim
import os
from similarity_fast import RelevantMap
Dice_Loss = DiceLoss(2)
Focal_Loss = FocalLoss()

def merge_matrices(a, b):
    both_have_value = a.bool() & b.bool()
    result = torch.zeros_like(a)
    result[both_have_value] = (a[both_have_value] + b[both_have_value]) / 2.0
    either_has_value = a.bool() | b.bool()
    result[either_has_value & ~both_have_value] = a[either_has_value & ~both_have_value] + b[
        either_has_value & ~both_have_value]
    return result


def iteration(X_batch, y_batch,real_batch,iter_num,model,optimizer,img_name,npy_file_path,patchsize,epoch,
              diameter = 15,train = True,method = "scribbles",use_sim=False):
    # ===================forward=====================
    shape = y_batch.shape
    pos = torch.from_numpy(np.zeros(shape, dtype=np.float32)).cuda()
    neg = torch.from_numpy(np.zeros(shape, dtype=np.float32)).cuda()
    cur_seg_result = torch.from_numpy(np.zeros(shape, dtype=np.float32)).cuda()
    real_batch = real_batch.cpu().numpy()
    bs = shape[0]
    img_point_list = [[] for i in range(bs)]



    for i in range(bs):
        img_point_list[i].append([])
        img_point_list[i].append([])

    img_pathSim_list = []

    if use_sim:
        for i in range(bs):
            img_n = img_name[i]
            if train == True:
                path  = os.path.join(npy_file_path,"train/img/",img_n)
            else:
                path = os.path.join(npy_file_path, "test/img/", img_n)

            temp = RelevantMap(path , patch_size=patchsize)
            img_pathSim_list.append(temp)

    for i in range(iter_num):
        pos_hint, neg_hint,pos_simmap,neg_simmap = user_interact_sim(X_batch,cur_seg_result,y_batch,real_batch, img_pathSim_list,img_point_list,patchsize,
                                                                     diameter ,method =method,use_sim=use_sim,epoch=epoch)
        pos = merge_matrices(pos,pos_hint)
        neg = merge_matrices(neg,neg_hint)
        x_with_hint_and_segresult = torch.cat((X_batch, pos.unsqueeze(1),neg.unsqueeze(1),cur_seg_result.unsqueeze(1)), dim=1)

        if train == True:
            output = model(x_with_hint_and_segresult,pos_simmap,neg_simmap)
            tmp2 = y_batch.detach().cpu().numpy()
            tmp = torch.argmax(output, dim=1).detach().cpu().numpy()
            cur_seg_result = torch.argmax(output, dim=1)
            loss = Focal_Loss(output, y_batch.unsqueeze(1))
            dc = calculate_dc(tmp, tmp2)
            jc = calculate_jc(tmp, tmp2)
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if train == False:
            with torch.no_grad():
                output = model(x_with_hint_and_segresult,pos_simmap,neg_simmap)
                tmp2 = y_batch.detach().cpu().numpy()
                tmp = torch.argmax(output, dim=1).detach().cpu().numpy()
                cur_seg_result = torch.argmax(output, dim=1)
                loss = Dice_Loss(output, y_batch.unsqueeze(1))
                dc = calculate_dc(tmp, tmp2)
                jc = calculate_jc(tmp, tmp2)
                # print(i,dc)

            temp1 = pos* (255/np.max( pos.cpu().numpy()))
            temp1 = temp1.cpu().numpy()
            cv.imwrite("pos" + str(i) + ".jpg", temp1[0, :, :])
            temp2 = neg* (255/np.max( neg.cpu().numpy()))
            temp2 = temp2.cpu().numpy()
            cv.imwrite("neg" + str(i) + ".jpg", temp2[0, :, :])
            temp3 = tmp
            temp3[temp3 == 1] = 255
            cv.imwrite("output" + str(i) + ".jpg", temp3[0, :, :])

    return loss,dc,jc,output,tmp,tmp2
