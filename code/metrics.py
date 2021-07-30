import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def evaluate(model, test_loader, threshold): 
    loss_sum = list()
    pre_list = list()
    ac_list = list()
    results = dict()
    for data in test_loader:
        data = data[0].cuda()
        n = data[:, 0]
        d_i = data[:, 1]
        d_j = data[:, 2]
     
        prediction_i, prediction_j, loss = model(n, d_i, d_j) 
        loss_sum.append(loss.item())  

        pre_list.extend(prediction_i.tolist())
        pre_list.extend(prediction_j.tolist())
        ac_list.extend([1 for i in range(len(prediction_i))])
        ac_list.extend([0 for i in range(len(prediction_j))])

    test_loss=round(np.mean(loss_sum),4)

    ac_list = np.array(ac_list)
    pre_list = np.array(pre_list)
    results['AUC'] = roc_auc_score(ac_list, pre_list)
    precision, recall, trds = precision_recall_curve(ac_list, pre_list)
    results['AUPR'] = auc(recall, precision)

    pre_list = np.where(pre_list>=threshold, 1, 0)
    results['ACC.'] = accuracy_score(ac_list, pre_list)
    results['P.'] = precision_score(ac_list, pre_list)
    results['R.'] = recall_score(ac_list, pre_list)
    results['F1'] = f1_score(ac_list, pre_list)

    # PR-curse-recommend-thresold
    index = np.arange(0,len(precision))
    e_index = index[precision==recall][0]
    e_value = precision[e_index]
    # e_point = [e_value, e_value]
    print('PR-threshold:%.2f' % trds[e_index])

    return results, test_loss
