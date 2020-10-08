import torch
#假设bbox1的维度是[N,4],bbox2的维度是[M,4]
#1.首先计算两个box左上角坐标的最大值和右下角坐标的最小值
#2.计算交集面积
#3.计算并值面积
def Iou(bbox1,bbox2):
    N = bbox1.size(0)
    M = bbox2.size(0)
    left_top_max = torch.max(
        bbox1[:,:2].unsqueeze(1).expand(N,M,2), #(N,4)->(N,1,4)->(N,M,4)
        bbox2[:,:2].unsqueeze(0).expand(N,M,2) #(M,4)->(1,M,4)->(N,M,4)
    )
    right_down_min = torch.min(
        bbox1[:,2:].unsqueeze(1).expand(N,M,2),
        bbox2[:,2:].unsqueeze(0).expand(N,M,2)
    )
    width_height = right_down_min-left_top_max
    
    width_height[width_height<0] = 0 #两个bbox没有交集
    inner = width_height[:,:,0]*width_height[:,:,1]
    
    area1 = (bbox1[:,2]-bbox1[:,0])*(bbox1[:,3]-bbox1[:,1]) #(N,)
    area2 = (bbox2[:,2]-bbox2[:,0])*(bbox2[:,3]-bbox2[:,1]) #(M,)
    area1 = area1.unsqueeze(1).expand(N,M)
    area2 = area2.unsqueeze(0).expand(N,M)
    union = area1+area2-inner
    print(union)
    iou = inner.div(union)
#     iou = torch.div(inner,union)
    
    return iou
    
    
