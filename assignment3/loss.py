import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Loss(nn.Module):
    def __init__(self, grid_size=7, num_bboxes=2, num_classes=20):
        """ Loss module for Yolo v1.
        Use grid_size, num_bboxes, num_classes information if necessary.

        Args:
            grid_size: (int) size of input grid.
            num_bboxes: (int) number of bboxes per each cell.
            num_classes: (int) number of the object classes.
        """
        super(Loss, self).__init__()
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Use this function if necessary.

        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]

        return iou

    def forward(self, pred_tensor, target_tensor):
        """ Compute loss.

        Args:
            pred_tensor (Tensor): predictions, sized [batch_size, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor (Tensor):  targets, sized [batch_size, S, S, Bx5+C].
        Returns:
            loss_xy (Tensor): localization loss for center positions (x, y) of bboxes.
            loss_wh (Tensor): localization loss for width, height of bboxes.
            loss_obj (Tensor): objectness loss.
            loss_noobj (Tensor): no-objectness loss.
            loss_class (Tensor): classification loss.
        """

        # Write your code here
        batch = target_tensor.size(0)
        n_elements = self.B * 5 + self.C
        
        # target_tensor = target_tensor.view(batch,-1,n_elements)
        mask_coord = target_tensor[:,:,:,4] > 0
        mask_coord = mask_coord.unsqueeze(-1).expand_as(target_tensor)

        pred_coord = pred_tensor[mask_coord].view(-1, n_elements)
        coord_target = target_tensor[mask_coord].view(-1, n_elements)

        box_pred = pred_coord[:,:self.B*5].contiguous().view(-1, 5)
        box_target = coord_target[:,:self.B*5].contiguous().view(-1, 5)
        boxtarget_size = box_target.size()

        coord_response_mask = torch.cuda.ByteTensor(boxtarget_size)
        coord_response_mask.zero_()
        coord_not_response_mask = torch.cuda.ByteTensor(boxtarget_size)
        coord_not_response_mask.zero_()

        class_pred = pred_coord[:,self.B*5:]
        class_target = coord_target[:,self.B*5:]

        mask_noobj = target_tensor[:,:,:,4] == 0
        mask_noobj = mask_noobj.unsqueeze(-1).expand_as(target_tensor)       
        pred_noobj = pred_tensor[mask_noobj].view(-1, n_elements)
        
        noobj_target = target_tensor[mask_noobj].view(-1,30)
        pred_noobj_mask = torch.cuda.ByteTensor(pred_noobj.size())
        pred_noobj_mask.zero_()
        pred_noobj_mask[:,4]=1
        pred_noobj_mask[:,9]=1
        
        pred_noobj_class = pred_noobj[pred_noobj_mask]
        noobj_target_class = noobj_target[pred_noobj_mask]

        box_target_iou = torch.zeros(boxtarget_size).cuda()
        
        for i in range(0, boxtarget_size[0], self.B):
            box1 = box_pred[i:i+self.B]
            box2 = box_target[i].view(-1,5)

            iou = self.compute_iou(box1[:,:4], box2[:,:4])
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda()
            
            coord_response_mask[i+max_index]=1
            coord_not_response_mask[i+max_index]=0
            
            box_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        box_target_iou = Variable(box_target_iou).cuda()

        box_pred_response = box_pred[coord_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coord_response_mask].view(-1,5)
        box_target_response = box_target[coord_response_mask].view(-1,5)

        #compute losses
        loss_xy = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False)
        loss_wh = F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)
        loss_obj = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],size_average=False)
        loss_noobj = F.mse_loss(pred_noobj_class, noobj_target_class,size_average=False)
        loss_class = F.mse_loss(class_pred, class_target,size_average=False) 

        return loss_xy, loss_wh, loss_obj, loss_noobj, loss_class