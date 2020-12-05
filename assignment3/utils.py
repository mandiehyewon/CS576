import cv2
import torch
from torchvision import transforms

def NMS(bboxes, scores, threshold=0.35):
    ''' Non Max Suppression
    Args:
        bboxes: (torch.tensors) list of bounding boxes. size:(N, 4) ((left_top_x, left_top_y, right_bottom_x, right_bottom_y), (...))
        probs: (torch.tensors) list of confidence probability. size:(N,) 
        threshold: (float)   
    Returns:
        keep_dim: (torch.tensors)
    '''
    # import pdb; pdb.set_trace()
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)  
    keep = []
    while order.numel() > 0:
        try:
            i = order[0]
        except:
            i = order.item()
        keep.append(i)

        if order.numel() == 1: break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    keep_dim = torch.LongTensor(keep)

    return keep_dim

def inference(model, image_path, device, VOC_CLASSES):
    """ Inference function
    Args:
        model: (nn.Module) Trained YOLO model.
        image_path: (str) Path for loading the image.
    """
    # load & pre-processing
    image_name = image_path.split('/')[-1]
    image = cv2.imread(image_path)
    h, w, c = image.shape
    
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = transform(torch.from_numpy(img).float().div(255).transpose(2, 1).transpose(1, 0)) #Normalization
    img = img.unsqueeze(0)
    img = img.to(device)

    # inference
    output_grid = model(img).cpu()

    #### YOU SHOULD IMPLEMENT FOLLOWING decoder FUNCTION ####
    # decode the output grid to the detected bounding boxes, classes and probabilities.
    bboxes, class_idxs, probs = decoder(output_grid)
    num_bboxes = bboxes.size(0)

    # draw bounding boxes & class name
    for i in range(num_bboxes):
        bbox = bboxes[i]
        try:
            class_name = VOC_CLASSES[int(class_idxs[i])]
            print (class_name)
        except:
            import pdb; pdb.set_trace()    
        prob = probs[i]

        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, '%s: %.2f'%(class_name, prob), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1,
                    8)

    cv2.imwrite(image_name.replace('.jpg', '_result.jpg'), image)

def decoder(grid):
    """ Decoder function that decode the output-grid to bounding box, class and probability. 
    Args:
        grid: (torch.tensors)
    Returns:
        bboxes: (torch.tensors) list of bounding boxes. size:(N, 4) ((left_top_x, left_top_y, right_bottom_x, right_bottom_y), (...))
        class_idxs: (torch.tensors) list of class index. size:(N,)
        probs: (torch.tensors) list of confidence probability. size:(N,)
    """

    grid_num = 7
    bboxes = []
    class_idxs = []
    probs = []

    ''' 
    complete decoder function here ....
    '''
    
    conf_thresh = 0.1 #set confidence threshold to 0.1
    cell_size = 1./float(grid_num)
    grid = grid.data
    pred_tensor = grid.squeeze(0) # [7, 7, 30]

    conf1 = pred_tensor[:,:,4].unsqueeze(2) # [7,7,1]
    conf2 = pred_tensor[:,:,9].unsqueeze(2)
    conf = torch.cat((conf1,conf2),2)

    conf_mask1 = conf > conf_thresh
    conf_mask2 = (conf==conf.max())
    conf_mask = (conf_mask1 + conf_mask2).gt(0)

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if conf_mask[i,j,b] == 1:
                    print(i,j,b)
                    box = pred_tensor[i,j,b*5:b*5+4]
                    contain_prob = torch.FloatTensor([pred_tensor[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i])*cell_size # up left corner of cell
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())#convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob, cls_index = torch.max(pred_tensor[i,j,10:],0)
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        bboxes.append(box_xy.view(1,4))
                        class_idxs.append(cls_index)
                        probs.append(contain_prob * max_prob)

    if len(bboxes) == 0: # Any box was not detected
        bboxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        class_idxs = torch.zeros(1)
        
    else:
        # import pdb; pdb.set_trace()
        #list of tensors -> tensors
        bboxes = torch.cat(bboxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        # class_idxs = torch.cat(class_idxs,0) #(n,)
        #bboxes = torch.stack(bboxes).squeeze()
        # probs = torch.stack(probs).squeeze()
        class_idxs = torch.stack(class_idxs).squeeze()

    print (bboxes.size())
    keep_dim = NMS(bboxes, probs, threshold=0.35) # Non Max Suppression

    return bboxes[keep_dim], class_idxs[keep_dim], probs[keep_dim]