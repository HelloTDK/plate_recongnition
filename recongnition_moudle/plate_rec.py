from recongnition_moudle.plateNet import myNet_ocr,myNet_ocr_color
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
import sys

def cv_imread(path):  #可以读取中文路径的图片
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            if temp.endswith('.jpg') or temp.endswith('.png') or temp.endswith('.JPG'):
                allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
color=['黑色','蓝色','绿色','白色','黄色']    
# plateName=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
plateName=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危防0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
#plateName=r"#0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
mean_value,std_value=(0.588,0.193)
def decodePlate(preds):
    pre=0
    newPreds=[]
    index=[]
    for i in range(len(preds)):
        if preds[i]!=0 and preds[i]!=pre:
            newPreds.append(preds[i])
            index.append(i)
        pre=preds[i]
    return newPreds,index

def image_processing(img,device):
    img = cv2.resize(img, (168,48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img

def get_plate_result(img,device,model,is_color=False):
    input = image_processing(img,device)
    if is_color:  #是否识别颜色
        preds,color_preds = model(input)
        color_preds = torch.softmax(color_preds,dim=-1)
        color_conf,color_index = torch.max(color_preds,dim=-1)
        color_conf=color_conf.item()
    else:
        preds = model(input)
    preds=torch.softmax(preds,dim=-1)
    prob,index=preds.max(dim=-1)
    index = index.view(-1).detach().cpu().numpy()
    prob=prob.view(-1).detach().cpu().numpy()
   
    
    # preds=preds.view(-1).detach().cpu().numpy()
    newPreds,new_index=decodePlate(index)
    prob=prob[new_index]
    plate=""
    for i in newPreds:
        plate+=plateName[i]
    # if not (plate[0] in plateName[1:44] ):
    #     return ""
    if is_color:
        return plate,prob,color[color_index],color_conf    #返回车牌号以及每个字符的概率,以及颜色，和颜色的概率
    else:
        return plate,prob

def init_model(device,model_path,is_color = False):
    # print( print(sys.path))
    # model_path ="plate_recognition/model/checkpoint_61_acc_0.9715.pth"
    check_point = torch.load(model_path,map_location=device)
    model_state=check_point['state_dict']
    cfg=check_point['cfg']
    color_classes=0
    if is_color:
        color_classes=5           #颜色类别数
    model = myNet_ocr_color(num_classes=len(plateName),export=True,cfg=cfg,color_num=color_classes)
   
    model.load_state_dict(model_state,strict=False)
    model.to(device)
    model.eval()
    return model

def batch_plate_rec(img_dir,output_dir,model_path,is_color=False):
    os.makedirs(output_dir,exist_ok=True)
    model = init_model(device,model_path,is_color=is_color)
    
    rec_num = 0
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = os.path.join(img_dir,img_name)
        
        img = cv2.imread(img_path)
        plate,prob= get_plate_result(img,device,model,is_color=is_color)
        prob = (prob*100).astype(np.int32)
        prob = "_".join([str(prob_elm) for prob_elm in prob])
        # cv2.imwrite(os.path.join(output_dir,f"{plate}_{rec_num}_{prob}.jpg"),img) 
        cv2.imwrite(os.path.join(output_dir,f"{plate}_{rec_num}.jpg"),img) 
        rec_num += 1
        print(f"已识别：{rec_num}，识别结果：{plate}")
        # print(plate,img_path)
        

# model = init_model(device)
if __name__ == '__main__':
   model_path = "./weights/checkpoint_250728.pth"
   img_dir = "/expdata/givap/data/plate_recong/mix/b1"
   output_dir = "./data/output_mix_b1"
   is_color = False
   batch_plate_rec(img_dir,output_dir,model_path,is_color=is_color)
#    image_path ="16685ffc91f809a3c8c45c50ef7e8c6.png"
#    testPath = r"D:\Code\Python\github-project\yolov8-plate-master\imgs"
#    fileList=[]
#    allFilePath(testPath,fileList)
# #    result = get_plate_result(image_path,device)
# #    print(result)
#    is_color = False
#    model = init_model(device,model_path,is_color=is_color)
#    right=0
#    begin = time.time()
   
#    for imge_path in fileList:
#         img=cv2.imread(imge_path)
#         if is_color:
#             plate,_,plate_color,_=get_plate_result(img,device,model,is_color=is_color)
#             print(plate)
#         else:
#             plate,_=get_plate_result(img,device,model,is_color=is_color)
#             print(plate,imge_path)
        
  
        
