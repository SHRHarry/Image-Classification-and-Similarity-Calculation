# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:28:10 2023

@author: ms024
"""

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from classification_dataset import ClassificationDataset

def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomCrop(204),
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])
    
def get_val_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])

def show_dataset(data_loader):
    for images, labels in data_loader:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, 4).permute(1,2,0))
        plt.show()
        break

def accuracy(preds, trues):
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    acc = np.sum(acc) / len(preds)
    
    return (acc * 100)

def RMSE(label, pred):
    label = np.array(label)
    pred = np.array(pred)
    
    return np.linalg.norm(label-pred, ord=2)/len(label)**0.5

def train():
    train_dataset = ClassificationDataset(r"E:\Job\ASUS\LIDL_POC\20230323\cls_6", mode = "train", transforms = get_train_transform())
    val_dataset = ClassificationDataset(r"E:\Job\ASUS\LIDL_POC\20230323\cls_6", mode = "val", transforms = get_val_transform())
    
    train_data_loader = DataLoader(dataset = train_dataset, num_workers = 0, batch_size = 16, shuffle = True)
    val_data_loader = DataLoader(dataset = val_dataset, num_workers = 0, batch_size = 16, shuffle = True)
    # show_dataset(train_data_loader)
    # show_dataset(val_data_loader)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = models.resnet18(pretrained=True)
    # model.fc = nn.Linear(512, 6)
    model.fc = nn.Linear(512, 1)
    model = model.to(device)
    
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    lr = 5e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)  
    
    model.train()
    epochs = 200
    baseline_acc = 1
    for i in range(epochs):
        print(f"===== Epoch = {i} =====")
        test_loss = 0
        correct = []
        size = len(train_data_loader.dataset)
        for batch, (images, labels) in enumerate(train_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            preds = model(images)
            # loss = loss_fn(preds, labels.long())
            loss = loss_fn(preds, labels)
            # print(f"preds = {preds}")
            # print(f"labels = {labels}")
            # print(f"loss = {loss}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(images)
                print(f"lr: {lr:>7f}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        size = len(val_data_loader.dataset)
        num_batches = len(val_data_loader)
        with torch.no_grad():
            for images, labels in val_data_loader:
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                test_loss += loss_fn(preds, labels.long()).item()
                # correct += (preds.argmax(1) == labels).type(torch.float).sum().item()
                correct.append(RMSE(labels.cpu().numpy(), preds.cpu().numpy()))
        
        test_loss /= num_batches
        print(f"Test Error: \n RMSE: {(100*np.mean(correct)):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        lr_scheduler.step()
        
        if np.mean(correct) < baseline_acc:
                torch.save(model.state_dict(), os.path.join("models", f"LIDL_20230426_{np.mean(correct):>0.3f}.pth"))
                baseline_acc = np.mean(correct)

def get_resnet18_feature(model, img):  
    x = model.conv1(img)
    x = model.bn1(x)
    x = nn.ReLU(inplace=True)(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    feature = model.avgpool(x)
    x = torch.flatten(feature, 1)
    prediction = model.fc(x)
    
    feature_np = feature.detach().cpu().numpy()

    return feature_np.reshape(-1, 1), prediction

def db_init(model):
    db_dir = r"./db_imgs"
    img_list = os.listdir(db_dir)
    
    feature_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for img_name in img_list:
        img_path = os.path.join(db_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = T.ToTensor()(img).unsqueeze(0)
        
        with torch.no_grad():
            img = img.to(device)
            pred_np, _ = get_resnet18_feature(model, img)
        feature_list.append(pred_np)
    
    return feature_list

def uncorrect(pred_reg, label, cnt):
    if(pred_reg<0):
        pred_reg = 0
    elif(pred_reg>5):
        pred_reg = 5
        
    if abs(pred_reg - label) > 0.5:
        cnt += 1
    
    return cnt
         
def evaluate():
    model = models.resnet18(pretrained=False)
    # model.fc = nn.Linear(512, 6)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load(r"E:\Job\ASUS\LIDL_POC\20230323\classifiaction\models\LIDL_20230426_0.444.pth"))
    model = model.cuda()
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dir = r"E:\Job\ASUS\LIDL_POC\20230323\cls_6\val"
    class_list = os.listdir(test_dir)
    
    avg_correct = []
    
    for class_idx in class_list:
        label = int(class_idx)
        img_dir = os.path.join(test_dir, class_idx)
        img_list = os.listdir(img_dir)
    
        correct = 0
        cnt = 0
    
        for img_name in img_list:
            image = cv2.imread(os.path.join(img_dir, img_name))
            image = cv2.resize(image, (224,224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = T.ToTensor()(image).unsqueeze(0)
        
            with torch.no_grad():
                image = image.to(device)
                # prediction = model(image)
                # predicted_class = prediction.argmax()
                prediction = model(image)
                predicted_class = prediction
            
            pred_reg = predicted_class[0][0].cpu().numpy()
            cnt = uncorrect(pred_reg, label, cnt)
            print(f"fill level = {pred_reg}, label = {label}, img_name = {img_name}")
        print(f"cnt = {cnt}")
        #     correct += (predicted_class == label).type(torch.float).sum().item()
        
        # correct /= len(img_list)
        # avg_correct.append(correct)
        # print(f"class_idx = {class_idx}, Accuracy: {(100*correct):>0.1f}%")
    
    # avg_correct = np.mean(avg_correct)
    # print(f"Avarage Accuracy: {(100*avg_correct):>0.1f}%")

def infer():
    model = models.resnet18(pretrained=False)
    # model.fc = nn.Linear(512, 6)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load(r"E:\Job\ASUS\LIDL_POC\20230323\classifiaction\models\LIDL_20230414_78.5.pth"))
    model = model.cuda()
    model.eval()
    
    feature_list = db_init(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    true_target = "0"
    img_dir = os.path.join(r"E:\Job\ASUS\LIDL_POC\20230323\cls_6\test", true_target)
    img_list = os.listdir(img_dir)
    rmse = []
    
    for img_name in img_list:
    
        image = cv2.imread(os.path.join(img_dir, img_name))
        image = cv2.resize(image, (224,224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = T.ToTensor()(image).unsqueeze(0)
    
        with torch.no_grad():
            image = image.to(device)
            pred_np, prediction = get_resnet18_feature(model, image)
            predicted_class = prediction.argmax()
        
        predicted_class = predicted_class.detach().cpu().numpy()
        prob_list = []
        for i, cls_feature in enumerate(feature_list):
            dist = np.dot(pred_np.reshape(-1), cls_feature) / (np.linalg.norm(pred_np) * np.linalg.norm(cls_feature))
            prob_list.append(dist[0])
        
        first_idx = np.argmax(prob_list)
        second_idx = np.argsort(prob_list)[-2]
        
        # if first_idx == 0:
        #     fill_level = 0
        
        # else:
        #     total_val = (first_idx * prob_list[first_idx]) + (second_idx * prob_list[second_idx])
        #     total_cnt = prob_list[first_idx] + prob_list[second_idx]
        #     fill_level = total_val/total_cnt
        #     # fill_level = (fill_level + predicted_class)/2
        
        second_prob = prob_list[second_idx]
        if first_idx == 0:
            fill_level = 0
        else:
            if first_idx > second_idx:
                fill_level = first_idx - (1 - second_prob)
            elif first_idx < second_idx:
                fill_level = first_idx + (1 - second_prob)
            
            fill_level = (fill_level + predicted_class)/2
        
        rmse.append(RMSE([int(true_target)], [predicted_class]))
        # np_image = image[0].cpu().numpy().transpose((1, 2, 0))
        # plt.imshow(np_image, cmap='gray')
        print(f"GT = {true_target}, fill_level = {fill_level:>0.3f}, pred = {predicted_class}, prob_list = {np.array(prob_list)}, name = {img_name}")
        # plt.title(f'fill level: {fill_level:>0.3f} / True Target: {true_target}')
        # plt.show()
    print(f"rmse = {np.mean(rmse)}")

def infer_feature():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 6)
    # model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load(r"E:\Job\ASUS\LIDL_POC\20230323\classifiaction\models\LIDL_20230414_78.5.pth"))
    model = model.cuda()
    model.eval()
    
    feature_list = db_init(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    true_target = "1"
    img_dir = os.path.join(r"E:\Job\ASUS\LIDL_POC\20230323\cls_6\test", true_target)
    img_list = os.listdir(img_dir)
    rmse = []
    
    for img_name in img_list:
    
        image = cv2.imread(os.path.join(img_dir, img_name))
        image = cv2.resize(image, (224,224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = T.ToTensor()(image).unsqueeze(0)
    
        with torch.no_grad():
            image = image.to(device)
            pred_np, prediction = get_resnet18_feature(model, image)
            # print(f"prediction = {prediction}")
            predicted_class = prediction.argmax()
        
        predicted_class = predicted_class.detach().cpu().numpy()
        prob_list = []
        for i, cls_feature in enumerate(feature_list):
            dist = np.dot(pred_np.reshape(-1), cls_feature) / (np.linalg.norm(pred_np) * np.linalg.norm(cls_feature))
            prob_list.append(dist[0])
        
        first_idx = np.argmax(prob_list)
        second_idx = np.argsort(prob_list)[-2]
        
        # if first_idx == 0:
        #     fill_level = 0
        
        # else:
        #     total_val = (first_idx * prob_list[first_idx]) + (second_idx * prob_list[second_idx])
        #     total_cnt = prob_list[first_idx] + prob_list[second_idx]
        #     fill_level = total_val/total_cnt
        #     # fill_level = (fill_level + predicted_class)/2
        
        second_prob = prob_list[second_idx]
        if first_idx == 0:
            fill_level = 0
        else:
            if first_idx > second_idx:
                fill_level = first_idx - (1 - second_prob)
            elif first_idx < second_idx:
                fill_level = first_idx + (1 - second_prob)
            
            fill_level = (fill_level + predicted_class)/2
        
        rmse.append(RMSE([int(true_target)], [predicted_class]))
        # np_image = image[0].cpu().numpy().transpose((1, 2, 0))
        # plt.imshow(np_image, cmap='gray')
        print(f"GT = {true_target}, fill_level = {fill_level:>0.3f}, pred = {predicted_class}, prob_list = {np.array(prob_list)}, name = {img_name}")
        # plt.title(f'fill level: {fill_level:>0.3f} / True Target: {true_target}')
        # plt.show()
    print(f"rmse = {np.mean(rmse)}")

if __name__ == "__main__":
    # train()
    # evaluate()
    # infer()
    infer_feature()