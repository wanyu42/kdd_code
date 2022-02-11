import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import resnet
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import *

class CompareDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        # stuff
        self.path = data_dict
        self.transform = transform
        self.label = torch.load(self.path + "label.pt", map_location = torch.device('cpu'))
        self.length = self.label.shape[0]
        self.tt = transforms.ToPILImage()
        
    def __getitem__(self, index):
        # stuff
        batch_idx = index // 128
        img_batch = torch.load(self.path+"images/batch"+str(batch_idx)+".pt", map_location="cpu")
        img = img_batch[index % 128].squeeze()
        org_batch = torch.load(self.path+"images/batch"+str(batch_idx)+"_org.pt", map_location="cpu")
        org = org_batch[index % 128].squeeze()
        img = self.tt(img)
        org = self.tt(org)
        label = self.label[index]

        if self.transform:
            img = self.transform(img)
            org = self.transform(org)

        return (img, org, label)

    def __len__(self):
        return self.length # of how many examples(images?) you have

if __name__=="__main__":
    datapath = "./results0/"
    modelpath = "./trained_models/last_CIFAR10_MASKED_ResNet180.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    dataset = CompareDataset(datapath, trans)
    test_loader = torch.utils.data.DataLoader(dataset,
                batch_size= 3000, shuffle=False)

    _,valid_loader,_ = get_cifar10_loader(50, 500)
    
    model = resnet.ResNet18()
    checkpoint = torch.load(modelpath, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    for idx, (noise, org, label) in enumerate(test_loader):
        noise, org, label = noise.to(device), org.to(device), label.to(device)
        with torch.no_grad():
            out_noise = model(noise)
            feature_noise = model.penultimate
            out_org = model(org)
            feature_org = model.penultimate
        break

    feature_noise = feature_noise.numpy()
    feature_org = feature_org.numpy()
    y = label.numpy()
    # np.where(y == 0)
     
    for idx, (valid, val_label) in enumerate(valid_loader):
        valid, label = valid.to(device), val_label.to(device)
        with torch.no_grad():
            out_valid = model(valid)
            feature_val = model.penultimate
            
        break
    feature_val = feature_val.numpy()
    y_val = val_label.numpy()

    

    feature_combine = np.concatenate([feature_org, feature_noise, feature_val])
    y_combine = np.concatenate([y, y, y_val])

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(feature_combine)
    df = pd.DataFrame()
    df["y"] = y_combine
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    df["noisy"] = feature_org.shape[0]*['origin'] + feature_noise.shape[0]*["noisy"] + feature_val.shape[0]*['origin']
    df['valid'] = feature_org.shape[0]*['train'] + feature_noise.shape[0]*["train"] + feature_val.shape[0]*['valid']

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), style="valid",
            palette=sns.color_palette("hls", 10), 
            data=df).set(title="Origin(0-9) and Noisy(10-19) T-SNE projection")

    plt.show()
    # X_org = []
    # X_noise = []
    # y = []
    # for i in range(5000):
    #     img, org, label = dataset[i]
    #     X_noise.append(img.numpy())
    #     X_org.append(org.numpy())
    #     y.append(label.numpy())
    # y = np.array(y)
    # X_noise = np.concatenate(X_noise)
    # X_org = np.concatenate(X_org)




