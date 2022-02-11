import torch
import torch.nn as nn
import resnet
from data_loader import CompareDataset, GaussianDatasetVerify, MixupDatasetVerify, get_cifar10_loader
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import torch.nn.functional as F #233
import torch.optim as optim
from dataset_gen import test

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 10,bias=True)
    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":

    # datapath = "./results10/"
    # modelpath = "./trained_models/last_CIFAR10_MASKED_ResNet18_noise0.001_results10.pt"
    # datapath = "./resultsweight0_0outclass_trainnoise0_2/"
    # modelpath = "./trained_models/last_CIFAR10_MASKED_ResNet18_noise0.2_resultsweight0_0outclass_trainnoise0_2.pt"
    # datapath = "./resultsweight0_0mixupnoise0_2/"
    # modelpath = "./trained_models/last_CIFAR10_MASKED_ResNet18_noise0.2_resultsweight0_0mixupnoise0_2.pt"
    datapath = "./resultsweight0_0mixupnoise0_2alpha0_8/"
    modelpath = "./trained_models/last_CIFAR10_MASKED_ResNet18_noise0.2_resultsweight0_0mixupnoise0_2alpha0_8.pt"

    # modelpath = "./trained_models/last_CIFAR10_Baseline_ResNet18_noise0.04_alpha0.0_results.pt"
    baseline = "generated"
    model_type = "traditional"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = resnet.ResNet18()
    checkpoint = torch.load(modelpath, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.train()

    trans = transforms.Compose([
            transforms.ToTensor(),
        ])
    if baseline == "generated":
        noisy_dataset = CompareDataset(datapath, trans)
    elif baseline == "baseline_noise":
        noisy_dataset = GaussianDatasetVerify(50, 0.04, "train", trans)
    elif baseline == "baseline_mixup":
        noisy_dataset = MixupDatasetVerify(0.8, 50, trans)

    noisy_loader = torch.utils.data.DataLoader(noisy_dataset,
                batch_size= 5000, shuffle=True)

    # _, valid_loader, _ = get_cifar10_loader(50, 500)
    _, _, valid_loader = get_cifar10_loader(50, 1000)

    noise_feature_total = []
    y = []
    if model_type == "traditional":
        # Noise dataset feature activations
        for idx, (noise, org, label) in enumerate(noisy_loader):
            noise, org, label = noise.to(device), org.to(device), label.to(device)
            with torch.no_grad():
                out_noise = model(noise)
                noise_feature = model.penultimate
            # break
            noise_feature_total.append(noise_feature.cpu().numpy())
            y.append(label.cpu().numpy())
            break
        # noise_feature = noise_feature.numpy()
        # y = label.numpy()
        noise_feature = np.concatenate(noise_feature_total)
        y = np.concatenate(y)
        # neigh = KNeighborsClassifier(n_neighbors=9)
        # neigh.fit(noise_feature, y)

        svm_classifier = SVC(C = 1, decision_function_shape='ovo')
        svm_classifier.fit(noise_feature, y)

        total_correct = 0.0
        count = 0.0
        for idx, (valid_X, valid_y) in enumerate(valid_loader):
            valid_X, valid_y = valid_X.to(device), valid_y.to(device)
            with torch.no_grad():
                valid_out = model(valid_X)
                valid_feature = model.penultimate

            valid_feature = valid_feature.cpu().numpy()
            valid_y = valid_y.cpu().numpy()
            # predict_y = neigh.predict(valid_feature)
            predict_y = svm_classifier.predict(valid_feature)
            total_correct += np.sum(predict_y == valid_y)
            count += valid_y.shape[0]

        print(total_correct/count)

    else:
        test(model, device, valid_loader)
        linear_model = LinearModel().to(device)
        optimizer = optim.SGD(linear_model.parameters(), lr= 0.1, momentum=0.9, weight_decay=5e-4)

        for epoch in range(20):
            
            for idx, (noise, org, label) in enumerate(noisy_loader):
                noise, org, label = noise.to(device), org.to(device), label.to(device)
                with torch.no_grad():
                    out_noise = model(noise)
                    noise_feature = model.penultimate.clone().detach()
                
                optimizer.zero_grad()
                output = linear_model(noise_feature)
                loss = F.cross_entropy(output, label)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                total_correct = 0.0
                count = 0.0
                for idx, (valid_X, valid_y) in enumerate(valid_loader):
                    valid_X, valid_y = valid_X.to(device), valid_y.to(device)
                    valid_out = model(valid_X)
                    valid_feature = model.penultimate.clone().detach()

                    output = linear_model(valid_feature)
                    #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    loss = F.cross_entropy(output, valid_y)
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    total_correct += pred.eq(valid_y.view_as(pred)).sum().item()
                    count += valid_y.shape[0]

                print("epoch {}: test_acc {}".format(epoch, total_correct/count))

            







