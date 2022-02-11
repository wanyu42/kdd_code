from numpy import result_type
import torch
import torch.nn as nn
from dataset_gen import generate_dataset
from data_loader import get_cifar10_loader, CIFAR10
from torchvision import transforms
import torch.nn.functional as F 
import os
import matplotlib.pyplot as plt
import resnet
from temperature_scaling import ModelWithTemperature


class ConvOutHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        self.output = output

    def close(self):
        self.hook.remove()

def init_gen(data, valid_dataset, target, device, start_point, noise_level, alpha):
    if start_point == "gaussian":
            data_r = (data.clone().detach() + noise_level * torch.randn(data.size()).to(device)).requires_grad_(True)

    elif start_point == "inclass":
        data_r = []
        for ii in range(target.shape[0]):
            data_val, _ = valid_dataset[torch.randint(target[ii].item()*50, (target[ii].item()+1)*50, (1,)).item()]
            data_r.append(data_val.to(device))
        data_r = torch.stack(data_r).requires_grad_(True)

    elif start_point == "outclass":
        data_r = []
        out_class = (torch.randint(1,10,(target.shape[0],) ).to(device) + target) % 10

        for ii in range(target.shape[0]):
            data_val, _ = valid_dataset[torch.randint(out_class[ii].item()*50, (out_class[ii].item()+1)*50, (1,)).item()]
            data_r.append(data_val.to(device))
        data_r = torch.stack(data_r).requires_grad_(True)
    
    elif start_point == "outclass_train":
        data_r = []

        for ii in range(target.shape[0]):
            outclass_indx_list = torch.where(target != target[ii])[0]
            random_indx = outclass_indx_list[torch.randint(0, outclass_indx_list.shape[0], (1,))]
            data_init = data[random_indx].squeeze()
            data_r.append(data_init.to(device))
        data_r = torch.stack(data_r).requires_grad_(True)

    elif start_point == "mixup":
        data_r = []
        out_class = (torch.randint(1,10,(target.shape[0],) ).to(device) + target) % 10

        for ii in range(target.shape[0]):
            data_val, _ = valid_dataset[torch.randint(out_class[ii].item()*50,\
                    (out_class[ii].item()+1)*50, (1,)).item()]
            data_mix = (1- alpha) * data[ii] + alpha * data_val.to(device)
            data_r.append(data_mix.to(device))
        data_r = torch.stack(data_r).requires_grad_(True)

    return data_r


def loss_feature_diff(model, data, data_r, target, conv_out_layers, loss_type='l2'):
    with torch.no_grad():
        output_target = model(data).clone().detach()
        feature_target = model.penultimate.clone().detach()
        previous_target = model.previous.clone().detach()
        # feature_target = model.model.penultimate.clone().detach()
        # previous_target = model.model.previous.clone().detach()
        if conv_out_layers is not None:
            conv_out = [mod.output for mod in conv_out_layers]

    output_r = model(data_r)
    feature_r = model.penultimate
    previous_r = model.previous
    #import ipdb; ipdb.set_trace()
    if conv_out_layers is not None:
        conv_out_r = [mod.output for mod in conv_out_layers]

        conv_diff = torch.tensor(0.0).to(device)
        for conv_layer_out, conv_layer_out_r in zip(conv_out[-5:], conv_out_r[-5:]):
            conv_diff += ((conv_layer_out - conv_layer_out_r)**2).sum()

    if loss_type == "l2":
        feature_diff = 0                   
        feature_diff += torch.sum((feature_r - feature_target)**2) / target.shape[0]
        feature_diff += torch.sum((previous_r - previous_target)**2) / target.shape[0]

    elif loss_type == "kl":
        kl = torch.nn.KLDivLoss(reduction = "batchmean", log_target = True)
        feature_diff = kl(F.log_softmax(feature_r,dim=1), F.log_softmax(feature_target, dim=1))

    if conv_out_layers is not None:
        return feature_diff, conv_diff
    else:
        return feature_diff




def pgd_step(loss_fn, data_r, target):
    grad = torch.autograd.grad(loss_fn, data_r,
                                    retain_graph=False, create_graph=False)[0]
    grad_norms = torch.norm(grad.view(target.shape[0], -1), p=2, dim=1)
    grad = 0.1 * grad/ ( grad_norms.view(target.shape[0], 1, 1, 1) + 1e-8 )

    data_r = data_r.detach() - grad
    data_r = torch.clamp(data_r, min=0.0, max=1.0).detach()

    return data_r

    

def generate_dataset(model, train_loader, valid_dataset, device,batch_size,resultpath, alpha = 0.0, start_point = "gaussian", 
        loss_type = "l2",weight = 1e-2, noise_level = 0.2,
        max_iterations=1000, inner_per_distance_record=10, inner_per_image=100):

    model.eval()
    model = model.to(device)
    tt = transforms.ToPILImage()

    global_feature_diff_history = []
    global_test_feature_diff_history = []
    global_image_diff_history = []
    global_loss_history = []
    global_conv_diff_history = []
    target_r = []

    test_model = resnet.ResNet18()
    test_model = test_model.to(device)

    for idx, (data, target) in enumerate(train_loader):
        print("=================== img" + str(idx) + "=========================")
        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
        ########### Starting point Initilization ###############################################################
        
        data_r = init_gen(data, valid_dataset, target, device, start_point, noise_level, alpha)        

        conv_out_bool = True
        conv_out_layers = []
        if conv_out_bool == True:
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    conv_out_layers.append(ConvOutHook(module))
        

        feature_diff_history = []
        test_feature_diff_history = []
        image_diff_history = []
        loss_history = []     
        conv_diff_history = []
        #######################################################################
        ############### Use PGD to generate noisy images ######################
        for inner in range(max_iterations):
            data_r.requires_grad = True

            feature_diff, conv_diff = loss_feature_diff(model, data, data_r, target, conv_out_layers, loss_type)
            test_feature_diff = loss_feature_diff(test_model, data, data_r, target, None, loss_type)
            # image_diff = -weight * torch.sum((torch.clamp(data_r, min=0.0, max=1.0) - data)**2) / target.shape[0]
            image_diff = torch.sum((torch.clamp(data_r, min=0.0, max=1.0) - data)**2) / target.shape[0]
            # loss_fn = -weight* image_diff
            # loss_fn = feature_diff - weight * image_diff
            loss_fn = feature_diff - weight * image_diff + weight_conv*conv_diff + 0.1*weight_conv * test_feature_diff
            # loss_fn = feature_diff - weight * image_diff + 0.1*weight_conv * test_feature_diff
            # loss_fn = feature_diff + torch.exp(weight * image_diff)
            # import ipdb; ipdb.set_trace()
            data_r = pgd_step(loss_fn, data_r, target)

            if inner % 20 == 19:
                test_model = resnet.ResNet18().to(device)


            if inner % inner_per_distance_record == 0:
                feature_diff_history.append(feature_diff.item())
                image_diff_history.append(image_diff.item())
                loss_history.append(loss_fn.item())
                conv_diff_history.append(conv_diff.item())
                test_feature_diff_history.append(test_feature_diff.item())

            if inner % inner_per_image == 0 or inner == max_iterations-1:
                print('Train Epoch:{}\tLoss:{:.6f}\tF_diff:{:.6f}\tC_diff:{:.3f}\tI_diff:{:.3f}'.format(inner, loss_fn.item(), 
                    feature_diff.item(), conv_diff.item(), image_diff.item()))

        data_r = torch.clamp(data_r, 0.0, 1.0)
        
        global_feature_diff_history.append(feature_diff_history)
        global_image_diff_history.append(image_diff_history)
        global_loss_history.append(loss_history)
        global_conv_diff_history.append(conv_diff_history)
        global_test_feature_diff_history.append(test_feature_diff_history)

        
        if idx == 0:
            break

    global_feature_diff_history = torch.tensor(global_feature_diff_history)
    global_image_diff_history = torch.tensor(global_image_diff_history)
    global_loss_history = torch.tensor(global_loss_history)
    global_conv_diff_history = torch.tensor(global_conv_diff_history)
    global_test_feature_diff_history = torch.tensor(global_test_feature_diff_history)

    fig = plt.figure(figsize=(12, 8))
    x_axis = list(range(0, max_iterations-1, inner_per_distance_record))
    plt.plot(global_image_diff_history.mean(dim=0), label = "image_dst")
    plt.plot(global_feature_diff_history.mean(dim=0), label = "feature_dst")
    plt.plot(global_conv_diff_history.mean(dim=0), label = "conv_dst")
    plt.plot(global_loss_history.mean(dim=0), label = "loss")
    plt.plot(global_test_feature_diff_history.mean(dim=0), label = "random_feature_dst")
    plt.xlabel('x - iterations')
    plt.legend()
    plt.title("dist")
    fig.savefig(resultpath+"fig_loss")
    plt.close()
    


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 128
    train_loader, _, test_loader = get_cifar10_loader(50, batch_size, transforms.ToTensor())

    dropout_random = False
    resnet_type='ResNet18'
    if resnet_type == 'ResNet18':   
        modelpath = "./trained_models/CIFAR10_VAL_ResNet18_epoch_200.pt"
    elif resnet_type == 'ResNet9':
            modelpath = './trained_models/last_CIFAR10_VAL_ResNet9_noise0.01_alpha0.0_results.pt'
    elif resnet_type == 'ResNet50':
            modelpath = './trained_models/last_CIFAR10_VAL_ResNet50_noise0.0_alpha0.0_results.pt'   
    if resnet_type == 'ResNet18':
        model = resnet.ResNet18(random=dropout_random)
    elif resnet_type == 'ResNet9':
            model = resnet.ResNet9(random=dropout_random)
    elif resnet_type == 'ResNet50':
            model = resnet.ResNet50(random=dropout_random)
    checkpoint = torch.load(modelpath, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # scaled_model = ModelWithTemperature(model)
    # scaled_model.set_temperature(test_loader)

    resultpath = 'loss_visual'
    alpha = 0.5
    weight = 1
    weight_conv = 0.2
    noise_level = 0.2
    start_point = 'mixup'
    loss_type = 'l2'
    valid_dataset = CIFAR10(50, 0.0, mode = "valid", transform=transforms.ToTensor())
    resultpath = 'more_loss_visualweight'+str(weight).replace('.', '_')+'cov'+str(weight_conv).replace('.', '_')+\
        "startpoint"+str(start_point).replace('.', '_') + 'alpha' + str(alpha).replace('.', '_')
    # resultpath = 'Imagescale_loss_visualweight'+str(weight).replace('.', '_')+\
    #     "startpoint"+str(start_point).replace('.', '_') + 'alpha' + str(alpha).replace('.', '_')
    # resultpath = 'temperature_loss_visualweight'+str(weight).replace('.', '_')+\
    #     "startpoint"+str(start_point).replace('.', '_') + 'alpha' + str(alpha).replace('.', '_')
    
    generate_dataset(model, train_loader, valid_dataset, device,batch_size,resultpath, alpha, start_point, 
        loss_type,weight, noise_level, max_iterations=1000, inner_per_distance_record=10, inner_per_image=100)
    # generate_dataset(scaled_model, train_loader, valid_dataset, device,batch_size,resultpath, alpha, start_point, 
    #     loss_type,weight, noise_level, max_iterations=2000, inner_per_distance_record=10, inner_per_image=100)

