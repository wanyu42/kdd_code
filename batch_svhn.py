import os
import time
import numpy as np

os.environ['MKL_THREADING_LAYER'] = 'GNU'

#weight_range = 10**np.array(list(range(0,-4,-1)), dtype=float)
# weight_range = [10, 20, 30, 40, 50, 60]
weight_range = [2, 5]
#noise_range = [0.001, 0.01, 0.05, 0.1, 0.2]
# noise 0.2 means pretrained models, 0.0 means random initialized models
noise_range = [0.2]
# noise_range = [0.004, 0.008, 0.02, 0.05, 0.1, 0.12, 0.13, 0.15, 0.17, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4]
# noise_range = [0.05, 0.1, 0.2]
# noise_range = [0.04, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4]
# noise_range = [0.004, 0.008, 0.02, 0.12, 0.13, 0.15, 0.17, 0.18, 0.25, 0.3, 0.35, 0.4]
# noise_range = [0.12, 0.13, 0.17, 0.18]
# noise_range = [0.2]
# starting = "outclass"
# starting = "gaussian"
# starting = "mixup"
# starting = "reproduce"
# starting = "shift"
# starting = "repromixup"
starting = "cifar100"
# starting = "half"
# starting = "permmixup"

alpha_range = [1.0]
# alpha_range = [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
# alpha_range = [0.3, 0.4, 0.6, 0.7]
# alpha_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# alpha_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
# alpha_range = [0.75, 0.85]

conv_range = [True]
# conv_range = [False]
weight_conv_range = [1]

conv_part = 'former'
# conv_part = 'middle'
# conv_part = 'latter'
defense = 'None'
# defense = 'GradPrune_09'
# dropout_random_range = [True, False]
# model_train_range = [True, False]
max_iter = 500
model_type = "ResNet18SVHN"
# model_type = "ResNet18"
feat_weight = 0

while True:
    choice = input("1: Noisy masked dataset. 2: Baseline\n")
    choice = int(choice)
    if choice == 1:
        break
    elif choice == 2:
        break
    else:
        print("enter 1 or 2")


if choice == 1:
    for noise_value in noise_range:
        for weight_value in weight_range:
            for alpha_value in alpha_range:
                for conv in conv_range:
                    for weight_conv in weight_conv_range:
                        command = 'sbatch main_svhn_job.sb '+ \
                            str(weight_value) + " " + \
                            str(noise_value) + " " +  \
                            starting + " " + \
                            str(alpha_value) + " " + \
                            str(conv) + " " + \
                            str(weight_conv) + " " + \
                            conv_part + " " + \
                            str(max_iter) + " " + \
                            str(defense) + " " + \
                            str(max_iter)+"weight"+ str(weight_value).replace(".", "_")+starting+"max"+str(max_iter)+"noise" \
                            +str(noise_value).replace(".", "_")+"alpha"+str(alpha_value).replace(".", "_") \
                            +"conv"+str(conv)+"wgt"+str(weight_conv).replace(".", "_")+conv_part+"f_wgt"+str(feat_weight) \
                            + " "+ model_type \
                            + " "+ str(feat_weight)
                print(command)
                os.system(command)


elif choice == 2:
    if starting == "gaussian":
        for noise in noise_range:
            command = 'sbatch baseline_job.sb ' + str(noise) + " " + starting + " " + str(0.0)
            print(command)
            os.system(command)

    elif starting == "mixup":
        for alpha in alpha_range:
            command = 'sbatch baseline_job.sb ' + str(0.0) + " " + starting + " " + str(alpha)
            print(command)
            os.system(command)
