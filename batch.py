import os
import time
import numpy as np

os.environ['MKL_THREADING_LAYER'] = 'GNU'

#weight_range = 10**np.array(list(range(0,-4,-1)), dtype=float)
weight_range = [10]
#noise_range = [0.001, 0.01, 0.05, 0.1, 0.2]
# noise 0.2 means pretrained models, 0.0 means random initialized models
noise_range = [0.2]
# noise_range = [0.004, 0.008, 0.02, 0.05, 0.1, 0.12, 0.13, 0.15, 0.17, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4]
# noise_range = [0.05, 0.1, 0.2]
# noise_range = [0.04, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4]
# noise_range = [0.004, 0.008, 0.02, 0.12, 0.13, 0.15, 0.17, 0.18, 0.25, 0.3, 0.35, 0.4]
# noise_range = [0.12, 0.13, 0.17, 0.18]
# noise_range = [0.2]
#logit_range = [True, False]
logit_range = [False]
# starting = "outclass"
# starting = "gaussian"
# starting = "mixup"
starting = "reproduce"
# starting = "repromixup"
# starting = "cifar100"
# starting = "half"
# starting = "permmixup"

alpha_range = [1.0]
# alpha_range = [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
# alpha_range = [0.3, 0.4, 0.6, 0.7]
# alpha_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# alpha_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
# alpha_range = [0.75, 0.85]

# dropout_random_range = [True, False]
# model_train_range = [True, False]
dropout_random_range = [False]
model_train_range = [False]
batch_stat = False
max_iter = 497

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
    ii = 40
    for dropout_random in dropout_random_range:
        for model_train in model_train_range:
            for logit_value in logit_range:
                for noise_value in noise_range:
                    for weight_value in weight_range:
                        for alpha_value in alpha_range:
                            
                            command = 'sbatch main_job.sb '+ str(weight_value) + " " + \
                                str(noise_value) + " " + str(logit_value) + " "  + \
                                "weight"+ str(weight_value).replace(".", "_")+starting+"max"+str(max_iter)+"noise"\
                                +str(noise_value).replace(".", "_")+"alpha"+str(alpha_value).replace(".", "_")\
                                +"dropout"+str(dropout_random) + "train" + str(model_train)+"batch" + str(batch_stat) \
                                + " " + starting + " " + str(alpha_value) + " " + str(dropout_random) + " " + \
                                str(model_train) + " " + str(batch_stat) + " " + str(max_iter)
                            print(command)
                            os.system(command)
                            ii += 1

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
