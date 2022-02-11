import matplotlib.pyplot as plt
import pandas as pd

# baseline_mixup = pd.read_excel("ResNet9.xlsx", sheet_name = "gaussian_baseline")
baseline_noise = pd.read_excel("ResNet9.xlsx", sheet_name= "gaussian_baseline")
# random_extractor_mixup = pd.read_excel("ResNet9.xlsx", sheet_name="mixup_random_extractor")
# mixup = pd.read_excel("ResNet9.xlsx", sheet_name= "mixup")
# mixup_0_01 = mixup.loc[mixup['weight'] == 0.01].loc[mixup['bn_fixed']==True].loc[mixup['dropout']==True]
# mixup_0_01_bn_modified_dropout = mixup.loc[mixup['weight'] == 0.01].loc[mixup['bn_fixed']==False].loc[mixup['dropout']==True]
# mixup_0_01_bn_fixed = mixup.loc[mixup['weight'] == 0.01].loc[mixup['bn_fixed']==True].loc[mixup['dropout']==False]
# mixup_0_01_bn_modified = mixup.loc[mixup['weight'] == 0.01].loc[mixup['bn_fixed']==False].loc[mixup['dropout']==False]

gaussian = pd.read_excel("ResNet9.xlsx", sheet_name= "comparison")
gaussian_1 = gaussian.loc[gaussian['weight'] == 1]


plt.figure(1)
# plt.plot(baseline_mixup['mse'], baseline_mixup['acc'], label = "baseline_mixup")
plt.plot(baseline_noise['mse'], baseline_noise['acc'], label = "baseline_noise")
# plt.plot(mixup_0_0['mse'], mixup_0_0['acc'], label = 'mixup_0')
plt.plot(gaussian_1['mse'], gaussian_1['acc'], label = 'gaussian_1_max_1000')
plt.plot(gaussian_1['mse_100'], gaussian_1['acc_100'], label = 'gaussian_1_max_100')
plt.plot(gaussian_1['mse_10'], gaussian_1['acc_10'], label = 'gaussian_1_max_10')
plt.plot(gaussian_1['mse_no_opt'], gaussian_1['acc_no_opt'], label = 'gaussian_no_opt')

# plt.plot(mixup_0_01['mse'], mixup_0_01['acc_dropout'], label = 'mixup_0_01_dropout')
# plt.plot(mixup_0_0['mse'], mixup_0_0['acc_dropout'], label = 'mixup_0_0_dropout')
# plt.plot(random_extractor_0_01['mse'], random_extractor_0_01['acc_random_initialized'], label = "mixup_random_extractor_0_01")
plt.ylim((0, 100))
plt.grid(True)
plt.legend()
plt.show()





