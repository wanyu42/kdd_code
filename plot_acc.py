import matplotlib.pyplot as plt
import pandas as pd

baseline_mixup = pd.read_excel("Results_acc.xlsx", sheet_name = "baseline_mixup")
baseline_noise = pd.read_excel("Results_acc.xlsx", sheet_name= "baseline_noise")
random_extractor_mixup = pd.read_excel("Results_acc.xlsx", sheet_name="mixup_random_extractor")
mixup = pd.read_excel("Results_acc.xlsx", sheet_name= "mixup")
mixup_0_01 = mixup.loc[mixup['weight'] == 0.01]
mixup_0_0 = mixup.loc[mixup['weight'] == 0.0]
random_extractor_0_01 = random_extractor_mixup.loc[random_extractor_mixup['weight']==0.01]



plt.figure(1)
plt.plot(baseline_mixup['mse'], baseline_mixup['acc'], label = "baseline_mixup")
plt.plot(baseline_noise['mse'], baseline_noise['acc'], label = "baseline_noise")
plt.plot(mixup_0_0['mse'], mixup_0_0['acc'], label = 'mixup_0')
plt.plot(mixup_0_01['mse'], mixup_0_01['acc'], label = 'mixup_0_01')
plt.plot(mixup_0_01['mse'], mixup_0_01['acc_dropout'], label = 'mixup_0_01_dropout')
plt.plot(mixup_0_0['mse'], mixup_0_0['acc_dropout'], label = 'mixup_0_0_dropout')
plt.plot(random_extractor_0_01['mse'], random_extractor_0_01['acc_random_initialized'], label = "mixup_random_extractor_0_01")
plt.ylim((10, 100))
plt.grid(True)
plt.legend()
plt.show()





