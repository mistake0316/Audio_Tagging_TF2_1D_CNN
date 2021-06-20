# Audio_Tagging_TF_1D_CNN
This is an 1D_CNN implement from Metformin-121 for competition "Tomofun 狗音辨識 AI 百萬挑戰賽" in T-BRain

## TODO
- [ ] Fix MULTI_CLASS_VERSION
- [ ] Make visialzation runnable (Folder "CAN_NOT_RUN_NOW")
- [ ] Add more description to argparse

## Dataset
Donwload dataset from Tbrain "Tomofun 狗音辨識 AI 百萬挑戰賽"

## Run the code
**0. Prepare Data**
<pre>
meta_train.csv

train
├── train_00001.wav
├── train_00002.wav
├── ...
└── train_01200.wav

public_test
├── public_00001.wav
├── public_00002.wav
├── ...
└── public_10000.wav

private_test
├── private_00001.wav
├── private_00002.wav
├── ...
└── private_20000.wav
</pre>

where meta_train.csv should in this format:

| Filename  | Label | Remark |
| - | - | - |
| train_00001  | 0  | Barking |
| train_00002  | 0  | Barking |
| ... | ... | ... |
| train_01200  | 5  | Dishes |

**1. Requirements**<br>
python : 3.7.6
>*#in requirements.txt*<br>
>matplotlib==3.4.2<br>
>dotmap==1.3.23<br>
>tensorflow==2.3.1<br>
>numpy==1.16.6<br>
>librosa==0.8.1<br>
>pandas==1.2.4<br>
>tqdm==4.61.1<br>

**2. Create TFRecord**<br>
<pre>
python3 data_utils.py
</pre>

**3. Training the Model**<br>
For multi-label-version
<pre>
bash train_multi_label_version.sh
# this command will create files in "MULTI_LABEL_VERSION_RETRAIN"
</pre>

For multi-class-version
<pre>
bash train_multi_class_version.sh
# This command will create files in "MULTI_CLASS_VERSION_RETRAIN"
# 
# This Option have some problem now, we will fix that soon,
# but the pretrain weight in MULTI_CLASS_VERSION is OK
# 這個方法目前有些問題，我們會盡快修復它
# 不過在 MULTI_CLASS_VERSION 中的模型權重沒有問題
</pre>

**4. Evaluate Public/Private Test** <br>
1. Open go_test.ipynb
2. Change **folder, EnsambleMODE** in the Section : **Choose One folder and EnsambleMode**
3. If do not have **private_test.pickle** and **public_test.pickle**, please un-comment the Section **Create Private/Public Test Data**
4. Run all cells in the notebook
5. The prediction will output at the **folder/EnsambleMODE_private_test.csv**

