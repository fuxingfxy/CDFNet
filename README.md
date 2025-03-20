# CDFNet
 CrossDFNet: Dual-Domain Spatial-Frequency Fusion for Enhanced Remote Sensing Image Segmentation  
 ## Preparation  
*  	environment	
```
conda create -n environment's name python=3.8
conda activate environment's name
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
*  	dataset

The dataset was downloaded from the official website, and Potsdam and Vaihingen were split using potsdam_patch_split.py and vaihingen_patch_split.py in the tools directory, respectively. 

## Using
You can train using,`python train_supervision.py` test on the Vaihingen dataset using `python vaihingen_test.py`，test on the Potsdam dataset using `python potsdam_test.py`,  and test on the LoveDA dataset using `python loveda_test.py`.  

### We will continue to update the code repository！
