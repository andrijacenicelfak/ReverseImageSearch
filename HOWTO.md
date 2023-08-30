### Creating the enviroment
conda create --name app python=3.11 
conda activate app
conda install -c conda-forge opencv
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge ultralytics
conda install -c anaconda pyqt

### For the video player to work you need to install codecs
https://www.codecguide.com/download_k-lite_codec_pack_basic.htm