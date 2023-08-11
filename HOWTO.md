### Creating the enviroment
conda create --name app python=3.11 
conda activate app
conda install -c conda-forge opencv
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge ultralytics
conda install -c anaconda pyqt