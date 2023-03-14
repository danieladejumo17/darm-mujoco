sudo apt update
sudo apt install build-essential -y
pip install ray[rllib]==2.2.0
pip install wandb
pip install tensorflow_probability
python setup.py install
pip install pandas -U
sudo apt-get install libglfw3 -y
sudo apt-get install libglfw3-dev -y
pip install --user glfw