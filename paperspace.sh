sudo apt update && sudo apt upgrade -y
sudo apt install linux-headers-$(uname -r) build-essential -y
sudo add-apt-repository ppa:graphics-drivers/ppa -y && sudo apt update
sudo apt install nvidia-driver-560 -y
sudo reboot
nvidia-smi