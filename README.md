# Biometrics

Requires python 3.6-3.8

Requires CUDA or amd latest drivers

Requires cmake for dlib (choco install cmake / sudo apt install cmake then add to path if not cmake -v)

# Download dataset

```shell
sudo apt update -yqq
sudo apt install -yqq pv dialog wget
wget https://dataset.erwankessler.com/nd1.tar
echo "849f842e39b632b2e428c1bc6f65e92a  nd1.tar" | md5sum -c -
(pv -n nd1.tar | tar xvf - -C . ) 2>&1 | dialog --gauge "Extracting file..." 6 50
mv nd1 fgrc_dataset
```
