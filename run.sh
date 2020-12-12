sudo apt update -yqq
sudo apt install -yqq pv dialog
mkdir -p ./data/out
find ./data -type f -name 'split_*' -not -path "./data/out/*" -exec mv -i {} ./data/out  \;
cd ./data/out
echo "Starting archive reconstruction"
pv -p split_* > nd1.tar
echo "849f842e39b632b2e428c1bc6f65e92a  nd1.tar" | md5sum -c -
cp nd1.tar ..
cd ..
(pv -n nd1.tar | tar xvf - -C . ) 2>&1 | dialog --gauge "Extracting file..." 6 50
mv nd1 fgrc_dataset