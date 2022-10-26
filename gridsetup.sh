conda activate omm
mkdir -p data
ln -sf /datastores/smm2 data/smm2
ls data/smm2/youtube_small/train > data/trainListSmall.txt
ls data/smm2/youtube_small/validation > data/valListSmall.txt
ls data/smm2/youtube_medium/train > data/trainListMedium.txt
ls data/smm2/youtube_medium/validation > data/valListMedium.txt
ls data/smm2/youtube_large/train > data/trainListLarge.txt
ls data/smm2/youtube_large/validation > data/valListLarge.txt
pip install -v -e .