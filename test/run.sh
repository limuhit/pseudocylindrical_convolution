channel_list=(192)
#(172 144 112 84 56 28)
for i in ${!channel_list[*]};
do(
    "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/PConv/test/trainDDP_Full.py"  --valid-dim "${channel_list[$i]}"  --alpha 1 --lr 0.00003 --epochs 4 --init
    "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/PConv/test/trainDDP_Full.py"  --valid-dim "${channel_list[$i]}"  --alpha 1 --lr 0.00001 --epochs 18 --init
    "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/PConv/test/trainDDP_Full.py"  --valid-dim "${channel_list[$i]}"  --alpha 1 --lr 0.00001 --epochs 16
    "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/PConv/test/trainDDP_Full.py"  --valid-dim "${channel_list[$i]}"  --alpha 2 --lr 0.00001 --epochs 16
    "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/PConv/test/trainDDP_Full.py"  --valid-dim "${channel_list[$i]}"  --alpha 3 --lr 0.00001 --epochs 16 --restart
    "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/PConv/test/trainDDP_Full.py"  --valid-dim "${channel_list[$i]}"  --alpha 3 --lr 0.000001 --epochs 8 
)
done