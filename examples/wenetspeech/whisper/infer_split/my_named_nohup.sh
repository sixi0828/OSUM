source ~/.bashrc

for i in {0..7}
do
    my_named_nohup infer_special_sets4split_gpu_x_$i bash ./infer_special_sets4split_gpu_x.sh $i
done