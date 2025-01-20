source ~/.bashrc
#my_named_nohup infer_special_sets4truncation_gpu_x_3_plus bash ./inter_special_sets_gxl6.sh 3
#for i in {7..7}
#do
#    ((gpu_id = i - 1))
#    my_named_nohup infer_${gpu_id} bash ./infer_${i}.sh 7
#done
gpu_id=0
#my_named_nohup infer_${gpu_id} bash ./infer_8.sh $gpu_id # style

gpu_id=1
#my_named_nohup infer_${gpu_id} bash ./infer_7.sh $gpu_id #emotion

gpu_id=2
my_named_nohup infer_${gpu_id} bash ./infer_6.sh $gpu_id # capiton

gpu_id=3
#my_named_nohup infer_${gpu_id} bash ./infer_4.sh $gpu_id # age

gpu_id=4
#my_named_nohup infer_${gpu_id} bash ./infer_3.sh $gpu_id # gender