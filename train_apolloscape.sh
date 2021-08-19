CUDA_VISIBLE_DEVICES=0,1,2 python train.py \
                --batch_size=3 \
                --testBatchSize=1 \
                --crop_height=288 \
                --crop_width=576 \
                --maxdisp=192 \
                --threads=8 \
                --lr=0.001 \
                --milestones=[15,30,45,60,75,90]
                --dataset='apolloscape' \
                --save_path='./run/Apolloscape/' \
                --resume='./run/sceneflow/best/checkpoint/best.pth' \
                --fea_num_layer 6 --mat_num_layers 12 \
                --fea_filter_multiplier 8 --fea_block_multiplier 4 --fea_step 3  \
                --mat_filter_multiplier 8 --mat_block_multiplier 4 --mat_step 3  \
                --net_arch_fea='run/sceneflow/best/architecture/feature_network_path.npy' \
                --cell_arch_fea='run/sceneflow/best/architecture/feature_genotype.npy' \
                --net_arch_mat='run/sceneflow/best/architecture/matching_network_path.npy' \
                --cell_arch_mat='run/sceneflow/best/architecture/matching_genotype.npy' \
                --nEpochs=150 2>&1 |tee ./run/Apolloscape/log_2nd.txt

               #--resume='./run/Kitti12/best/best_1.16.pth'
