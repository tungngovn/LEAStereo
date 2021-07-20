python edit_predict.py \
                --apolloscape=1  --cuda=False  --maxdisp=192 \
                --crop_height=384  --crop_width=1248  \
                --data_path='./dataset/apolloscape/stereo_test/' \
                --test_list='./dataloaders/lists/apolloscape_test.list' \
                --save_path='./predict/apolloscape/images/' \
                --fea_num_layer 6 --mat_num_layers 12\
                --fea_filter_multiplier 8 --fea_block_multiplier 4 --fea_step 3  \
                --mat_filter_multiplier 8 --mat_block_multiplier 4 --mat_step 3  \
                --net_arch_fea='run/sceneflow/best/architecture/feature_network_path.npy' \
                --cell_arch_fea='run/sceneflow/best/architecture/feature_genotype.npy' \
                --net_arch_mat='run/sceneflow/best/architecture/matching_network_path.npy' \
                --cell_arch_mat='run/sceneflow/best/architecture/matching_genotype.npy' \
                --resume './run/Apolloscape/best.pth' 

