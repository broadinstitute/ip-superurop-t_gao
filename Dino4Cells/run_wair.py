import os
wair_model_name_list = ['DenseNet121_change_avg_512_all_more_train_add_3_v2',
	'DenseNet121_change_avg_512_all_more_train_add_3_v3',
	'DenseNet121_change_avg_512_all_more_train_add_3_v5',
	'DenseNet169_change_avg_512_all_more_train_add_3_v5',
	'se_resnext50_32x4d_512_all_more_train_add_3_v5',
	'Xception_osmr_512_all_more_train_add_3_v5',
	'ibn_densenet121_osmr_512_all_more_train_add_3_v5_2']

for m in wair_model_name_list:
	# os.system(f"CUDA_VISIBLE_DEVICES=3 python get_embeddings.py --pretrained_weights None --dataset_path '/dgx1nas1/storage/data/mdoron/human_protein_atlas/website_whole_images.csv' --output_dir '/dgx1nas1/storage/data/mdoron/Dino4Cells/results/HPA_website_subset_wair_{m}/' --image_size 512 --arch vit_small --patch_size 8 --num_channels 3 --batch_size_per_gpu 32 --channel_type wair --model_type {m}")
    os.system(f'python classify_both.py --features_path /dgx1nas1/storage/data/mdoron/Dino4Cells/results/HPA_website_subset_wair_{m}/features.pth --df_path /dgx1nas1/storage/data/mdoron/human_protein_atlas/website_whole_images.csv --train_inds /dgx1nas1/storage/data/mdoron/human_protein_atlas/website_IDs_train.csv --test_inds /dgx1nas1/storage/data/mdoron/human_protein_atlas/website_IDs_test.csv --protein_classifier /dgx1nas1/storage/data/mdoron/Dino4Cells/results/HPA_website_subset_wair_{m}/protein_classifier.pth --cell_type_classifier /dgx1nas1/storage/data/mdoron/Dino4Cells/results/HPA_website_subset_wair_{m}/cell_type_classifier.pth --pretrained_classifier')

