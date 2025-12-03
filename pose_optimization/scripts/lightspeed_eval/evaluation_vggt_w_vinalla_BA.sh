#!/bin/bash
# orig megasam
# CUDA_VISIBLE_DEVICES=1 python anycam/scripts/evaluate_trajectories.py \
#     -cn evaluate_trajectories \
#     ++other_model=vggt \
#     ++model_path=pretrained_models/anycam_seq8 \
#     ++dataset=anycam/configs/dataset_cfgs/lightspeed.yaml \
#     ++fit_video.dataset_type=lightspeed \
#     ++fit_video.prediction.recon_data_path=/data2/zhuoyuan/cvpr2026/vggt \
#     ++fit_video.do_ba_refinement=false \
#     ++out_path=/data/zhuoyuan/anycam_exp/ \
#     ++fit_video.prediction.use_provided_depth=true \
#     ++fit_video.prediction.use_provided_masks=true \
#     ++fit_video.prediction.mask_path=/data/zhuoyuan/code/Sa2VA/outputs/SA2VA_8B_lightspeed_uniform_keyframe_selector_num_samples_8 \
#     ++fit_video.prediction.use_provided_flow=false \
#     ++fit_video.prediction.flow_model=unimatch \
#     ++fit_video.prediction.depth_predictor=unidepth  >> exps_BA/lightspeed_vggt_wo_BA_full.txt


#  anycam + BA
CUDA_VISIBLE_DEVICES=2 python anycam/scripts/evaluate_trajectories.py \
    -cn evaluate_trajectories \
    ++other_model=vggt \
    ++model_path=pretrained_models/anycam_seq8 \
    ++dataset=anycam/configs/dataset_cfgs/lightspeed.yaml \
    ++fit_video.dataset_type=lightspeed \
    ++fit_video.prediction.recon_data_path=/data2/zhuoyuan/cvpr2026/vggt \
    ++fit_video.do_ba_refinement=true \
    ++fit_video.ba_refinement.apply_semantic_filtering=false \
    ++fit_video.ba_refinement.w_track3d=5e-7 \
    ++fit_video.ba_refinement.add_track3d_loss=true \
    ++fit_video.ba_refinement.min_seq_len_threshold=8 \
    ++fit_video.ba_refinement.dynamic_loss_type=ray \
    ++fit_video.ba_refinement.optimize_rot=true \
    ++fit_video.ba_refinement.ba_type=global \
    ++fit_video.ba_refinement.visualize_tracks=false \
    ++fit_video.ba_refinement.generate_depth=false \
    ++fit_video.ba_refinement.max_uncert=0.05 \
    ++fit_video.ba_refinement.visualize_uncertainty=false \
    ++fit_video.ba_refinement.lr=1e-4 \
    ++fit_video.ba_refinement_level=0 \
    ++fit_video.ba_refinement.with_rerun=false \
    ++fit_video.ba_refinement.grid_size=16 \
    ++fit_video.ba_refinement.vinalla_ba=true \
    ++out_path=/data/zhuoyuan/anycam_exp/ \
    ++fit_video.prediction.use_provided_depth=true \
    ++fit_video.prediction.use_provided_masks=false \
    ++fit_video.prediction.mask_path=/data/zhuoyuan/code/Sa2VA/outputs/SA2VA_8B_lightspeed_uniform_keyframe_selector_num_samples_8 \
    ++fit_video.prediction.use_provided_flow=false \
    ++fit_video.prediction.flow_model=unimatch \
    ++fit_video.prediction.depth_predictor=unidepth >> exps_BA/lightspeed_vggt_vinalla_BA_full.txt