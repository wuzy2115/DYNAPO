#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# DATA_PATH=/data/zhuoyuan/DAVIS/JPEGImages/480p
DATA_PATH=/data/zhuoyuan/DAVIS_2017_test/DAVIS/JPEGImages/480p
CKPT_PATH=checkpoints/megasam_final.pth

# evalset=(
# bear              dog             lab-coat            schoolgirls
# bike-packing      dog-agility     lady-running        scooter-black
# blackswan         dog-gooses      libby               scooter-board
# bmx-bumps         dogs-jump       lindy-hop           scooter-gray
# bmx-trees         dogs-scale      loading             sheep
# boat              drift-chicane   longboard           shooting
# boxing-fisheye    drift-straight  lucia               skate-park
# breakdance        drift-turn      mallard-fly         snowboard
# breakdance-flare  drone           mallard-water       soapbox
# bus               elephant        mbike-trick         soccerball
# camel             flamingo        miami-surf          stroller
# car-roundabout    goat            motocross-bumps     stunt
# car-shadow        gold-fish       motocross-jump      surf
# car-turn          hike            motorbike           swing
# cat-girl          hockey          night-race          tennis
# classic-car       horsejump-high  paragliding         tractor-sand
# color-run         horsejump-low   paragliding-launch  train
# cows              india           parkour             tuk-tuk
# crossing          judo            pigs                upside-down
# dance-jump        kid-football    planes-water        varanus-cage
# dance-twirl       kite-surf       rallye              walking
# dancing           kite-walk       rhino
# disc-jockey       koala           rollerblade
# )
evalset=(
aerobatics    dog-control      hurdles        ocean-birds      slackline
bike-trial    dolphins         inflatable     orchid           speed-skating
boxing        e-bike           juggle         people-sunset    subway
burnout       giant-slalom     kart-turn      planes-crossing  swing-boy
carousel      girl-dog         kids-turning   pole-vault       tackle
car-race      golf             lions          rollercoaster    tandem
cats-car      grass-chopper    lock           running          tennis-vest
chamaleon     guitar-violin    man-bike       salsa            tractor
choreography  gym              mbike-santa    seasnake         turtle
deer          helicopter       monkeys        selfie           varanus-tree
demolition    horsejump-stick  monkeys-trees  skate-jump       vietnam
dive-in       hoverboard       mtb-race       skydive          wings-turn
)


for seq in ${evalset[@]}; do
    CUDA_VISIBLE_DEVICES=9 python camera_tracking_scripts/test_demo.py \
    --datapath=$DATA_PATH/$seq \
    --weights=$CKPT_PATH \
    --scene_name $seq \
    --mono_depth_path $(pwd)/Depth-Anything/video_visualization \
    --metric_depth_path $(pwd)/UniDepth/outputs \
    --disable_vis $@
done

