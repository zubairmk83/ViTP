dir=$1
model=$2
gpus=$3
bash ./tools/dist_test.sh ./configs/${dir}/${model}.py ./work_dirs/${model}/iter_120000.pth ${gpus} #--format-only --eval-options submission_dir=./results/${model}

# ./tools/dist_test.sh ./configs/${dir}/${model}.py ./work_dirs/${model}/epoch_8.pth 8 --format-only --eval-options submission_dir=./results/e8_${model}
# ./tools/dist_test.sh ./configs/${dir}/${model}.py ./work_dirs/${model}/epoch_12.pth 8 --format-only --eval-options submission_dir=./results/e12_${model}
# ./tools/dist_test.sh ./configs/${dir}/${model}.py ./work_dirs/${model}/epoch_11.pth 8 --format-only --eval-options submission_dir=./results/e11_${model}
