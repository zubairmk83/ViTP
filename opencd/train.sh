dir=$1
model=$2
gpus=$3
bash ./tools/dist_train.sh /nfs/liyuxuan/zhangyicheng/open-cd/configs/${dir}/${model}.py ${gpus} #--auto-resume
