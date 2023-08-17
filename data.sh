BASE_ROOT=.

IMAGE_ROOT=$E:\deep code learning\Datasets\CUHK-PEDES/imgs
JSON_ROOT=$E:\deep code learning\Datasets\CUHK-PEDES/reid_raw.json
OUT_ROOT=$BASE_ROOT/cuhkpedes/processed_data


echo "Process CUHK-PEDES dataset and save it as pickle form"

python ${BASE_ROOT}/datasets/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 3
