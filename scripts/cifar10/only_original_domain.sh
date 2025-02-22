python src/main.py \
        --training_phase 'only original domain' \
        --dataset_name CIFAR10 \
        --new_domain truck \
        --original_domain_for_increment automobile \
        --original_models_dir models/cifar10/truck/automobile