python src/main.py \
        --training_phase 'domain incremental learning' \
        --dataset_name CIFAR10 \
        --new_domain truck \
        --original_domain_for_increment automobile \
        --batch_size 64 \
        --fixed_stats \
        --results_dir results/cifar10/truck/automobile/fixed_stats \
        --models_dir models/cifar10/truck/automobile/fixed_stats \
        --original_models_dir models/cifar10/truck/automobile