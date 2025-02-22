python src/main.py \
        --training_phase 'domain incremental learning' \
        --dataset_name MNIST \
        --new_domain 9 \
        --original_domain_for_increment 8 \
        --batch_size 64 \
        --results_dir results/mnist/9/8/updated_stats \
        --models_dir models/mnist/9/8/updated_stats \
        --original_models_dir models/mnist/9/8