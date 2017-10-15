import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = {
    'root_dir' : root_dir,
    'original_imd_dir': os.path.join(root_dir, 'dataset/formula_images/'),
    'original_train_path': os.path.join(root_dir, 'dataset/im2latex_train.lst'),
    'original_val_path': os.path.join(root_dir, 'dataset/im2latex_validate.lst'),
    'original_test_path': os.path.join(root_dir, 'dataset/im2latex_test.lst'),
    'original_formula_path': os.path.join(root_dir, 'dataset/im2latex_formulas.lst'),

    'img_dir': os.path.join(root_dir, 'mathcv/target/preprocessed_dataset/formula_images/'),
    'train_path': os.path.join(root_dir, 'mathcv/target/preprocessed_dataset/train.lst'),
    'val_path': os.path.join(root_dir, 'mathcv/target/preprocessed_dataset/validate.lst'),
    'test_path': os.path.join(root_dir, 'mathcv/target/preprocessed_dataset/test.lst'),
    'formula_path': os.path.join(root_dir, 'mathcv/target/preprocessed_dataset/formulas.lst'),

    'mapper_path': os.path.join(root_dir, 'mathcv/target/preprocessed_dataset/mapper.txt'),

    'image_height': 125,
    'image_width': 625,
    'downsample_ratio': 2.0,
    'image_postfix': '.png',
    'padding_size': '[8,8,8,8]',
    'label_length': 202,
    'num_thread_preprocess': 10,

    'batch_size': 30,
    'epochs': 1,
    'train_limit': 150,
    'val_limit': 50,
    'test_limit': 10,
    'dropout_prob': 0.75,
    'decoder_memory_dim': 512,
    'learning_rate': 0.1
}
