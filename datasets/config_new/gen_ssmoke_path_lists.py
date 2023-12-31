import os


# Define the directories
train_image_dir = os.path.abspath('../../datasets/S-Smoke/curated/train/images')
train_label_dir = os.path.abspath('../../datasets/S-Smoke/curated/train/seg_labels')
test_image_dir = os.path.abspath('../../datasets/S-Smoke/curated/test/images')
test_label_dir = os.path.abspath('../../datasets/S-Smoke/curated/test/seg_labels')

# Get the list of file names in each directory
train_image_files = os.listdir(train_image_dir)
train_label_files = os.listdir(train_label_dir)
test_image_files = os.listdir(test_image_dir)
test_label_files = os.listdir(test_label_dir)

with open('train.list', 'w') as f:
    for image_file in train_image_files:
        for label_file in train_label_files:
            image_filename = os.path.basename(image_file).split('.')[0]
            label_filename = os.path.basename(label_file).split('.')[0]
            if image_filename == label_filename:
                f.write(f'{os.path.join(train_image_dir, image_file)}\t{os.path.join(train_label_dir, label_file)}\n')

with open('test.list', 'w') as f:
    for image_file in test_image_files:
        for label_file in test_label_files:
            image_filename = os.path.basename(image_file).split('.')[0]
            label_filename = os.path.basename(label_file).split('.')[0]
            if image_filename == label_filename:
                f.write(f'{os.path.join(test_image_dir, image_file)}\t{os.path.join(test_label_dir, label_file)}\n')

with open('val.list', 'w') as f:
    for image_file in test_image_files:
        for label_file in test_label_files:
            image_filename = os.path.basename(image_file).split('.')[0]
            label_filename = os.path.basename(label_file).split('.')[0]
            if image_filename == label_filename:
                f.write(f'{os.path.join(test_image_dir, image_file)}\t{os.path.join(test_label_dir, label_file)}\n')