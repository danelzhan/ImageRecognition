import os
import numpy as np

train_val_images = 'train_images.npy' # Train 80%, Validation 20%
train_val_labels = 'train_labels.npy' # Train 80%, Validation 20%
test_images = 'test_images.npy'
test_labels = 'test_labels.npy'

train_val_images = np.load(train_val_images)
train_val_labels = np.load(train_val_labels)

# 90% of the training data for training, 10% for validation
train_images = train_val_images[:int(train_val_images.shape[0] * 0.9)]
train_labels = train_val_labels[:int(train_val_labels.shape[0] * 0.9)]

val_images = train_val_images[int(train_val_images.shape[0] * 0.9):]
val_labels = train_val_labels[int(train_val_labels.shape[0] * 0.9):]

test_images = np.load(test_images)
test_labels = np.load(test_labels)


def _save_pgm(path, image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).round()
        image = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image.shape
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(image.tobytes())


archive_dir = os.path.dirname(os.path.abspath(__file__))
special_dir = os.path.join(archive_dir, "special_train_images")
max_images = min(5000, train_images.shape[0])

for idx in range(max_images):
    label = int(train_labels[idx])
    label_dir = os.path.join(special_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)
    filename = f"{idx:05d}.pgm"
    _save_pgm(os.path.join(label_dir, filename), train_images[idx])
