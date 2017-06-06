UNet for recognizing houses
===========================

This is a solution for task 2 of the deep neural networks course.

Contents:
- UNet implementation in file models.py (class InnerModel).
- data augmentation in training - six versions are generated for each sample (rotations, flips),
  implemented in datasource.TransposeAugment.
- data augmentation in test - each image is fed to the network in two forms: original and transposed
  and then the results (transformed back) are averaged. This happens in model.create_test_model.
- 1024 of images are randomly selected for test (each time with the same seed) and not included in
  the train set.

This was trained on a single GTX 1070 instance with 8G of GPU RAM and SSD drive. Loss on training
after 6h was 0.147682.
