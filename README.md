# Deep Learning Lab: Experiments 8 & 9
## Effect of Attention Type and Attention Position in Pretrained ResNet50

## 1. Objective

To study the effect of different attention mechanisms and their placement within a pretrained ResNet50 architecture on image classification performance. The experiment compares classification accuracy, macro precision, macro recall, macro F1-score, model parameters, inference time, and attention visualizations.

## 2. Overview

This experiment uses transfer learning with a pretrained ResNet50 model to classify flower images into five categories. Different attention mechanisms are incorporated into the network to investigate how feature recalibration and spatial relationships affect classification performance.

The experiment includes:
- Baseline ResNet50 without additional attention.
- Channel attention.
- Spatial attention.
- Combined channel and spatial attention.
- Self-attention.
- Combined channel, spatial, and self-attention.
- Attention-position visualization at different feature stages (P2, P3, and P4).
- Additional attention mechanisms such as Squeeze-and-Excitation (SE) and Efficient Channel Attention (ECA), where implemented.

## 3. Dataset

**Dataset:** TensorFlow Flowers (`tf_flowers`)

The dataset contains images belonging to five flower classes:

1. Daisy
2. Dandelion
3. Roses
4. Sunflowers
5. Tulips

### Dataset split

The dataset is divided into three subsets:

| Subset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The training set is used to optimize model parameters, the validation set is used to monitor performance during training, and the test set is used for final evaluation.

## 4. Technologies Used

- Python
- TensorFlow and Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab or Jupyter Notebook

## 5. Model Architecture

### Baseline ResNet50

The baseline model uses ResNet50 pretrained on ImageNet. The classification head consists of global average pooling, dropout, and a dense output layer with five classes and softmax activation.

### Attention variants

| Model | Attention mechanism |
|---|---|
| A0 | Baseline ResNet50 |
| A1 | Channel attention |
| A2 | Spatial attention |
| A3 | Channel + spatial attention |
| A4 | Self-attention |
| A5 | Combined channel + spatial + self-attention |

Attention modules are incorporated at the selected feature stage, P3, for the attention-type comparison. The attention-position experiment investigates the effect of applying attention at different stages, P2, P3, and P4.

## 6. Experimental Procedure

1. Import the required libraries and load the TensorFlow Flowers dataset.
2. Split the dataset into training, validation, and testing subsets.
3. Resize images to 224 × 224 pixels.
4. Apply ResNet50 ImageNet preprocessing and training-only data augmentation.
5. Load the pretrained ResNet50 backbone and construct the baseline classifier.
6. Train the baseline model for 10 epochs.
7. Implement the attention mechanisms and integrate them into the selected feature stage.
8. Train each attention variant using the same dataset split and training protocol.
9. Record training and validation accuracy and loss.
10. Evaluate the trained models on the test set.
11. Compare classification metrics, parameter counts, and inference time.
12. Generate confusion matrices and attention visualizations.

## 7. Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy:** Proportion of correctly classified test images.
- **Macro Precision:** Average precision across all five classes.
- **Macro Recall:** Average recall across all five classes.
- **Macro F1-score:** Average F1-score across all five classes.
- **Total Parameters:** Total number of model parameters.
- **Inference Time:** Average time required to classify one image under the measured evaluation setup.

## 8. Visualizations

The experiment generates the following plots and visualizations:

1. Training accuracy across models.
2. Validation accuracy across models.
3. Training loss across models.
4. Validation loss across models.
5. Attention mechanism versus test accuracy.
6. Attention mechanism versus macro F1-score.
7. Accuracy versus total model parameters.
8. Macro F1-score versus inference time.
9. Baseline ResNet50 confusion matrix.
10. Channel + spatial attention confusion matrix.
11. Input image, spatial attention heatmap, and attention overlay.
12. Channel attention weight distributions.
13. Self-attention response and overlay.
14. Attention maps and overlays at P2, P3, and P4.
15. Baseline and attention-model Grad-CAM comparisons.

These visualizations help explain both classification performance and the regions or features emphasized by the attention mechanisms.

## 9. Results and Discussion

The baseline ResNet50 provides a reference for evaluating the impact of attention mechanisms. The channel and channel–spatial attention models can be compared against this baseline to determine whether attention improves classification performance.

Spatial attention and self-attention must be evaluated using the same test set and evaluation protocol. The confusion matrices help identify classes that are frequently misclassified, while heatmaps and overlays provide qualitative insight into the models' focus.

The results should be interpreted using the actual measured test metrics. Attention mechanisms do not necessarily improve every model, and additional attention layers may increase computational cost without improving accuracy.

## 10. Expected Outcome

The experiment demonstrates how different attention mechanisms and their positions affect a pretrained CNN's image classification performance. It also illustrates the trade-off between classification quality, model complexity, inference time, and interpretability.

## 11. How to Run

1. Open the experiment notebook in Google Colab or Jupyter Notebook.
2. Install the required dependencies if they are not already available.
3. Execute the notebook cells in order.
4. Allow each model to complete training.
5. Run the evaluation and visualization cells.
6. Record the final test metrics and include the generated plots in the laboratory report.

## 12. Conclusion

This experiment compares a pretrained ResNet50 baseline with multiple attention-based variants on the TensorFlow Flowers dataset. Quantitative metrics measure classification performance and computational cost, while confusion matrices, attention maps, and Grad-CAM visualizations support qualitative analysis. The combined findings help determine which attention mechanism and feature-stage placement are most suitable for this image-classification task.

**Note:** Report actual experimental results from the executed notebook. Results can vary with random initialization, data splitting, training configuration, and hardware.
