
# SignLangClassifier

## Abstract

This project explores the development and comparative evaluation of deep learning models for the classification of American Sign Language (ASL) hand gestures into corresponding alphabetical letters (A-Z) and numerical digits (0-9). The aim is to design and assess the performance of two distinct machine learning approaches: a scratch-built model and a fine-tuned pretrained model. The scratch-built model is developed from the ground up, using custom architectural design and trained exclusively on a curated dataset of 35 ASL gesture classes. In contrast, the fine-tuned model leverages transfer learning by adapting a pretrained model to the same ASL dataset, optimizing its performance for this specific task.

This study involves a detailed analysis of the models' design, training, and evaluation, with metrics such as accuracy, precision, recall, and F1-score used to measure their performance. Experimental results reveal the advantages and limitations of each approach, offering insights into the trade-offs between building a model from scratch and fine-tuning a pretrained architecture. The findings provide a comparative understanding of how different methodologies impact the efficiency and accuracy of ASL classification, contributing to the broader field of sign language recognition and its potential applications in assistive technologies.

## Outline Scratch Model (V1)

The starting outline for our baseline model in this project focuses on utilizing foundational concepts in TensorFlow, which we learned during assignments under the guidance of Professor Wenlu Chang. Our objective was to create a simple, yet effective, model to classify American Sign Language (ASL) hand gestures into corresponding numerical and alphabetical categories. To achieve this, we used grayscale image processing and focused on constructing a minimalistic pipeline that aligns with our coursework experience. This approach allowed us to apply the theoretical concepts from class to a practical, real-world problem in a straightforward manner.

The dataset preparation began with defining the path to the ASL image dataset stored in Google Drive. Using the os module, we ensured that the directory containing class folders (e.g., "0", "1", "a", "b", etc.) was correctly accessed. Each folder represented a specific class, and its contents were image files corresponding to that class. We looped through these folders systematically, verifying their existence and iterating over each file to extract images. This methodical traversal ensured no data was missed, and any issues with missing or corrupted files were logged for review.

![image](https://github.com/user-attachments/assets/231c9cc4-a094-4e27-be77-c3ddbea8e613)


Once the images were accessed, we employed the Python Imaging Library (PIL) to preprocess them. Each image was converted to grayscale using the "L" mode in PIL, which effectively reduces the computational complexity by eliminating color channels while retaining essential structural information. The images were then resized to a uniform dimension of 28x28 pixels. This resizing step served two purposes: it standardized input dimensions for the neural network and reduced the memory footprint, making the model more efficient for training and inference.

After preprocessing, the images were normalized by scaling their pixel values to the range [0, 1]. This normalization was crucial for ensuring consistent input data distribution, as neural networks tend to perform better when input features are normalized. The normalized images were stored in a NumPy array, which allowed for efficient manipulation and compatibility with TensorFlow. Alongside the image data, the corresponding class labels (derived from folder names) were also stored in an array, forming a complete dataset of inputs and outputs.

With the dataset prepared, we split it into training, validation, and testing subsets. We followed the standard practice of allocating 70% of the data for training, with the remaining 30% evenly divided between validation and testing. This splitting process was performed using train_test_split from the sklearn library, which ensured a randomized but reproducible division of the data. The resulting subsets were verified for shape and size, confirming that each contained images of dimension 28x28 with a single grayscale channel.

For visualization, we displayed one sample image from each class using matplotlib. This step was not merely for aesthetics but served as an important validation of the dataset integrity and preprocessing pipeline. It allowed us to confirm that the images were correctly resized, normalized, and labeled. The grid-like layout of images provided an intuitive overview of the dataset and reinforced our confidence in its readiness for training.

The neural network itself was designed as a sequential model in TensorFlow. Starting with a flattening layer, the 28x28 grayscale images were converted into a one-dimensional array of 784 pixels. This flattening step bridged the gap between the spatial representation of the images and the fully connected layers that followed. The first hidden layer consisted of 300 neurons with ReLU activation, which introduced non-linearity and enabled the network to learn complex features. A second hidden layer, with 100 neurons and ReLU activation, further refined these learned features.

![image](https://github.com/user-attachments/assets/8289d2c0-fb88-4bc7-8d46-f02bb747654d)


The output layer was designed with 36 neurons (one for each class, including numbers 0-9 and letters a-z), using the softmax activation function. Softmax converted the raw outputs into probabilities, ensuring that the sum of probabilities across all classes equaled 1. This made it easy to interpret the network's predictions, as the class with the highest probability could be directly taken as the output.

We compiled the model using sparse categorical cross-entropy as the loss function, which is well-suited for multi-class classification problems. The Adam optimizer was chosen for its efficiency and adaptability during weight updates. To monitor the training process, we included accuracy as an evaluation metric. These choices were guided by our class assignments, where similar configurations yielded reliable results.

Finally, the model was trained on the dataset for 20 epochs with a batch size of 100. During training, validation data was used to monitor the model's performance and prevent overfitting. Post-training, the model was evaluated on the test dataset to assess its generalization capability. The results, including training and validation accuracy/loss plots, were visualized to provide a comprehensive understanding of the model's learning dynamics.

![image](https://github.com/user-attachments/assets/db8297a7-efa5-42d1-acb0-ec7291eb35b3)


In summary, this baseline model represents a fundamental approach to solving the ASL classification problem. By adhering to the concepts and techniques taught in class, we constructed a clear and logical pipeline, starting from data preprocessing and ending with model evaluation. This process not only reinforced our understanding of neural networks but also laid the groundwork for more advanced experiments in future iterations of this project.

## Final Scratch Model (V2)

The second iteration of our scratch-built model marked significant advancements in design, training efficiency, and scalability, guided by Professor Wenlu Zhang's recommendations. Transitioning to PyTorch enabled a more streamlined and modular development process. Key improvements included a robust training loop with real-time metrics computation, automated learning rate adjustments, and seamless GPU integration, which significantly reduced training times and facilitated iterative experimentation.

To study the impact of dataset complexity, we employed a dynamic training pipeline with incremental class addition. Beginning with numeric digits (0-9) and gradually expanding to alphabetic characters (A-Z), the model maintained strong performance, achieving 95% accuracy on three classes and stabilizing around 90% as the class count exceeded 18. Data augmentation techniques such as random rotations, flips, and affine transformations improved dataset diversity, mitigating underfitting and enhancing generalization.

Inspired by the VGGNet framework, the architecture incorporated three convolutional blocks with increasing filter sizes (32, 64, 128), max-pooling layers for down-sampling, and dropout regularization to prevent overfitting. This hierarchical design enabled the model to effectively capture complex patterns while maintaining computational efficiency. A refined dataset splitting strategy (70% training, 15% validation, 15% testing) and cross-validation further ensured robust performance evaluation.

![image](https://github.com/user-attachments/assets/a81e1ced-7b37-47a1-b001-e9e7a68fab44)

![image](https://github.com/user-attachments/assets/6df9b540-8553-448a-b66b-9e20f939934e)


Visualization tools played a critical role in optimizing the model. Metrics like precision, recall, and confusion matrices highlighted areas for improvement, while training and validation loss plots guided architectural adjustments to balance bias and variance. The final model demonstrated strong generalization capabilities, closely aligning with industry standards and positioning it as a practical solution for real-world ASL classification tasks.

![image](https://github.com/user-attachments/assets/13de44e5-edac-4362-8f02-6f0f21a518f1)


