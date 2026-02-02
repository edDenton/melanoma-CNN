# melanoma-CNN

This project uses a convolutional neural network coded entirely from scratch using only numpy. The CNN features the following layers to classify:
1. Conv2D
2. MaxPool2D
3. Flatten
4. DenseLayer

This neural network was trained to determine whether images of moles could potentially have melanoma. The images are classified to be either malignant and benign.

Website URL: https://eddenton.github.io/melanoma-CNN/

The backend was hosted on Render: https://melanoma-cnn-backend.onrender.com

Process:
* Website accepts image -> User presses button
* Image gets resized to 128x128 to improve upload speed to server
* Image is passed to backend to forward propagate through model
* Model classifies the image to either malignant or benign
* Result is passed back through to the front end and displayed classification and confidence percentage

Some drawbacks that limit performance:
1. The free version of Render returns very slowly
2. Due to time and resources, the model isn't as well trained as I would like to be
3. Model doesn't have any check for whether the image submitted is actually a mole