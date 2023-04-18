<template>
  <div class="main-page">
    <NavBar v-model:activeItem="activeItem" @setScrollTo="setScrollTo" ref="nav" class="aside" />
    <div class="page-info">
      <div class="container" ref="main">
        <div class="main">
          <div v-if="activeItem === 'Prerequisites'">
            <Prerequisites />
          </div>
          <template v-else>
            <div style="margin-bottom: 20px">
              <div class="page-title">{{ titleObj[activeItem].title }}</div>
              <div class="read-time">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                  <path
                    fill="#01579b"
                    d="M12 20c4.4 0 8-3.6 8-8s-3.6-8-8-8s-8 3.6-8 8s3.6 8 8 8m0-18c5.5 0 10 4.5 10 10s-4.5 10-10 10S2 17.5 2 12S6.5 2 12 2m5 11.9l-.7 1.3l-5.3-2.9V7h1.5v4.4l4.5 2.5Z"
                  />
                </svg>
                <span style="font-weight: 700">Estimated Time</span>:
                {{ titleObj[activeItem].readTime }} minutes
              </div>
            </div>
            <div v-if="activeItem === 11">
              <div>
                Convolutional Neural Networks (CNNs) are a type of deep learning model specifically
                designed for processing grid-like data, such as images, video frames, and
                spectrograms. They have gained popularity due to their remarkable success in image
                recognition, object detection, and image generation tasks. CNNs are particularly
                well-suited for these tasks because they can automatically learn to recognize
                complex patterns in the input data by using local spatial information.
              </div>
              <div class="mt40">
                The basic concept of CNNs revolves around the convolution operation, which is a
                mathematical operation that combines two functions to produce a third function. In
                the context of CNNs, the convolution operation is performed between the input data
                (e.g., an image) and a set of filters or kernels. Each filter is responsible for
                learning a specific feature or pattern in the input data, such as edges, corners, or
                textures.
              </div>
              <div class="mt40">
                CNNs consist of several layers, including convolutional layers, activation
                functions, pooling layers, and fully connected layers. The layers work together to
                process input data and extract useful features for the desired task:
              </div>
              <div class="mt40 bt1">
                <div>
                  <span class="stong"> Convolutional layer </span>: Applies filters to the input
                  data through a convolution operation, creating feature maps that highlight the
                  learned features.
                </div>
                <div class="mt20 mb20">
                  <span class="stong"> Activation function </span>: Introduces non-linearity into
                  the network, allowing it to learn complex patterns. The most commonly used
                  activation function is the Rectified Linear Unit (ReLU).
                </div>
                <img style="width: 100%" src="../assets/img/IntroductiontoCNNs.gif" alt="" />
              </div>
              <div class="mt40 stong mb20">Related Links</div>
              <div class="mb10">Websites:</div>
              <div class="mb30">
                <a href="https://www.tensorflow.org/tutorials/images/cnn"
                  >TensorFlow documentation: Introduction to Convolutional Neural Networks</a
                >
              </div>
              <div class="mb10">Paper:</div>
              <div>
                <a href="https://www.deeplearningbook.org/contents/convnets.html"
                  >Deep Learning book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville:</a
                >
              </div>
              <div class="mt20">
                <a
                  href="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"
                  >ImageNet classification with deep convolutional neural networks. Advances in
                  neural information processing systems</a
                >
              </div>
              <div class="mt20">
                <a href="https://arxiv.org/pdf/1409.1556.pdf"
                  >Very deep convolutional networks for large-scale image recognition.</a
                >
              </div>
              <div class="mt20">
                <a href="https://arxiv.org/pdf/1512.03385.pdf"
                  >Deep residual learning for image recognition. Proceedings of the IEEE conference
                  on computer vision and pattern recognition</a
                >
              </div>
            </div>
            <div v-if="activeItem === 21">
              <div style="font-weight: 700; font-size: 30px">Basic Components</div>
              <div class="mt40">
                <span class="stong">1. Input Layer</span>: The input layer is the first layer in a
                Convolutional Neural Network (CNN) that takes the raw data as input. In image
                recognition tasks, for example, the input layer receives an image represented as an
                array of pixel values. Typically, images are represented as a 3D array with
                dimensions corresponding to the height, width, and the color channels (red, green,
                blue). Example: For a color image of 28x28 pixels, the input layer would have
                28x28x3 = 2352 neurons.
              </div>
              <div class="mt40">
                <span class="stong">2. Hidden Layers</span>: Hidden layers in a CNN are the layers
                between the input and output layers, responsible for processing and learning
                features from the input data. A typical CNN has multiple hidden layers, which can
                include convolutional layers, pooling layers, and fully connected layers.
              </div>
              <div class="mt40">
                <span class="stong">3. Output Layer</span>: The output layer is the last layer of a
                CNN, which produces the final predictions. The number of neurons in the output layer
                depends on the task being performed. For classification tasks, the output layer
                typically has as many neurons as the number of classes, and a softmax activation
                function is applied to provide probabilities for each class. Example: In a 10-class
                image classification problem, the output layer would have 10 neurons.
              </div>
              <img style="width: 100%" src="../assets/img/OutputLayer.png" alt="" />
              <div class="mt40" style="font-weight: 700; font-size: 30px">
                More advanced details:
              </div>
              <div class="mt40">
                <span class="stong">4. Convolution Kernels and Channels</span>: Convolution kernels,
                or filters, are small matrices applied to the input data or feature maps in a
                sliding window fashion, resulting in feature maps. They are responsible for
                detecting features like edges, corners, and textures within the input data. Channels
                refer to the depth of input data or feature maps. For example, a color image has 3
                channels (red, green, and blue), while the output feature maps after a convolution
                operation may have multiple channels, each representing a different learned feature.
              </div>
              <img
                style="width: 100%"
                src="../assets/img/ConvolutionKernelsAndChannels.gif"
                alt=""
              />
              <div class="mt40">
                <span class="stong">5. Back-propagation and Weights</span>: Back-propagation is the
                learning algorithm used to train neural networks, including CNNs, by minimizing the
                error between the predicted and true output. During training, the network's weights
                are adjusted iteratively using the gradient of the loss function with respect to
                each weight. The gradients are calculated using the chain rule and are propagated
                backward from the output layer to the input layer, updating the weights and biases
                to minimize the loss. Here is a link of a simulator of Back-propagation in NN for
                you to find Back-propagation changes weights:
                <a href="https://www.mladdict.com/neural-network-simulator">
                  Neural Network Simulator (mladdict.com)</a
                >
              </div>
              <img
                class="mt20"
                src="../assets/img/BackPropagationAndWeights.gif"
                style="width: 100%"
                alt=""
              />
              <div class="mt40">
                <span class="stong">Activation Function</span>: Activation functions are non-linear
                functions applied to the output of a neuron or layer in a CNN. They introduce
                non-linearity into the network, allowing it to learn complex patterns and
                relationships in the input data. Common activation functions include Rectified
                Linear Unit (ReLU), sigmoid, and hyperbolic tangent (tanh). For example, ReLU is
                defined as f(x) = max(0, x), where x is the input to the function. Activation
                functions are applied element-wise to the output of the convolution or fully
                connected layers.
              </div>
              <div class="mt40 mb20">
                <span class="stong">Output layer</span>: The last fully connected layer is connected
                to the output layer, which typically uses a softmax activation function for
                multi-class classification problems, or a linear activation function for regression
                problems.
              </div>
              <div class="mt40 stong mb20">Related Links</div>
              <div class="mb10">Websites:</div>
              <div class="mb10">
                <a
                  href="https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/"
                  >CNN
                </a>
              </div>
              <div class="mb10">
                <a href="https://www.youtube.com/watch?v=tIeHLnjs5U8"
                  >Back-propagation introduction
                </a>
              </div>
              <div class="mb10">
                <a
                  href="https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning"
                  >Activation Function</a
                >
              </div>
              <div class="mt10">Paper:</div>
              <div class="mt10">
                <a href="https://ieeexplore.ieee.org/document/9422066">Activation Function</a>
              </div>
            </div>
            <div v-if="activeItem === 31">
              <div>
                Convolutional Neural Network (CNN) architecture is a deep learning model
                specifically designed for processing grid-like data, such as images, video frames,
                and spectrograms. The architecture consists of several layers that work together to
                extract features from the input data and perform the desired task, such as
                classification, segmentation, or object detection.
              </div>
              <div>The typical CNN architecture includes the following layers:</div>
              <div class="mt40">
                <span class="stong">Input layer</span>: This layer receives the raw input data, such
                as an image, and passes it to the subsequent layers in the network.
              </div>
              <div class="mt40">
                <span class="stong">Convolutional layer(s)</span>: These layers apply a set of
                filters (kernels) to the input data through a convolution operation. Each filter is
                responsible for learning a specific feature or pattern in the input data. The output
                of the convolution operation is a set of feature maps that highlight the learned
                features.
              </div>
              <div class="mt40">
                <span class="stong">Activation function</span>: After the convolution operation, an
                activation function is applied to introduce non-linearity into the network. The most
                commonly used activation function in CNNs is the Rectified Linear Unit (ReLU), but
                other activation functions, such as sigmoid and tanh, can also be used.
              </div>
              <div class="mt40">
                <span class="stong">Pooling layer(s)</span>: These layers reduce the spatial
                dimensions (width and height) of the feature maps, which helps decrease the
                computational complexity of the network and improve its ability to recognize
                patterns at different scales. Common pooling operations include max pooling and
                average pooling.
              </div>
              <div class="mt40">
                <span class="stong">Additional components (optional)</span>: Some CNN architectures
                may include additional components such as batch normalization (for normalizing
                feature maps across the batch) or dropout (for regularization to prevent
                overfitting). features.
              </div>

              <div class="mt40">
                <span class="stong">Fully connected layer(s)</span>: The final layers of a CNN are
                usually one or more fully connected layers. These layers perform classification or
                regression tasks by taking the output of the previous layers (flattened into a
                one-dimensional vector) and producing the final output of the network.
              </div>
              <div class="mt40">
                <span class="stong">Output layer</span>: The last fully connected layer is connected
                to the output layer, which typically uses a softmax activation function for
                multi-class classification problems or a linear activation function for regression
                problems.
              </div>
              <div class="mt40 mb20">
                There are various CNN architectures, such as LeNet-5, AlexNet, VGG, ResNet,
                Inception, and DenseNet, that have been proposed over the years, with each offering
                different improvements and enhancements. These architectures have been tailored for
                specific tasks and have influenced the development of new CNN models.
              </div>
              <img src="../assets/img/CNNArchitecture.gif" alt="" style="width: 100%" />
              <div class="mt40 stong mb20">Related Links</div>
              <div class="mb10">Websites:</div>
              <div class="mb30">
                <a
                  href="https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035"
                  >An Overview of ResNet and its Variants
                </a>
              </div>
              <div class="mb10">Paper:</div>
              <div>
                <a href="https://arxiv.org/pdf/1409.4842.pdf">Going deeper with convolutions</a>
              </div>
            </div>
            <div v-if="activeItem === 41">
              <div>
                In the context of deep learning and Convolutional Neural Networks (CNNs), the model
                refers to the architecture and structure of the neural network, including the number
                of layers, type of layers (convolutional, pooling, fully connected), and their
                arrangement. A well-designed model can significantly impact the performance and
                accuracy of the network in solving specific tasks.
              </div>
              <div>
                Hyperparameters, on the other hand, are adjustable parameters that determine the
                overall behavior of the model during the training process. They are not learned by
                the network but are set before training. Some common hyperparameters include:
              </div>
              <div class="mt40">
                <span class="stong">Learning rate</span>: The step size the model takes while
                updating the weights during training. A small learning rate may result in slow
                convergence, while a large learning rate may cause the model to overshoot the
                optimal weights.
              </div>
              <div class="mt40">
                <span class="stong">Batch size</span>: The number of samples used for each update in
                the training process. Larger batch sizes can lead to more stable gradients, but also
                require more memory.
              </div>
              <div class="mt40">
                <span class="stong">Number of epochs</span>: The number of times the entire dataset
                is passed through the network during training. More epochs may lead to better
                performance but also increase the risk of overfitting.
              </div>
              <div class="mt40">
                <span class="stong">Activation functions</span>: Functions applied to the output of
                each layer, introducing non-linearity and enabling the model to learn complex
                patterns. Common activation functions include ReLU, sigmoid, and tanh.
              </div>
              <div class="mt40">
                <span class="stong">Regularization techniques</span>: Methods to prevent overfitting
                by adding constraints to the model. Common regularization techniques include L1 and
                L2 regularization, dropout, and batch normalization.
              </div>

              <div class="mt40 mb20">
                <span class="stong">Optimizers</span>: Algorithms used to update the weights of the
                network during training. Examples include Stochastic Gradient Descent (SGD), Adam,
                RMSprop, and Adagrad. Selecting the right model architecture and hyperparameters is
                crucial for obtaining good performance on a given task. It often involves
                experimenting with different combinations, using techniques like grid search, random
                search, or Bayesian optimization to find the best configuration. Additionally,
                researchers may use transfer learning to leverage pre-trained models and fine-tune
                them for specific tasks, which can lead to faster convergence and better
                performance.
              </div>
              <img src="../assets/img/Modelandhyperparameters.gif" alt="" style="width: 100%" />
              <div class="mt40 stong mb20">Related Links</div>
              <div class="mb10">Websites:</div>
              <div class="mb10">
                <a
                  href="https://towardsdatascience.com/understanding-hyperparameters-and-its-optimisation-techniques-f0debba07568"
                  >Understanding Hyperparameters and its Optimisation techniques:
                </a>
              </div>
              <div class="mb10">
                <a href="https://course.fast.ai/">Practical Deep Learning for Coders </a>
              </div>
              <div class="mb30">
                <a href="https://www.deeplearning.ai/ai-notes/optimization"
                  >Hyperparameter Tuning in Deep Learning</a
                >
              </div>
              <div class="mb10">Paper:</div>
              <div class="mb10">
                <a href="https://jmlr.org/papers/v13/bergstra12a.html"
                  >Random search for hyper-parameter optimization.
                </a>
              </div>
              <div class="mb10">
                <a
                  href="https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf"
                  >Practical Bayesian optimization of machine learning algorithms.
                </a>
              </div>
              <div class="mb30">
                <a href="http://jmlr.org/papers/v15/srivastava14a.html"
                  >Dropout: A simple way to prevent neural networks from overfitting.
                </a>
              </div>
            </div>
            <div v-if="activeItem === 51">
              <div>
                In the context of deep learning and Convolutional Neural Networks (CNNs), designing
                and customizing a model involves creating a suitable network architecture and
                setting the hyperparameters to achieve optimal performance for a specific task.
              </div>
              <div>Here are general steps to design and customize a CNN:</div>
              <div class="mt40">
                <span class="stong">Define the problem</span>: Understand the problem you are trying
                to solve, such as image classification, object detection, or segmentation. Determine
                the input and output format, as well as the size and nature of the dataset.
              </div>
              <div class="mt40">
                <span class="stong">Select a base architecture</span>: Choose an existing CNN
                architecture as a starting point, such as VGG, ResNet, or Inception. You can
                leverage transfer learning by using pre-trained weights to initialize your network,
                which can speed up training and improve performance.
              </div>
              <div class="mt40">
                <span class="stong">Customize the architecture</span>: Modify the base architecture
                to suit your specific problem. You can add or remove layers, change layer parameters
                (e.g., filter sizes, strides, padding), or alter the activation functions. Make sure
                the output layer matches the number of classes
              </div>
              <div class="mt40">
                <span class="stong">Preprocess the data</span>: Prepare your dataset by applying
                normalization, data augmentation, and resizing or cropping images to fit the input
                size of the network. Split the dataset into training, validation, and testing sets.
              </div>
              <div class="mt40">
                <span class="stong">Set hyperparameters</span>: Choose appropriate values for
                learning rate, batch size, number of epochs, activation functions, regularization
                techniques, and optimizers. You can use techniques like grid search, random search,
                or Bayesian optimization to find the best hyperparameter configuration.
              </div>
              <div class="mt40">
                <span class="stong">Train the model</span>: Train the customized CNN using your
                preprocessed dataset. Monitor the training and validation losses and accuracy to
                ensure proper convergence and prevent overfitting. Use techniques such as early
                stopping, learning rate scheduling, or data augmentation to improve generalization.
              </div>
              <div class="mt40">
                <span class="stong">Evaluate and fine-tune</span>: Test the performance of your
                trained model on the test set. Analyze the results to identify areas for
                improvement, such as adding more data, adjusting hyperparameters, or modifying the
                architecture. Iterate the process until you achieve satisfactory performance.
              </div>
              <div class="mt40">
                <span class="stong">Deploy and monitor</span>: Once the model is fine-tuned and
                performing well, deploy it in your application or service. Continuously monitor its
                performance and update the model as needed to maintain its effectiveness.
              </div>
              <div class="mt40 mb20">
                Designing and customizing a CNN requires a good understanding of both the problem
                you are trying to solve and the underlying principles of deep learning. By
                iteratively refining the architecture and hyperparameters, you can create a model
                that performs well on your specific task.
              </div>
              <div class="mt40 stong mb20">Related Links</div>
              <div class="mb10">Websites:</div>
              <div class="mb10">
                <a href="http://cs231n.github.io"
                  >CS231n: Convolutional Neural Networks for Visual Recognition
                </a>
              </div>
              <div class="mb10">
                <a href="https://ruder.io/optimizing-gradient-descent"
                  >An overview of gradient descent optimization algorithms</a
                >
              </div>
              <div class="mb30">
                <a href="https://towardsdatascience.com/transfer-learning-using-keras-d804b2e04ef8"
                  >Transfer Learning using pre-trained models in Keras</a
                >
              </div>
              <div class="mb10">Paper:</div>
              <div class="mb10">
                <a
                  href="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"
                  >ImageNet classification with deep convolutional neural networks.
                </a>
              </div>
              <div class="mb30">
                <a href="https://arxiv.org/abs/1409.1556"
                  >Very deep convolutional networks for large-scale image recognition.
                </a>
              </div>
            </div>
            <div v-if="[12, 22, 32, 42, 52, 6].includes(activeItem)">
              <QuestionItem :key="activeItem" :list="answer[activeItem]" />
            </div>
            <div v-if="activeItem === 7">
              <div>
                <span class="stong">Objective</span>: Learn to train and test a CNN model using a
                dataset.
              </div>
              <div class="mt40">
                <span style="font-weight: 700; font-size: 30px">Instructions</span>:
              </div>
              <div class="mt20">
                <span class="stong">1. Choose a dataset</span>: Select a suitable dataset for image
                classification, such as CIFAR-10, CIFAR-100, or a custom dataset. Divide the dataset
                into training, validation, and testing sets.
              </div>
              <div class="mt40">
                <span class="stong">2. Preprocess the data</span>: Apply normalization, data
                augmentation, and resizing or cropping to the images. Make sure they fit the input
                size of the network.
              </div>
              <div class="mt40">
                <span class="stong">3. Select a base architecture</span>: Choose an existing CNN
                architecture as a starting point, such as VGG, ResNet, or Inception. Use pre-trained
                weights for transfer learning if available.
              </div>
              <div class="mt40">
                <span class="stong">4. Customize the architecture</span>: Modify the base
                architecture to suit your specific problem. Make sure the output layer matches the
                number of classes in the dataset.
              </div>
              <div class="mt40">
                <span class="stong">5. Set hyperparameters</span>: Choose appropriate values for
                learning rate, batch size, number of epochs, activation functions, regularization
                techniques, and optimizers.
              </div>
              <div class="mt40">
                <span class="stong">6. Train the model</span>: Train the customized CNN using the
                preprocessed dataset. Monitor the training and validation losses and accuracy to
                ensure proper convergence and prevent overfitting.
              </div>
              <div class="mt40 mb20">
                <span class="stong">7. Test the model</span>: Evaluate the trained model on the test
                set and analyze the results. Calculate metrics such as accuracy, precision, recall,
                and F1-score.
              </div>
              <div style="display: flex">
                <img src="../assets/img/TrainingandTesting1.gif" alt="" style="width: 50%" />
                <img src="../assets/img/TrainingandTesting2.gif" alt="" style="width: 50%" />
              </div>
            </div>
            <div v-if="activeItem === 8">
              <div>
                <span class="stong">Objective</span>: Apply a trained CNN model to recognize faces
                in images.
              </div>
              <div class="mt40">
                <span class="stong">Instructions</span>: Obtain a facial recognition dataset: Use a
                publicly available dataset such as LFW (Labeled Faces in the Wild) or create your
                own dataset with labeled faces.
              </div>
              <div class="mt40">
                <span class="stong">Preprocess the data</span>: Prepare the dataset by applying
                normalization, data augmentation, resizing or cropping images, and splitting it into
                training, validation, and testing sets.
              </div>
              <div class="mt40">
                <span class="stong">Train a CNN model</span>: Train a CNN model on the facial
                recognition dataset, following the same steps as in Part 2. You may need to adjust
                the architecture and hyperparameters to suit the facial recognition problem.
              </div>
              <div class="mt40">
                <span class="stong">Test the model</span>: Evaluate the trained CNN model on the
                test set and analyze the results. Calculate metrics such as accuracy, precision,
                recall, and F1-score.
              </div>
              <div class="mt40 mb20">
                <span class="stong">Apply the model to real-world images</span>: Test the model on
                new, unlabeled images to see how well it can recognize faces in real-world
                situations. You can use OpenCV or other libraries to detect faces in images and then
                apply the trained CNN model for recognition.
              </div>
              <img src="../assets/img/Facialrecognition.gif" alt="" style="width: 100%" />
              <div class="mt40">
                <a href="https://teachablemachine.withgoogle.com/train/image"
                  >Try to build and train your own recognition model：</a
                >
              </div>
            </div>
          </template>
          <!-- 下一个 上一个 -->
          <div class="next-last" v-if="activeItem !== 'Prerequisites'">
            <div>
              <div
                v-if="activeItem !== 11"
                style="display: flex; align-items: flex-end; cursor: pointer"
                @click="clickNextOrPre(false)"
                class="previouos"
              >
                <div style="margin-bottom: -5px">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                  >
                    <path
                      fill="#1a73e8"
                      d="M20 11v2H8l5.5 5.5l-1.42 1.42L4.16 12l7.92-7.92L13.5 5.5L8 11h12Z"
                    />
                  </svg>
                </div>

                <div style="margin-left: 15px">
                  <div class="text">Previouos</div>
                  <span class="title"> {{ getTitle(false) }}</span>
                </div>
              </div>
            </div>
            <div
              v-if="activeItem !== 8"
              style="display: flex; align-items: flex-end; cursor: pointer"
              @click="clickNextOrPre(true)"
              class="next"
            >
              <div style="margin-right: 15px">
                <div class="text">Next</div>
                <span class="title">{{ getTitle(true) }}</span>
              </div>
              <div style="margin-bottom: -5px">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                  <path
                    fill="#1a73e8"
                    d="M4 11v2h12l-5.5 5.5l1.42 1.42L19.84 12l-7.92-7.92L10.5 5.5L16 11H4Z"
                  />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script setup>
import { ref } from 'vue'
import NavBar from '../components/NavBar.vue'
import Prerequisites from '../components/Prerequisites.vue'
import QuestionItem from '../components/QuestionItem.vue'
const activeItem = ref('Prerequisites')
const nav = ref(null)
const main = ref(null)
const titleObj = {
  11: { title: 'Introduction to CNN', readTime: 10 },
  12: { title: 'Introduction to CNN:Check Your Understanding', readTime: 5 },
  21: { title: 'Components of CNN', readTime: 17 },
  22: { title: 'Components of CNN:Check Your Understanding', readTime: 3 },
  31: { title: 'CNN architecture', readTime: 20 },
  32: { title: 'CNN architecture:Check Your Understanding', readTime: 5 },
  41: { title: 'Model and hyperparameters', readTime: 40 },
  42: { title: 'Model and hyperparameters:Check Your Understanding', readTime: 20 },
  51: { title: 'Design and Customize', readTime: 2 },
  52: { title: 'Design and Customize:Check Your Understanding', readTime: 2 },
  6: { title: 'MCQs of Concepts', readTime: 15 },
  7: { title: 'Training and Testing', readTime: 20 },
  8: { title: 'Facial recognition', readTime: 25 }
}

const answer = {
  6: [
    {
      question:
        'Which of the following is NOT a typical layer in a Convolutional Neural Network (CNN)?',
      answer: [
        { id: 1, option: 'a', label: 'Convolutional layer' },
        { id: 2, option: 'b', label: 'Pooling layer' },
        { id: 3, option: 'c', label: 'Fully connected layer' },
        { id: 4, option: 'd', label: 'Recurrent layer' }
      ],
      rightOptionId: 4
    },
    {
      question: 'In a CNN, what does the convolutional layer do?',
      answer: [
        { id: 1, option: 'a', label: 'In a CNN, what does the convolutional layer do?' },
        { id: 2, option: 'b', label: 'Reduces the spatial dimensions of the input' },
        { id: 3, option: 'c', label: 'Combines information from previous layers' },
        { id: 4, option: 'd', label: 'Connects every input to every output' }
      ],
      rightOptionId: 1
    },
    {
      question: 'What is the primary purpose of the pooling layer in a CNN?',
      answer: [
        { id: 1, option: 'a', label: 'To introduce non-linearity into the network' },
        { id: 2, option: 'b', label: 'To reduce the spatial dimensions of the input' },
        { id: 3, option: 'c', label: 'To learn hierarchical features from the input' },
        { id: 4, option: 'd', label: 'To connect every input to every output' }
      ],
      rightOptionId: 2
    },
    {
      question: 'Which of the following activation functions is commonly used in CNNs',
      answer: [
        { id: 1, option: 'a', label: 'Sigmoid' },
        { id: 2, option: 'b', label: 'Tanh' },
        { id: 3, option: 'c', label: 'ReLU (Rectified Linear Unit)' },
        { id: 4, option: 'd', label: 'All of the above' }
      ],
      rightOptionId: 4
    },
    {
      question: 'What is the role of a loss function in training a CNN?',
      answer: [
        {
          id: 1,
          option: 'a',
          label: 'To measure the difference between the predicted output and the ground truth'
        },
        { id: 2, option: 'b', label: 'To adjust the weights of the network' },
        { id: 3, option: 'c', label: "To optimize the network's performance" },
        { id: 4, option: 'd', label: 'To define the architecture of the network' }
      ],
      rightOptionId: 1
    }
  ],
  12: [
    {
      question: 'What is the primary purpose of a convolutional layer in a CNN?',
      answer: [
        { id: 1, option: 'A', label: 'Introducing non-linearity into the network' },
        { id: 2, option: 'B', label: 'Reducing the spatial dimensions of the feature maps' },
        { id: 3, option: 'C', label: 'Learning specific features or patterns in the input data' },
        { id: 4, option: 'D', label: 'Performing classification or regression tasks' }
      ],
      rightOptionId: 3
    },
    {
      question: 'What is the most commonly used activation function in CNNs?',
      answer: [
        { id: 1, option: 'A', label: 'Sigmoid function' },
        { id: 2, option: 'B', label: 'Hyperbolic tangent function (tanh)' },
        { id: 3, option: 'C', label: 'Rectified Linear Unit (ReLU)' },
        { id: 4, option: 'D', label: 'Linear function' }
      ],
      rightOptionId: 3
    },
    {
      question: 'What is the primary purpose of a pooling layer in a CNN?',
      answer: [
        { id: 1, option: 'A', label: 'Introducing non-linearity into the network' },
        { id: 2, option: 'B', label: 'Reducing the spatial dimensions of the feature maps' },
        { id: 3, option: 'C', label: 'Learning specific features or patterns in the input data' },
        { id: 4, option: 'D', label: 'Performing classification or regression tasks' }
      ],
      rightOptionId: 2
    }
  ],
  22: [
    {
      question:
        'Which component in a CNN is responsible for reducing the spatial dimensions of the feature maps?',
      answer: [
        { id: 1, option: 'A', label: 'Convolutional layer' },
        { id: 2, option: 'B', label: 'Activation function' },
        { id: 3, option: 'C', label: 'Pooling layer' },
        { id: 4, option: 'D', label: 'Fully connected layer' }
      ],
      rightOptionId: 3
    },
    {
      question: 'What is the primary role of an activation function in a CNN?',
      answer: [
        { id: 1, option: 'A', label: 'Learning features or patterns from the input data' },
        { id: 2, option: 'B', label: 'Introducing non-linearity into the network' },
        { id: 3, option: 'C', label: 'Reducing the spatial dimensions of the feature maps' },
        { id: 4, option: 'D', label: 'Performing classification or regression tasks' }
      ],
      rightOptionId: 2
    },
    {
      question:
        'In a CNN, which layer is typically responsible for producing the final output of the network, such as class probabilities or regression values?',
      answer: [
        { id: 1, option: 'A', label: 'Convolutional layer' },
        { id: 2, option: 'B', label: 'Pooling layer' },
        { id: 3, option: 'C', label: 'Activation function' },
        { id: 4, option: 'D', label: 'Fully connected layer' }
      ],
      rightOptionId: 4
    }
  ],
  32: [
    {
      question: 'In a typical CNN architecture, what is the role of the input layer',
      answer: [
        { id: 1, option: 'A', label: 'Reducing the spatial dimensions of the input data' },
        { id: 2, option: 'B', label: 'Learning features or patterns from the input data' },
        { id: 3, option: 'C', label: 'Introducing non-linearity into the network' },
        {
          id: 4,
          option: 'D',
          label: 'Receiving the raw input data and passing it to subsequent layers'
        }
      ],
      rightOptionId: 4
    },
    {
      question: 'Which of the following is NOT a common component in a CNN architecture?',
      answer: [
        { id: 1, option: 'A', label: 'Convolutional layer' },
        { id: 2, option: 'B', label: 'Activation function' },
        { id: 3, option: 'C', label: 'Pooling layer' },
        { id: 4, option: 'D', label: 'Recurrent layer' }
      ],
      rightOptionId: 4
    },
    {
      question:
        'Which of the following CNN architectures introduced the concept of skip connections or residual connections to address the vanishing gradient problem in deep networks?',
      answer: [
        { id: 1, option: 'A', label: 'LeNet-5' },
        { id: 2, option: 'B', label: 'AlexNet' },
        { id: 3, option: 'C', label: 'VGG' },
        { id: 4, option: 'D', label: 'ResNet' }
      ],
      rightOptionId: 4
    }
  ],
  42: [
    {
      question:
        'Which of the following is not a hyperparameter in a Convolutional Neural Network (CNN)?',
      answer: [
        { id: 1, option: 'A', label: 'Learning rate' },
        { id: 2, option: 'B', label: 'Activation function' },
        { id: 3, option: 'C', label: 'Batch size' },
        { id: 4, option: 'D', label: 'Filter weights in a convolutional layer' }
      ],
      rightOptionId: 4
    },
    {
      question: 'When training a CNN, increasing the number of epochs may lead to',
      answer: [
        { id: 1, option: 'A', label: 'Faster convergence' },
        { id: 2, option: 'B', label: 'Better generalization' },
        { id: 3, option: 'C', label: 'Reduced risk of overfitting' },
        { id: 4, option: 'D', label: 'Increased risk of overfitting' }
      ],
      rightOptionId: 4
    },
    {
      question: 'Which of the following is not a common optimizer used in training CNNs?',
      answer: [
        { id: 1, option: 'A', label: 'Stochastic Gradient Descent (SGD)' },
        { id: 2, option: 'B', label: 'Adam' },
        { id: 3, option: 'C', label: 'RMSprop' },
        { id: 4, option: 'D', label: 'k-Nearest Neighbors (k-NN)' }
      ],
      rightOptionId: 4
    }
  ],
  52: [
    {
      question:
        'Which of the following is not a common pre-processing step for images in a Convolutional Neural Network (CNN)?',
      answer: [
        { id: 1, option: 'A', label: 'Normalization' },
        { id: 2, option: 'B', label: 'Data augmentation' },
        { id: 3, option: 'C', label: 'Image sharpening' },
        { id: 4, option: 'D', label: 'Resizing or cropping images' }
      ],
      rightOptionId: 3
    },
    {
      question: 'When customizing a CNN architecture, the output layer should?',
      answer: [
        { id: 1, option: 'A', label: 'Always have the same number of neurons as the input layer' },
        {
          id: 2,
          option: 'B',
          label: 'Always have a fixed number of neurons, regardless of the problem'
        },
        { id: 3, option: 'C', label: 'Match the number of classes for a classification problem' },
        { id: 4, option: 'D', label: 'Always use a softmax activation function' }
      ],
      rightOptionId: 3
    },
    {
      question:
        'Which of the following is not a technique to find the best hyperparameter configuration for a CNN?',
      answer: [
        { id: 1, option: 'A', label: 'Grid search' },
        { id: 2, option: 'B', label: 'Random search' },
        { id: 3, option: 'C', label: 'Simulated annealing' },
        { id: 4, option: 'D', label: 'Particle swarm optimization' }
      ],
      rightOptionId: 4
    }
  ]
}
const list = [11, 12, 21, 22, 31, 32, 41, 42, 51, 52, 6, 7, 8]

/**
 * @Description isNext
 */
const clickNextOrPre = (isNext) => {
  const index = list.findIndex((item) => item === activeItem.value)
  nav.value.clickInfoItem({ id: isNext ? list[index + 1] : list[index - 1] })
  setScrollTo()
}
const setScrollTo = () => {
  main.value.scrollTo(0, 0)
}
const getTitle = (isNext) => {
  const index = list.findIndex((item) => item === activeItem.value) + (isNext ? 1 : -1)
  return titleObj[list[index]]?.title
}
</script>
<style lang="less" scoped>
.mt10 {
  margin-top: 10px;
}
.mt5 {
  margin-top: 5px;
}
.mb5 {
  margin-bottom: 5px;
}
.mb10 {
  margin-bottom: 10px;
}
.mt20 {
  margin-top: 20px;
}
.mb20 {
  margin-bottom: 20px;
}
.mt30 {
  margin-top: 30px;
}
.mb30 {
  margin-bottom: 30px;
}
.mt40 {
  margin-top: 40px;
}
.mb40 {
  margin-bottom: 40px;
}
.bt1 {
  border-bottom: 1px solid #ccc;
}
.pb40 {
  padding-bottom: 40px;
}
.main-page {
  display: flex;
  width: 100%;
  height: calc(100% - 46px);
  .aside {
    min-width: 280px;
  }
}
.page-info {
  display: flex;
  justify-content: center;
  min-width: 936px;
  overflow: auto;
  box-sizing: border-box;
  background: #e9eaec;
  width: 100%;
  .container {
    box-sizing: border-box;
    width: 100%;
    height: 100%;
    padding: 20px 140px;
    overflow: auto;
    overflow-x: hidden !important;
    z-index: 999;
    .main {
      background-color: #fff;
      box-shadow: 0 2px 12px 0 rgb(0 0 0 / 10%);
      border-radius: 4px;
      padding: 20px 40px;
      border: 1px solid #ccc;
    }
  }
}
.page-title {
  font: 400 28px/34px Google Sans, Noto Sans, Noto Sans JP, Noto Sans KR, Noto Naskh Arabic,
    Noto Sans Thai, Noto Sans Hebrew, Noto Sans Bengali, sans-serif;
}
.read-time {
  display: flex;
  align-items: center;
  color: #01579b;
  background: #e1f5fe;
  margin: 0 -40px;
  height: 60px;
  line-height: 60px;
  padding-inline: 40px;
  margin-top: 20px;
  font-size: 14px;
  svg {
    margin-right: 10px;
  }
}
.stong {
  font-weight: 700;
  font-size: 18px;
}
.next-last {
  display: flex;
  justify-content: space-between;
  margin: 0 -10px;
  margin-top: 40px;

  .text {
    font-size: 14px;
    color: #1a73e8;
  }
  .title {
    color: #174ea6;
  }
  .next {
    justify-content: flex-end;
  }
  .previouos {
    justify-self: flex-start;
  }
}
</style>
