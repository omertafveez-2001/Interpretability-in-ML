# Interpretability-in-ML
Interpretability techniques in Machine Learning including Feature Viz, Interp in LLMs, and other techniques.

## What is ML Interpretability? 
Machine Learning, while dominant in its performance, poses some threats to ethical guidelines, safety in its responses regardless of the *type* of Deep Neural Network. It could be LLMs, VLMs, Motion in Robotics, Self-Driving cars etc. <br>

Our goal is to be able to answer what exactly is going inside the model by using the following questions <br>
1. What did the model learn?
2. What features from the input make the model generate certain outputs?

Knowing these things or these questions or knowing the mechanisms well, it allows us to 
1. Debug and tune the model
2. Increase our trust in the model
3. Discover novel insights from data. 

<br>
This gave rise to novel architectures such as Sparse Autoencoders that are widely adopted in Deep Learning to uncover hidden realities of large neural networks. 

### Tricking the Classifier
Here I use a pre-trained ResNet18 that was pretrained using ImageNet to demonstrate how tricking a classifier can happen very easily by injecting small noise. The idea is pretty simple. <br>
Backpropogation is computed across all the layers in the model starting from the output *Y* to the first layer of the model. But what if we could compute gradient of the output w.r.t the input tensor. <br>
By cloning the input tensor (to avoid making changes to the original tensor) and enabling its gradient allows us to compute backpropogation using  `loss.backward` after computing loss between the target class and the output. Then we add noise to the gradient of the input tensor and add it to the tensor much like gradient update **EXCEPT** that we **ADD** it instead of subtracting it. <br>

Very important and subtle detain in gradient update is the '-' sign which indicates moving against direction of the gradient which points to the decrease in loss. Here instead of subtracting, we *add* the updated noised gradient to the input tensor. Apart from '+' sign, everything is same as the gradient update. <br>

We retrive the final tensor from the model after updating the tensor with noised injection after `x` number of steps such that noised input is passed to the model at each iteration to make sure it gets further away from the target class.

<div style="display: flex; justify-content: space-around; align-items: center;">

  <div style="text-align: center; margin-right: 10px;">
    <img src="images/clown_fish.jpg" alt="Original Image" width="300">
    <p>Original Image</p>
  </div>

  <div style="text-align: center;">
    <img src="images/clown_fish_noised.png" alt="Adversarial Image" width="300", height=200>
    <p>Adversarial Image</p>
  </div>

</div>

Now that we have retrieved the adversarial tensor (shown above), the predicted class is 393 (Persian Cat) instead of 407 (Fish). So models look at patterns in their input that may not make sense for humans. 

### Using Leap Labs for Interpretability
Leap labs comes with an interpretability engine to evaluate how our models "think" before predicting a class.
- It shows what our model has learned and shows what our model should think like to predict a certain class. 
- It shows entanglement between classes: ie features that are shared across different classes which can help us identify where and why our model got confused. Higher entanglement is usally attained between similar objects. 
- Isolate features to help us identify what part of the input is the model referring to for a particular image. Or what features is it looking at each input for each class. Something useful for studying entanglement. 

### Feature Visualisation
It is also one of the important techniques to studying ML model's behaviours. It lets us understand what feature a particular unit in a neural network has learned. For example, what features is a convolutional layer looking at? <br>

For feature visulisation, we explore and use the differentiability of the model. 

#### Understanding Feature Visualisation
Since neural networks are differentiable w.r.t inputs, we can use a layer's activations as objective and optimize the input to maximize the this objective. <br>

We can optimize for logits as well as an objective. In all the cases we optimize an initial noisy image to maximize a particular output class. <br>

However it is not very simple since optimizing from random noise ends up having high-frequency features that do not look very natural.

<div style="display: flex; justify-content: center; align-items: center;">
  <div style="text-align: center;">
    <img src="images/features_vis.png" alt="Original Image" width="400", height=200>
    <p><strong>Source:</strong> Tim Sainburg</p>
  </div>
</div>

So we use *regularization* to force the optimization to produce more naturally looking images. <br>

### Regularization
Since it is an optimization problem, we can set up constraints much like we do in lagrange optimization in Calculus II or in Convex Optimization. Adding such constraints to our objective make the gradient move in directions that exhibit more of a certain pattern. 

- L1 Regularization: pushes weights to 0. The model uses the least number of features. 

We optimize the objective function in such a way that it generalises well to the data it has seen. We optimize it in such a way that it penalizes high variance neighboring pixels because such a portion of an image indicates noise. <br>

This is called `Feature Penalization`. Pixels very close to each other should be similar so it penalizes high variance neighbouring pixels. But this could penalize the edges where pixels change drastically for example person's clothes and background. <br>

We use transformation robustness where we transform the input (rotate, scale) so that feature visualisation is invariant to transformations of the image. <br>

This is why the prototyping works really well in Leap Labs and the images look very natural. 

<div style="display: flex; justify-content: space-around; align-items: center;">
  <div style="text-align: center; margin-right: 10px;">
    <div style="display: flex; justify-content: center; gap: 20px;">
      <img src="images/leap_labs.png" alt="Original Image 1" width="400", height=200>
      <img src="images/leap_labs2.png" alt="Original Image 2" width="700" height=150>
    </div>
    <p style="margin-top: 10px;">
      <strong>Source</strong>: 
      <a href="https://arxiv.org/pdf/2309.17144" target="_blank" style="color: #0645AD; text-decoration: none;">
        Leap Labs Paper: Prototype Generation: Robust Feature Visualisation for Data Independent Interpretability
      </a>
    </p>
  </div>
</div>

## Interpretability for Language Models
A language model is much like a stochastic probabilistic machine. It predicts the next token by assiging logits to it and using softmax or argmax we compute the token with highest logit/probability. <br>

Now we could proceed exactly how we did for vision models, compute backpropogation of the output w.r.t the input and optimize the input to maximize a particular output. However, the input space is discrete unlike vision where we have pixels ranging from 0-255 values. <br>

Similar experiement was run Leap Labs in which they found the words most associated with the token in the prompt. For example for token *good*: <br>

`got Rip Hut Jesus Shooting basketball Protective Beautiful Laughing good` <br>

Instead of input_tokens (input_ids) we try to optimize random input embeddings to maximize a specific output. However, this still posits two problems:

1. The embeddings that are optimizing may not correspond to any embedding in the model's dictionary
2. We need to force the model to make the input converge to an embedding that is present in the model's dictionary. 

