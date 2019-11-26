### Beyond Hello World, A Computer Vision Example

### Start Coding-

[Run this code on Jupyter Notebook with Google Colab](https://github.com/Dipeshpal/Machine-Learning-for-Vision-With-Tensorflow/blob/master/Beyond_Hello_World%2C_A_Computer_Vision_Example.ipynb)

Let's start with our import of TensorFlow

    import tensorflow as tf
    print(tf.__version__)

The Fashion MNIST data is available directly in the tf.keras datasets API. You load it like this:

    mnist = tf.keras.datasets.fashion_mnist

Calling load_data on this object will give you two sets of two lists, these will be the training and testing values for the graphics that contain the clothing items and their labels.

    (training_images, training_labels),  (test_images, test_labels) = mnist.load_data()

What does these values look like? Let's print a training image, and a training label to see...Experiment with different indices in the array. For example, also take a look at index 42...that's a a different boot than the one at index 0

    import matplotlib.pyplot as plt
    plt.imshow(training_images[0])
    print(training_labels[0])
    print(training_images[0])

You'll notice that all of the values in the number are between 0 and 255. If we are training a neural network, for various reasons it's easier if we treat all values as between 0 and 1, a process called '**normalizing**'...and fortunately in Python it's easy to normalize a list like this without looping. You do it like this:

    training_images = training_images / 255.0
    test_images = test_images / 255.0

Now you might be wondering why there are 2 sets...training and testing -- remember we spoke about this in the intro? The idea is to have 1 set of data for training, and then another set of data...that the model hasn't yet seen...to see how good it would be at classifying values. After all, when you're done, you're going to want to try it out with data that it hadn't previously seen!

Let's now design the model. There's quite a few new concepts here, but don't worry, you'll get the hang of them.

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
**Sequential**: That defines a SEQUENCE of layers in the neural network

**Flatten**: Remember earlier where our images were a square, when you printed them out? Flatten just takes that square and turns it into a 1 dimensional set.

**Dense**: Adds a layer of neurons

Each layer of neurons need an  **activation function**  to tell them what to do. There's lots of options, but just use these for now.

**Relu**  effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.

**Softmax**  takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!

The next thing to do, now the model is defined, is to actually build it. You do this by compiling it with an optimizer and loss function as before -- and then you train it by calling *_model.fit *_ asking it to fit your training data to your training labels -- i.e. have it figure out the relationship between the training data and its actual labels, so in future if you have data that looks like the training data, then it can make a prediction for what that data would look like.

    model.compile(optimizer = tf.train.AdamOptimizer(),
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=10)
    
Once it's done training -- you should see an accuracy value at the end of the final epoch. It might look something like 0.9098. This tells you that your neural network is about 91% accurate in classifying the training data. I.E., it figured out a pattern match between the image and the labels that worked 91% of the time. Not great, but not bad considering it was only trained for 5 epochs and done quite quickly.

But how would it work with unseen data? That's why we have the test images. We can call model.evaluate, and pass in the two sets, and it will report back the loss for each. Let's give it a try:

    model.evaluate(test_images, test_labels)

Let's predict some classes of some images-

    classifications = model.predict(test_images)
    print(classifications[0])
    
try running print(test_labels[0]) -- and you'll get a 9. Does that help you understand why this list looks the way it does?

    print(test_labels[0])

#### What does this list represent?

1.  It's 10 random meaningless values
2.  It's the first 10 classifications that the computer made
3.  It's the probability that this item is each of the 10 classes


#### Answer:

The correct answer is (3)

The output of the model is a list of 10 numbers. These numbers are a probability that the value being classified is the corresponding value, i.e. the first value in the list is the probability that the handwriting is of a '0', the next is a '1' etc. Notice that they are all VERY LOW probabilities.

For the 7, the probability was .999+, i.e. the neural network is telling us that it's almost certainly a 7.

#### How do you know that this list tells you that the item is an ankle boot?

1.  There's not enough information to answer that question
2.  The 10th element on the list is the biggest, and the ankle boot is labelled 9
3.  The ankle boot is label 9, and there are 0->9 elements in the list


#### Answer

The correct answer is (2). Both the list and the labels are 0 based, so the ankle boot having label 9 means that it is the 10th of the 10 classes. The list having the 10th element being the highest value means that the Neural Network has predicted that the item it is classifying is most likely an ankle boot


More options and readings can be found on colab notebook
[Run this code on Jupyter Notebook with Google Colab](https://github.com/Dipeshpal/Machine-Learning-for-Vision-With-Tensorflow/blob/master/Beyond_Hello_World%2C_A_Computer_Vision_Example.ipynb)
