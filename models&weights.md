# Model Selection Considerations
In order to group images by similarity we first need to compute an embedding for each image, which is effectively a vector representation of the image. Not all embeddings are as good as one another though. The embedding itself is effectively looking for "important" parts of the image and paying more attention to that than what it's previusly determined to be "less important".

Each model is trained to understand what parts of an image are useful in knowing how to categorise that image. ImageNet for example is a collection images hand labeled with the presence or absence of 1000 object categories such as foxes or bananas. It's conceivable that all of our images are equally "bannana-like" according to a model trained on ImageNet. However, this is unlikely.

## Option 1 - Custom Trained Models

This is the ideal scenario. Instead of using a generalist model, it's better to use a model that has been fine-tuned on the dataset being used. That way, the underlying model better understands the input images. This can be done via a number of methods:

### Supervised learning
