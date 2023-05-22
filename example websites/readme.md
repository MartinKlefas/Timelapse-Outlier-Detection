
# Interactive Webpages for Clustering API

**[first-site.html](first-site.html)** is a minimalistic test implementation that simply sends a request to the API to retrieve the default image. It serves to verify that the server is correctly configured for file reading and writing.

**[custom.html](custom.html)** interacts with the clustering API to generate clustering plots and image groupings. It is compatible with Linux, WSL, or Windows implementations of the API. Moreover, this page allows for the request of a Scree Plot, which can provide guidance for your PCA selection.

**[custom_interactive.html](custom_interactive.html)** provides the capability to create [mpld3 interactive JavaScript visualizations](https://mpld3.github.io/index.html). This feature should be used solely for smaller datasets due to performance considerations.

**[faiss_interface.html](faiss_interface.html)** is a fundamental page that enables the upload of multiple images to the API, which then returns information regarding the group assignment for these images. As with other pages, group "-1" represents noise.

> Written with [StackEdit](https://stackedit.io/).