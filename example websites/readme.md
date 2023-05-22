# Example webpages to interact with the clustering API

[first-site.html](first-site.html) is a very basic test implementation that just pings the api to get the default image back. It's there to check that the server's set up in a way that can read and write files.

[custom.html](custom.html) This interacts with the clustering API to get clustering plots and image groupings. It will work with either the Linux, WSL or Windows implementatios of the API. You can also request a Scree Plot to guide your PCA selection.

[custom_interactive.html](custom_interactive.html) Allows you to get na [mpld3 interactive javascript visualisaitons](https://mpld3.github.io/index.html) and should only be used for small datasets.

[faiss_interface.html](faiss_interface.html) Is a basic page to allow you to upload a series of images to the API and have it return a the inforamtion on which group these should be in. As on other pages group "-1" represents noise.
