# Carapace: Visualizing data dynamics and geometry with phase space
Carapace is a Machine Learning Architecture which uses phase space embedding and recurrence plots to visualize the dynamics of data.

It works best in high-dimensionality environments where geometry matters more than percise wavelengths, as reccurence plots often lose percission in favor of visualizing general trends and patterns

There are two folders, one containing tests with synthetic data and no baseline, and one with organic data tests in comparison to a baseline model.

## Architecture
The carapace neural network architecture itself is just a convolutional neural network, however the original aspect of it is in it's data processing. it autosegments data to windows of samples in a time series, training a neural network on those windows to identify the optimal embedding process to phase space. This allows for flexibility across datasets.

It then converts the collection of windows to phase space using the learned embedding method, using that phase space projection to generate an accompanying recurrence plot. The recurrence plot, being an image, can be analyzed by a Convolutional Neural Network.

## Development Process
The first idea was to visualize data as a series of fractals, allowing CNNS to deduce meaning from those images. However, eventually it was settled on to 
