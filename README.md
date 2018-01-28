# cardiovector

This is a Python package that implements various algorithms and utilities related to vectorcardiography - method of representing changes of electric potential in the heart, alternative to electrocardiography. 

In here, you will find algorithms for:
- reconstructing a vectorcardiogram from 12-lead ECG,
- plotting VCG signals in 3D space or as 2D projections,
- extracting various morphological features from VCG loops (for e.g. machine learning).

# What is vectorcardiography?

A technique which tries to represent electrical changes in human heart using an electrical dipole which changes its magnitude and orientation with time. Those changes can then be plotted in 3D space, as a system of loops of different size and orientation. Such plots might be used for diagnosis. 

Vectorcardiography is almost as old as ECG, but had never got as much of popularity as the latter, even though it was often proven to enhance diagnostic value as compared to the one based solely on ECG. The dealbreaker for the medical world was that it required dedicated (...I mean _different from the one used in ECG_) lead layout on patient's body, which in turn required additional staff training and equipment, so in the end it was never considered a standard medical means of diagnosis.

It turns out, on the other hand, that one may reconstruct a VCG signal from a standard 12-lead ECG with a high degree of similarity to a physically measured vectorcardiogram. There are many, many different algorithms being developed for that. Recent expansion of machine learning might also remove the need for training the staff to read VCG signals - automatic extraction of morphological features could provide additional information to a medical decision support system.

# Contents of the package

The package is still under development.

Please check out IPython notebooks in repo's root directory. They contain the latest code that showcases working features.

# Bibliography

Coming soon!