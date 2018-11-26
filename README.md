# [Fabrik] Research implementation method for RNN package support in tensorflow import
This research was made for Google Code-in task from [Cloud-CV](https://cloudcv.org).

Before running code install dependencies via typing this command:
```
pip install -r requirements.txt
```
Better to evaluate it inside virtual environment.
## Problem
As for now, Fabrik cannot export models for Tensorflow containing any recurrence. And it's pretty hard to implement, because there's no operations "SimpleRNN" or "LSTM", so we have to find another way to do it.
## Observations
If we look at several `.pbtxt` files, we may notice some facts:
- The nodes which belong to specific layer have prefix, which has the following pattern: `<type_of_layer>_<number>`. We can indicate type of layer by trying to fetch this pattern. Also different layers have different numbers.
- Some nodes have an initialized values, unlike Keras models. It's painful when it's orthogonal and we have to determine it.
- If we try to search dropout rates, we'll end up with nothing. But we'll find `1 - dropout` values. Nodes containing them named `keep_prob`.
- Regularizations `l1` includes abs operation and `l2` includes square operation. And we can notice respective nodes which can be used as regularization indicator for specific values.
There's also `observe_consts.py` to make observations easier.
## Suggestions
There's `rnn_detector.py` that can extract some values from `.pbtxt` models. It just fetches needed nodes and read values from them. Initializer extraction logic is defined in `init_detector.py`.
You can play around with code and attached models typing:
```
python rnn_detector.py models/<desired_model_file>
```
You can compare result with [original settings](models/models_original_settings.md).
## Notes
- Detecting truncating initializers are not working at the moment. Also detection of random normal is naive. There's `RandomNormal` and `TruncatedNormal` nodes inside model. Fetching them will make import more robust.
- There are some values which script cannot detect. I was unable to fetch any indicative nodes of constraints. For the rest of regularizers and "checkbox" values I didn't do anything.
- LSTM also have additional activation function. It's Sigmoid by default. But it isn't changing inside file if I set another function.

