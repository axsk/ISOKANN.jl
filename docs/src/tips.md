# Tips and tricks

## Choosing an optimizer and regularization

During the ISOKANN iteration we train a neural network to represent the current "target", i.e. our current estimate of the $\chi$ function (as computed by the `isotarget` function).
For this supervised learning (essentially regression) task, we use a form of stochastic gradient descent that is performed by an optimizer from Flux.jl / Optimisers.jl

If the training is not converging it may be worthwile to test another optimizer. You can choose the optimizer during the construction of the `Iso` object through passing the keyword agument `opt`.
The default is 
```julia
Iso(opt=NesterovRegularized())
```
which has proven to be more stable then the common Adam optimizer. To switch to Adam you may try `opt=AdamRegularized()`.

Both these commands are convenience constructors for an `OptimizerChain`, combining the optimizer with regularization.

(Tikhonov/L2) regularization is adding a penalty to the training loss proportional to the magnitude of the model weights, and thus enforcing simpler/smoother models and therefore penalizing overfitting.

The constructors for `AdamRegularized` and `NesterovRegularized` both provide argument to adjust both the learning rate and regularization:
```julia
NesterovRegularized(lr=1e-3, reg=1e-4)
```

If your model seems to be overfitting you may want to try a higher regularization, e.g. a magnitude higher then the default,so `1e-3` instead of `1e-4`.