"""Content-based two-tower network.

Two independent networks embed a user and a dish into the same 32-dimensional
space; their dot product is the prediction:

    v_u = l2_normalize(user_NN(x_u))
    v_m = l2_normalize(item_NN(x_m))
    prediction = v_u . v_m

Because neither tower ever sees an id - only features - this model can score a
customer who has ordered nothing and a dish nobody has rated. That is the whole
reason it exists alongside collaborative filtering.

## The scale problem, and how it is handled

L2-normalising both towers makes the dot product a cosine similarity, bounded to
[-1, 1]. Training that directly against 1-5 star ratings with MSE is not just
inaccurate, it is unsatisfiable: the target is outside the range the output can
reach, so the optimiser drives every embedding to a degenerate corner trying.

So targets are mapped onto the output's own range during training and mapped
back afterwards:

    train on   y = (rating - 3) / 2      in [-1, 1]
    predict    rating = 2 * dot + 3      back in [1, 5]

That leaves the content score in **rating units**, which is what makes the
hybrid blend in `ranking.py` meaningful - it is adding two numbers that mean the
same thing. Blending a raw cosine with a star rating, as a literal reading of
the brief would have it, adds quantities on different scales and lets whichever
happens to be larger dominate for no principled reason.

The implicit path has the mirror problem: sigmoid of a value bounded to [-1, 1]
can only produce probabilities in [0.27, 0.73], so the model could never express
confidence. A single learnable scale on the logit fixes that, and is the
standard temperature term from contrastive learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from ml.cofi import RatingMode

#: Output width of both towers. They must match - that is the architecture.
EMBEDDING_DIM = 32

#: Ratings live in [1, 5]; the midpoint and half-range map them onto [-1, 1].
RATING_MIDPOINT = 3.0
RATING_HALF_RANGE = 2.0


def scale_ratings(ratings: np.ndarray) -> np.ndarray:
    """Map 1-5 star ratings onto the [-1, 1] range a cosine can actually reach."""
    return (ratings - RATING_MIDPOINT) / RATING_HALF_RANGE


def unscale_ratings(scores: np.ndarray) -> np.ndarray:
    """Map cosine outputs back to star ratings."""
    return scores * RATING_HALF_RANGE + RATING_MIDPOINT


def build_tower(n_features: int, name: str) -> tf.keras.Model:
    """One 256 -> 128 -> 32 tower.

    The final layer is deliberately linear: a ReLU there would confine every
    embedding to the non-negative orthant, where no two vectors can ever have a
    cosine similarity below zero and the model loses its only way to say "this
    person actively dislikes this".
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,), name=f"{name}_input"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(EMBEDDING_DIM),
        ],
        name=name,
    )


def build_two_tower(
    n_user_features: int, n_item_features: int, mode: RatingMode = RatingMode.EXPLICIT
) -> tf.keras.Model:
    """The full model: two towers, L2-normalised, combined with a dot product."""
    user_input = tf.keras.layers.Input(shape=(n_user_features,), name="user_features")
    item_input = tf.keras.layers.Input(shape=(n_item_features,), name="item_features")

    user_tower = build_tower(n_user_features, "user_NN")
    item_tower = build_tower(n_item_features, "item_NN")

    vu = tf.keras.layers.Lambda(lambda t: tf.linalg.l2_normalize(t, axis=1), name="normalise_user")(
        user_tower(user_input)
    )
    vm = tf.keras.layers.Lambda(lambda t: tf.linalg.l2_normalize(t, axis=1), name="normalise_item")(
        item_tower(item_input)
    )

    dot = tf.keras.layers.Dot(axes=1, name="similarity")([vu, vm])

    if mode is RatingMode.IMPLICIT:
        # See the module docstring: a bounded logit cannot express confidence,
        # so the dot product is scaled before the sigmoid.
        output = ScaledSigmoid(name="conversion_probability")(dot)
    else:
        output = dot

    return tf.keras.Model(inputs=[user_input, item_input], outputs=output, name="two_tower")


class ScaledSigmoid(tf.keras.layers.Layer):
    """sigmoid(w * x) with a single learnable positive scale."""

    def __init__(self, initial_scale: float = 4.0, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.initial_scale = initial_scale

    def build(self, input_shape: tf.TensorShape) -> None:
        self.scale = self.add_weight(
            name="logit_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.sigmoid(self.scale * inputs)

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "initial_scale": self.initial_scale}


@dataclass
class TowerWeights:
    """A tower's parameters as plain numpy, for TensorFlow-free serving.

    The API loads these and runs the forward pass in numpy. Three dense layers
    are a handful of matrix multiplies; shipping a ~600MB framework to the
    serving image to perform them would be absurd.
    """

    kernels: list[np.ndarray] = field(default_factory=list)
    biases: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def from_keras(cls, tower: tf.keras.Model) -> TowerWeights:
        kernels: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for layer in tower.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                kernel, bias = layer.get_weights()
                kernels.append(np.asarray(kernel, dtype=np.float32))
                biases.append(np.asarray(bias, dtype=np.float32))
        return cls(kernels=kernels, biases=biases)


def extract_tower(model: tf.keras.Model, name: str) -> tf.keras.Model:
    """Pull one named tower back out of the assembled model."""
    layer = model.get_layer(name)
    if not isinstance(layer, tf.keras.Model):
        raise TypeError(f"layer {name!r} is not a tower")
    return layer
