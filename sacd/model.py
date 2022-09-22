import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def Conv(filters, kernel_size, activation, padding):
    return layers.Conv2D(filters=filters, kernel_size=kernel_size,
                         activation=activation,
                         padding=padding,
                         kernel_initializer=keras.initializers.HeUniform(),
                         bias_initializer='zeros')


def Dense(units, activation=None):
    return layers.Dense(units=units, activation=activation,
                        kernel_initializer=keras.initializers.HeUniform(),
                        bias_initializer='zeros')


class BaseNetwork(keras.Model):
    def save(self, path):
        self.save_weights(path)

    def load(self, path):
        self.load_weights(path)


class DQNBase(BaseNetwork):

    def __init__(self):
        super(DQNBase, self).__init__()

        self.net = keras.Sequential(
            [
                Conv(64, 3, activation="relu", padding="same"),
                Conv(128, 3, activation="relu", padding="same"),
                Conv(64, 3, activation="relu", padding="same"),
                Conv(32, 3, activation="relu", padding="valid"),
                Conv(32, 3, activation="relu", padding="valid"),
                layers.Flatten(),
            ]
        )

    def call(self, states):
        return self.net(states)


class QNetwork(BaseNetwork):

    def __init__(self, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()

        if not shared:
            self.conv = DQNBase()

        if not dueling_net:
            self.head = keras.Sequential([
                Dense(256, 'relu'),
                Dense(num_actions),
            ])
        else:
            self.a_head = keras.Sequential([
                Dense(512, 'relu'),
                Dense(num_actions),
            ])
            self.v_head = keras.Sequential([
                Dense(512, 'relu'),
                Dense(1),
            ])

        self.shared = shared
        self.dueling_net = dueling_net

    def call(self, states):
        if not self.shared:
            states = self.conv(states)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(num_actions, shared, dueling_net)
        self.Q2 = QNetwork(num_actions, shared, dueling_net)

    def call(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CateoricalPolicy(BaseNetwork):

    def __init__(self, num_actions, shared=False):
        super().__init__()
        if not shared:
            self.conv = DQNBase()

        self.head = keras.Sequential([
            Dense(256, 'relu'),
            Dense(num_actions),
        ])
        self.shared = shared

    def act(self, states):
        if not self.shared:
            states = self.conv(states)

        action_logits = self.head(states)
        greedy_actions = tf.argmax(
            action_logits, axis=1)
        return greedy_actions

    def sample(self, states):
        if not self.shared:
            states = self.conv(states)

        action_probs = tf.math.softmax(self.head(states), axis=1)
        actions = tf.random.categorical(action_probs, 1)[0, 0]

        # Avoid numerical instability.
        log_action_probs = tf.math.log(tf.clip_by_value(
            action_probs, clip_value_min=1e-8, clip_value_max=action_probs.dtype.max))

        return actions, action_probs, log_action_probs

    def call(self, state):
        if not self.shared:
            state = self.conv(state)
        return self.head(state)
