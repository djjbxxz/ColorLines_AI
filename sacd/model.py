import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils
from tf_agents.utils import common as common_utils
from tf_agents.networks.encoding_network import _copy_layer
from tf_agents.specs.distribution_spec import DistributionSpec


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


preprocessing_layer = [
    Conv(64, 3, activation="relu", padding="same"),
    Conv(128, 3, activation="relu", padding="same"),
    Conv(64, 3, activation="relu", padding="same"),
    Conv(32, 3, activation="relu", padding="valid"),
    Conv(32, 3, activation="relu", padding="valid"),
    layers.Flatten(),
]


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
        # add batch dimension
        if states.shape.rank == 3:
            states = tf.expand_dims(states, axis=0)
        return self.net(states)


class QNetwork(network.Network):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 name='QNetwork'):
        # observation_spec = {observation_spec,tf.TensorSpec(shape=(), dtype=tf.int32)}
        super(QNetwork, self).__init__(
            input_tensor_spec=observation_spec, state_spec=(), name=name)

        # For simplicity we will only support a single action float output.
        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError(
                'Only a single action is supported by this network')
        self._single_action_spec = flat_action_spec[0]

        # kernel_initializer = keras.initializers.HeUniform()
        # bias_initializer = 'zeros'

        # kernel_initializer = tf.keras.initializers.VarianceScaling(
        #     scale=1. / 3., mode='fan_in', distribution='uniform')
        # initializer = tf.keras.initializers.RandomUniform(
        #     minval=-0.003, maxval=0.003)

        self._base_layer = DQNBase()

        self._post_layer = keras.Sequential([
            Dense(256, 'relu'),
            Dense(config.POTENTIAL_MOVE_NUM),
        ])

    def call(self, observations, network_state=(), action=None):
        outer_rank = nest_utils.get_outer_rank(
            observations, self.input_tensor_spec)
        # We use batch_squash here in case the observations have a time sequence
        # compoment.
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(
            batch_squash.flatten, observations)

        observations = tf.cast(observations, tf.float32)
        state = self._base_layer(observations)
        pred_Q = self._post_layer(state)

        if not action is None:
            pred_Q = tf.gather(pred_Q, action, axis=1, batch_dims=1)

        # actions = common_utils.scale_to_spec(actions, self._single_action_spec)
        # actions = batch_squash.unflatten(actions)
        return pred_Q, network_state


class TwinnedQNetwork(network.Network):
    def __init__(self,
                 shared=False,
                 dueling_net=False
                 ):
        super().__init__()
        self.Q1 = QNetwork(config.POTENTIAL_MOVE_NUM, shared, dueling_net)
        self.Q2 = QNetwork(config.POTENTIAL_MOVE_NUM, shared, dueling_net)

    def create_variables(self, training):
        self.Q1.create_variables(training=training)
        self.Q2.create_variables(training=training)

    def call(self, states, training=False):
        q1 = self.Q1(states, training=training)
        q2 = self.Q2(states, training=training)
        return q1, q2


class ActorNetwork(network.Network):

    def __init__(self,
                 observation_spec,
                 name='PolicyNetwork'):
        super(ActorNetwork, self).__init__(
            input_tensor_spec=observation_spec, state_spec=(), name=name)
        self.conv = DQNBase()

        self.head = keras.Sequential([
            Dense(256, 'relu'),
            Dense(config.POTENTIAL_MOVE_NUM),
        ])

    # def act(self, states):
    #     if not self.shared:
    #         states = self.conv(states)

    #     action_logits = self.head(states)
    #     greedy_actions = tf.argmax(
    #         action_logits, axis=1)
    #     return greedy_actions

    # def sample(self, states,training):
    #     if not self.shared:
    #         states = self.conv(states,training=training)

    #     action_probs = tf.math.softmax(self.head(states), axis=1)
    #     actions = tf.random.categorical(action_probs, 1)[0, 0]

    #     # Avoid numerical instability.
    #     log_action_probs = tf.math.log(tf.clip_by_value(
    #         action_probs, clip_value_min=1e-8, clip_value_max=action_probs.dtype.max))

    #     return actions, action_probs, log_action_probs

    def call(self, state):
        state = tf.cast(state, tf.float32)
        return self.head(self.conv(state)), ()
