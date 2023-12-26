import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Conv2D, Flatten, BatchNormalization, Add, ReLU
from keras.layers import Dense as _Dense
from keras.initializers import HeUniform
from tf_agents.networks import network, actor_distribution_network
from tf_agents.utils import common, nest_utils
from tf_agents.networks import categorical_projection_network
from tf_agents.networks import utils


class BaseNetwork(keras.Model):
    def save(self, path):
        self.save_weights(path)

    def load(self, path):
        self.load_weights(path)


def Dense(units, activation=None):
    return _Dense(
        units=units,
        activation=activation,
        kernel_initializer=HeUniform())


class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size=3, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                            kernel_initializer=HeUniform(), activation=None)
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                            kernel_initializer=HeUniform(), activation=None)
        self.bn2 = BatchNormalization()
        self.add = Add()
        self.relu2 = ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.add([x, inputs])
        x = self.relu2(x)
        return x


class DQNBase(BaseNetwork):

    def __init__(self, num_residual_blocks=3):
        super(DQNBase, self).__init__()
        self.input_preprocess = keras.Sequential([
            Conv2D(filters=64, kernel_size=3, strides=1,
                   padding='same', activation=None),
            BatchNormalization(),
            ReLU(),
        ])
        self.net = keras.Sequential([ResidualBlock(64)
                                    for _ in range(num_residual_blocks)])

    def call(self, states, training):
        # add batch dimension
        # states = tf.expand_dims(states, axis=-1)
        if states.shape.rank == 1:
            states = tf.expand_dims(states, axis=0)
        states = self.input_preprocess(states, training)
        return self.net(states, training)


class ValueNetwork(network.Network):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 observation_and_action_constraint_splitter=None,
                 name='QNetwork'):
        super(ValueNetwork, self).__init__(
            input_tensor_spec=observation_spec, state_spec=(), name=name)
        # For simplicity we will only support a single action float output.
        self._action_spec = action_spec
        self.observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError(
                'Only a single action is supported by this network')
        self._single_action_spec = flat_action_spec[0]

        self._base_layer = DQNBase()
        self._post_layer = keras.Sequential([
            Conv2D(filters=1, kernel_size=1, strides=1,
                   padding='same', activation=None),
            BatchNormalization(),
            ReLU(),
            Flatten(),
            Dense(128, 'relu'),
            Dense(1),
        ])

    def call(self, observations, step_types=None, training=False, network_state=()):
        outer_rank = nest_utils.get_outer_rank(
            observations, self.input_tensor_spec)
        # We use batch_squash here in case the observations have a time sequence
        # compoment.
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(
            batch_squash.flatten, observations)
        if self.observation_and_action_constraint_splitter is not None:
            observations, _ = self.observation_and_action_constraint_splitter(
                observations)
        # if observations._rank()==2:
        #     observations = tf.expand_dims(observations, axis=0)
            # if self.observation_and_action_constraint_splitter is not None:
            #     mask = tf.expand_dims(mask, axis=0)
        observations = tf.cast(observations, tf.float32)
        state = self._base_layer(observations, training)
        pred_Q = self._post_layer(state, training)
        pred_Q = tf.squeeze(pred_Q, axis=-1)
        pred_Q = batch_squash.unflatten(pred_Q)
        return pred_Q, network_state


class ActorNetwork(network.Network):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 observation_and_action_constraint_splitter=None,
                 name='PolicyNetwork'):
        super(ActorNetwork, self).__init__(
            input_tensor_spec=observation_spec, state_spec=(), name=name)
        self.observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        self.conv = DQNBase()

        self.head = keras.Sequential([
            Conv2D(filters=2, kernel_size=1, strides=1,
                   padding='same', activation=None),
            BatchNormalization(),
            ReLU(),
            Flatten(),
        ])
        self.proj_net = categorical_projection_network.CategoricalProjectionNetwork(
            action_spec)

    def call(self, observations, training):
        # We use batch_squash here in case the observations have a time sequence
        # compoment.
        outer_rank = nest_utils.get_outer_rank(
            observations, self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        mask = None
        if self.observation_and_action_constraint_splitter is not None:
            observations, mask = self.observation_and_action_constraint_splitter(
                observations)
        observations = tf.nest.map_structure(
            batch_squash.flatten, observations)

        observations = tf.cast(observations, tf.float32)
        output = self.head(self.conv(observations, training), training)

        output = batch_squash.unflatten(output)
        return self.proj_net(output, outer_rank=0, training=training, mask=mask)
