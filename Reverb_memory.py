import reverb
import config
from reverb.tf_client import TFClient
import tensorflow as tf

table_signature = {'state': tf.TensorSpec(shape=(2, *config.NN_INPUT_SHAPE[-3:]), dtype=tf.int32),
                   'action': tf.TensorSpec(shape=(2), dtype=tf.int32),
                   'reward': tf.TensorSpec(shape=(2), dtype=tf.float32),
                   'done': tf.TensorSpec(shape=(2), dtype=tf.bool)}


class Memory:
    def __init__(self) -> None:
        self.server = reverb.Server(tables=[
            reverb.Table(
                name='my_table',
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=100,
                rate_limiter=reverb.rate_limiters.MinSize(1),
                signature=table_signature
            )])
        self.client = reverb.Client(f'localhost:{self.server.port}')
        self.dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=f'localhost:{self.server.port}',
            table='my_table',
            max_in_flight_samples_per_worker=10).batch(config.batch_size)
        self._cached_step = 0
        self._writer = self.client.trajectory_writer(num_keep_alive_refs=2)

    def append(self, state, action, reward, done):
        self._writer.append({'state': state,
                            'action': action,
                             'reward': reward,
                             'done': done
                             })
        self._cached_step += 1
        if self._cached_step >= 2:
            self._writer.create_item(
                table='my_table',
                priority=1.0,
                trajectory={
                    'state': self._writer.history['state'][-2:],
                    'action': self._writer.history['action'][-2:],
                    'reward': self._writer.history['reward'][-2:],
                    'done': self._writer.history['done'][-2:]
                })
            self._writer.flush()

    def _postprocess(self, data):
        states = data['state'][:,0,:]
        actions = data['action'][:,0]
        rewards = data['reward'][:,1]
        next_states = data['state'][:,1,:]
        dones = data['done'][:,1]
        return (states, actions, rewards, next_states, dones)

    def sample(self):
        for item in self.dataset.take(1):
            result = item
            break
        self.last_sampled_item = result.info.key
        data = self._postprocess(result.data)
        return data, tf.cast(result.info.priority,tf.float32)

    def update_priority(self, priorities):
        self.client.mutate_priorities(
            'my_table', zip(self.last_sampled_item, priorities))

    def clear_cache(self):
        self._cached_step = 0


if __name__ == '__main__':
    state = tf.zeros(shape=config.NN_INPUT_SHAPE[-3:], dtype=tf.int32)
    action = tf.zeros(shape=(), dtype=tf.int32)
    reward = tf.zeros(shape=(), dtype=tf.float32)
    next_state = tf.zeros(shape=config.NN_INPUT_SHAPE[-3:], dtype=tf.int32)
    done = tf.zeros(shape=(), dtype=tf.bool)
    a = Memory()
    a.append(state, action, reward, next_state, done)
    # for sample in a.dataset.take(1):
    #     # Results in the following format.
    #     print(sample.info.key)              # ([2], uint64)
    #     print(sample.info.probability)      # ([2], float64)
    a.sample()
    pass
