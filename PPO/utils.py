import yaml
import threading
import logging
import os
import time
import numpy as np
import tensorflow as tf
import shutil


def isDebug():
    import sys
    return True if sys.gettrace() else False


class Logger():
    def __init__(self, logdir):
        self.logdir = logdir
        self.tf_writer = None
        self.start_time = time.time()
        self.n_eps = 0
        self.is_tb_enabled = not isDebug()  # not isDebug()

        if self.is_tb_enabled:
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
            # create_noop_writer
            self.writer = tf.summary.create_file_writer(self.logdir)
            self.writer.set_as_default()

            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(self.logdir + '/logger.log'),
                ],
                datefmt='%Y/%m/%d-%I:%M:%S'
            )
            # self.save_config(self.logdir)

    def log_str(self, content):
        if self.is_tb_enabled:
            logging.info(content)
        else:
            print(f"{self.get_current_time_str()}  {content}")

    def get_current_time_str(self) -> str:
        return time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))

    def add_scalar(self, tag, scalar_value, global_step):
        if self.is_tb_enabled:
            tf.summary.scalar(name=tag, data=scalar_value, step=global_step)

    def eval_log(self, step, episode_length, episode_reward):
        with tf.name_scope('Eval'):
            self.add_scalar(tag='avg_episode_length',
                            scalar_value=episode_length, global_step=step)
            self.add_scalar(tag='avg_episode_reward',
                            scalar_value=episode_reward, global_step=step)
        self.log_str(
            f"step {step} | avg_episode_length={episode_length:.3f} | avg_episode_reward={episode_reward:.3f}")

    def save_config(self, file_to_copy: list[str]):
        '''
        Save config files to logdir
        file_to_copy: list[str]
            list of file path to copy to logdir
            For example: ['PPO/config.yaml', 'environment/config.yaml']
        '''
        path = self.logdir
        self.target_file_list = []
        for file in file_to_copy:
            target_file = os.path.join(path, file.replace('/', '_'))
            shutil.copy(file, target_file)
            self.target_file_list.append(target_file)


def reduce_gpu_memory_usage():
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('Memory efficiency enabled')
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


class Checkpoint:
    def __init__(self, dir: str, save_interval,
                 policy,
                 step_counter
                 ) -> None:
        self._path = dir
        self.train_step_counter = step_counter
        self._save_interval = save_interval
        self._is_Debug = isDebug()
        self._checkpoint = tf.train.Checkpoint(
            policy=policy,
            step_counter=step_counter,
        )
        self._latest_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=os.path.join(dir, 'latest'), max_to_keep=5000, checkpoint_interval=save_interval, step_counter=step_counter)
        self._best_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=os.path.join(dir, 'best'), max_to_keep=20)

    def load(self, path, catergory: str = 'latest'):
        if os.path.isdir(path):
            if path.endswith('latest'):
                path = os.path.dirname(path)
                catergory = 'latest'
            elif path.endswith('best'):
                path = os.path.dirname(path)
                catergory = 'best'

            ckp_manager = tf.train.CheckpointManager(
                self._checkpoint, directory=os.path.join(path, catergory),max_to_keep=None)

            self._checkpoint.restore(ckp_manager.latest_checkpoint).assert_consumed()
            print(f'Using checkpointer from {ckp_manager.latest_checkpoint}')
        elif os.path.isfile(path):
            self._checkpoint.restore(path[:-6]).assert_consumed()
            print(f'Using checkpointer from {path}')
        else:
            print(f'No checkpoint found at {path}')
            raise FileNotFoundError
    def save_best(self):
        if not self._is_Debug:
            self._best_manager.save(checkpoint_number=self.train_step_counter,check_interval=False)

    def save_latest(self):
        if not self._is_Debug:
            self._latest_manager.save(checkpoint_number=self.train_step_counter)


class FileWatcher:
    '''
    Watch a file for changes. The watcher will run in a separate thread.
    '''

    def __init__(self, filename, _func: callable, watch_interval=1):
        self.filename = filename
        self.interval = watch_interval
        self.last_modified = os.path.getmtime(filename)
        self._func: callable = _func
        self.should_stop = False

    def _watch_loop(self):
        while not self.should_stop:
            current_modified = os.path.getmtime(self.filename)
            if current_modified != self.last_modified:
                self.last_modified = current_modified
            self._func()
            time.sleep(self.interval)

    def start(self):
        self.should_stop = False
        self._thread = threading.Thread(target=self._watch_loop)
        self._thread.start()

    def stop(self):
        self.should_stop = True
        self._thread.join()


class HyperParamsWatcher:
    def __init__(self, config_file_to_watch: str, params_mapping_dict: dict, agent, watch_interval=1):
        '''
        Monitor a file for params changes. The watcher will run in a separate thread.
        params_mapping_dict: dict, mapping name from config file to agent's variables names. eg: {"learning_rate": '_learning_rate'}
        '''
        self.watcher = FileWatcher(
            config_file_to_watch, self.onchanged, watch_interval)
        self.params_mapping_dict = params_mapping_dict
        self.agent = agent
        print(f"HyperParamsWatcher: watching at {config_file_to_watch}")

    def onchanged(self):
        '''
        Called when the file is changed.
        '''
        with open(self.watcher.filename) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        updates = self.updateparams(config)
        return updates

    def updateparams(self, yaml_config: dict):
        '''
        Update the params of the agent.
        '''
        change = []
        any_param_changed = False
        for new_param_key, new_param_value in yaml_config.items():
            if new_param_key not in self.params_mapping_dict:
                continue
            old_param_value = getattr(
                self.agent, self.params_mapping_dict[new_param_key])
            if old_param_value != new_param_value:
                setattr(
                    self.agent, self.params_mapping_dict[new_param_key], new_param_value)
                any_param_changed = True
                print(
                    f"Update {new_param_key} from {old_param_value} to {new_param_value}")
            change.append(
                (self.params_mapping_dict[new_param_key], (old_param_value, new_param_value)))
        if any_param_changed:
            if self.agent.graph_enabled:
                self.agent.useGraphOptimization()
        return change

    def start(self):
        self.watcher.start()

    def stop(self):
        self.watcher.stop()
