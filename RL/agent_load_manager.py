
import os
import os.path as osp
import tensorflow as tf

class AgentManager:
    def __init__(self,agent, load_path, save_path):
        print("Save PATH;{}".format(save_path))
        print("Load PATH;{}".format(load_path))

        save_path = osp.expanduser(save_path)
        self.checkpoint_prefix = os.path.join(save_path, "trained_model_")
        ckpt = tf.train.Checkpoint(model=agent)
        if load_path is not None:
            load_path = osp.expanduser(load_path)
            manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
            if manager.latest_checkpoint:
                ckpt.restore(manager.latest_checkpoint)
                print("Model Restored{}".format(manager.latest_checkpoint))
        self.ckpt = ckpt
    def save(self):
        save_file = self.ckpt.save(file_prefix=self.checkpoint_prefix)
        print('Model saved to ', save_file)
        print('\n')
