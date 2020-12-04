import os


def delete_checkpoint_with_index(index):
    prefix = 'ckpts/ckpt_' + str(index) + '.'
    folder = os.path.dirname(prefix)
    prefix_only_name = os.path.basename(prefix)
    assert os.path.isdir(folder)
    for name in os.listdir(folder):
        if prefix_only_name in name:
            file_path = os.path.join(folder, name)
            os.remove(file_path)
