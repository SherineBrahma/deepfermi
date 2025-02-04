import shutil
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


def Save(save_path):
    """
    Returns a function that saves various types of objects
    (e.g., model weights, variables, figures) concerning this project to
    the specified directory, supporting optional subfolders and custom
    parameters.
    """

    save_data_dir = Path(save_path)
    Path(save_data_dir).mkdir(parents=True, exist_ok=True)

    def save_func(save_obj, save_obj_type, save_obj_name, **kwargs):

        if save_obj_type == 'model':
            if kwargs.get('sub_folder') is None:
                torch.save(
                    save_obj.state_dict(),
                    Path.joinpath(save_data_dir, save_obj_name)
                    )
            else:
                folder = kwargs['sub_folder']
                sub_folder = Path.joinpath(save_data_dir, folder)
                Path(sub_folder).mkdir(parents=True, exist_ok=True)
                torch.save(
                    save_obj.state_dict(),
                    Path.joinpath(sub_folder, save_obj_name)
                    )

        elif save_obj_type == 'var':
            if kwargs.get('sub_folder') is None:
                np.save(Path.joinpath(save_data_dir, save_obj_name), save_obj)
            else:
                folder = kwargs['sub_folder']
                sub_folder = Path.joinpath(save_data_dir, folder)
                Path(sub_folder).mkdir(parents=True, exist_ok=True)
                np.save(Path.joinpath(sub_folder, save_obj_name), save_obj)

        elif save_obj_type == 'fig':
            save_obj.savefig(
                Path.joinpath(save_data_dir, save_obj_name),
                dpi=500
                )

        elif save_obj_type == 'numpy':
            np.save(
                Path.joinpath(save_data_dir, save_obj_name),
                save_obj
                )

        elif save_obj_type == 'train_params':

            with open(Path.joinpath(save_data_dir, save_obj_name),
                      'w',
                      encoding='utf-8') as file:
                save_obj_dic = vars(save_obj)
                for k in (save_obj_dic.keys()):
                    if k.startswith('_'):
                        continue
                    file.write("%s : %s \n" % (k, save_obj_dic[k]))

        elif save_obj_type == 'text':

            if kwargs.get('mode') is None:
                mode = 'w'
            else:
                mode = kwargs['mode']
            with open(Path.joinpath(save_data_dir, save_obj_name),
                      mode,
                      encoding='utf-8') as file:
                file.write(save_obj)

        elif save_obj_type == 'gif':

            if kwargs.get('clim') is None:
                clim = [0, 8]
            else:
                clim = kwargs['clim']

            if kwargs.get('cmap') is None:
                cmap = 'gray'
            else:
                cmap = kwargs['cmap']
            gif_dir = Path.joinpath(save_data_dir, 'gif_cache')
            Path(gif_dir).mkdir(parents=True, exist_ok=True)
            for n_image in range(save_obj.shape[2]):
                x_curr = save_obj[:, :, n_image]
                fig = plt.figure()
                im = plt.imshow(x_curr, cmap=cmap)
                plt.axis('off')
                im.set_clim(clim[0], clim[1])
                plt.colorbar(im)
                gif_name = str(n_image)
                gif_name = str.zfill(
                    gif_name,
                    int(np.floor(np.log10(save_obj.shape[2]) + 1))
                    )
                fig.savefig(Path.joinpath(gif_dir, gif_name), dpi=500)
                plt.close()
            filenames = list(sorted(Path(gif_dir).glob('*.png*')))
            images = []
            for filename in filenames:
                images.append(imageio.imread(filename))
            imageio.mimsave(
                Path.joinpath(save_data_dir, save_obj_name),
                images
                )
            shutil.rmtree(gif_dir)

    return save_func
