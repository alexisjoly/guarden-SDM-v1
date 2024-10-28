import itertools
import numpy as np
import os
import math
from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



EQUATOR_ARC_SECOND_IN_METERS = 30.87  # meters


class PatchProvider(object):
    def __init__(self, size, normalize) -> None:
        self.patch_size = size
        self.normalize = normalize
        self.nb_layers = 0
        self.transformer = None
        
    @abstractmethod
    def __getitem__(self, item):
        pass
    
    def __repr__(self):
        return self.__str__()
    
    @abstractmethod
    def __str__(self):
        pass
    
    def __len__(self):
        return self.nb_layers
    
    def plot_patch(self, item, save=None):
        patch = self[item]
        nb_plot = self.nb_layers
        names = self.layers_names
        self._plot_patch_core(item, patch, nb_plot, names, save)
    
    def _plot_patch_core(self, item, patch, nb_plot, names, save, nb_additional=0):
        if self.nb_layers==1:
            plt.figure(figsize=(10, 10))
            plt.imshow(patch[0])
        else:
            # calculate the number of rows and columns for the subplots grid
            rows = int(math.ceil(math.sqrt(nb_plot)))
            cols = int(math.ceil(nb_plot / rows))

            # create a figure with a grid of subplots
            fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

            # flatten the subplots array to easily access the subplots
            axs = axs.flatten()

            # loop through the layers of patch data
            idx = 0
            for i in range(nb_plot-nb_additional):
                # display the layer on the corresponding subplot
                axs[idx].imshow(patch[idx])
                axs[idx].set_title('layer_{}: {}'.format(idx, names[idx]))
                axs[idx].axis('off')
                idx+=1
            for i in range(nb_additional):
                axs[idx].imshow(patch[idx])
                axs[idx].set_title(str(names[idx]))
                axs[idx].axis('off')
                idx+=1

            # remove empty subplots
            for i in range(nb_plot, rows*cols):
                fig.delaxes(axs[i])

        plt.suptitle('Tensor for item: '+str(item), fontsize=16)

        # show the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save:
            plt.savefig(save)
        else:
            plt.show()

class MultiPatchProvider(PatchProvider):
    def __init__(self, providers, transform=None):
        self.providers = providers
        self.nb_layers = sum([len(provider) for provider in self.providers])
        self.layers_names = list(itertools.chain.from_iterable([provider.layers_names for provider in self.providers]))
        self.transform = transform
    
    def __getitem__(self, item):
        patch = np.concatenate([provider[item] for provider in self.providers])
        if self.transform:
            patch = self.transform(patch)
        return patch
    
    def __str__(self):
        result = 'Providers:\n'
        for provider in self.providers:
            result += str(provider)
            result += '\n'
        return result

class GLC23AltiPatchProviderNPY(PatchProvider):
    def __init__(self, root_path) -> None:
        super().__init__(128, True)
        self.root_path = root_path
        self.layers_names = ['elevation']

    def __getitem__(self, item):
        patch_id = int(item['patchID'])
        path = os.path.join(self.root_path, str(patch_id)[-2:], str(patch_id)[-4:-2], str(patch_id)+'.npy')
        return np.load(path)

class JpegPatchProvider(PatchProvider):
    """JPEG patches provider for GLC23.
    
    Provides tensors of multi-modal patches from JPEG patch files
    of rasters of the GLC23 challenge.

    Attributes:
        PatchProvider (_type_): _description_
    """
    def __init__(self, root_path, select=None, normalize=True, patch_transform=None, size=128, dataset_stats='jpeg_patches_stats.csv', ndvi=False):
        """Class constructor.

        Args:
            root_path (str): root path to the directory containg all patches modalities
            channel_list (list, optional): list of channels to provide for the output tensor. Defaults to None.
            normalize (bool, optional): normalize data. Defaults to False.
            patch_transform (callable, optional): custom transformation functions. Defaults to None.
            size (int, optional): default tensor sizes (must match patch sizes). Defaults to 128.
            dataset_stats (str, optional): path to the csv file containing the mean and std values of the
                                           jpeg patches dataset if `normalize` is True. If the file doesn't
                                           exist, the values will be calculated and the file will be created once.
                                           Defaults to 'jpeg_patches_stats.csv'
        """
        super().__init__(size, normalize)
        self.patch_transform = patch_transform
        self.root_path = root_path
        self.ext = '.jpeg'
        self.dataset_stats = os.path.join(self.root_path, dataset_stats)

        self.channel_folder = {'red': 'rgb', 'green': 'rgb', 'blue': 'rgb',
                          'swir1':'swir1',
                          'swir2':'swir2',
                          'nir':'nir'}
        if not select:
            sub_dirs = next(os.walk(root_path))[1]
            select = [k for k,v in self.channel_folder.items() if v in sub_dirs]

        self.channels = [c.lower() for c in select]
        self.ndvi = ndvi
        if self.ndvi:
            self.channels.append('ndvi')
        self.nb_layers = len(self.channels)
        self.bands_names = self.channels
        self.layers_names = self.bands_names

    def __getitem__(self, item):
        """Return a tensor composed of every channels of a jpeg patch.

        Args:
            item (dict): dictionnary containing the patchID necessary to 
                         identify the jpeg patch to return.

        Raises:
            KeyError: the 'patchID' key is missing from item
            Exception: item is not a dictionnary as expected

        Returns:
            (tensor): multi-channel patch tensor.
        """
        try:
            id_ = str(int(item['patchID']))
        except KeyError as e:
            raise KeyError('The patchID key does not exists.')
        except Exception as e:
            raise Exception('An error has occured when trying to load a patch patchID.'
                            'Check that the input argument is a dict containing the "patchID" key.')

        # folders that contain patches
        sub_folder_1 = id_[-2:]
        sub_folder_2 = id_[-4:-2]
        list_tensor = {'order': [], 'tensors':[]}

        for channel in self.channels:
            if channel not in list_tensor['order'] and channel != 'ndvi':
                path = os.path.join(self.root_path, self.channel_folder[channel], sub_folder_1, sub_folder_2, id_+self.ext)
                try:
                    img = np.asarray(Image.open(path))
                    if set(['red','green','blue']).issubset(self.channels) and channel in ['red','green','blue']:
                        img = img.transpose((2,0,1))
                        list_tensor['order'].extend(['red','green','blue'])
                    else:
                        if channel in ['red','green','blue']:
                            img = img[:,:,'rgb'.find(channel[0])]
                        img = np.expand_dims(img, axis=0)
                        list_tensor['order'].append(channel)
                except Exception as e:
                    print(e)
                    img = np.zeros((1, self.patch_size, self.patch_size))
                    list_tensor['order'].append(channel)
                if self.normalize:
                    img = (img-97.25338302612305)/40.70420644799345
                for depth in img:
                    list_tensor['tensors'].append(np.expand_dims(depth, axis=0))
        tensor = np.concatenate(list_tensor['tensors'])
        if channel == 'ndvi':
            ndvi = np.expand_dims((tensor[3]-tensor[0])/(tensor[3]+tensor[0]), axis=0)
            tensor = np.concatenate((tensor, ndvi))
        if self.patch_transform:
            for transform in self.patch_transform:
                tensor = transform(tensor)
        #self.channels = list_tensor['order']
        self.n_rows = img.shape[1]
        self.n_cols = img.shape[2]
        return tensor

    def __str__(self):
        result = '-' * 50 + '\n'
        result += 'n_layers: ' + str(self.nb_layers) + '\n'
        result += 'n_rows: ' + str(self.n_rows) + '\n'
        result += 'n_cols: ' + str(self.n_cols) + '\n'
        result += '-' * 50
        return result