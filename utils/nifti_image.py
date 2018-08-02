'''
Author: Samuel Remedios

Extends ImageDataGenerator and flow_from_directory() to handle the NIfTI file format
by means of the NiBabel package.

Extension of the keras-preprocessing image utilities available at:
https://github.com/keras-team/kears-preprocessing/blob/master/keras_preprocessing/image.py
'''

from keras.preprocessing.image import *
import nibabel as nib


class NIfTIDirectoryIterator(DirectoryIterator):
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256, 256), num_channels=1,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 follow_links=False, split=None):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.num_channels = num_channels
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode

        # Counter number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                   white_list_formats=white_list_formats,
                                   follow_links=follow_links,
                                   split=None)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))
        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))

        # Build an index of the images in the different class subfolders
        results = []
        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(_list_valid_filenames_in_directory,
                                 (dirpath, white_list_formats, split,
                                  self.class_indices, follow_links)))
        for res in results:
            classes, filenames = res.get()
            self.classes[i:i + len(classes)] = classes
            self.filenames += filenames
            i += len(classes)

        pool.close()
        pool.join()
        super(NIfTIDirectoryIterator, self).__init__(self.samples,
                                                     batch_size,
                                                     shuffle,
                                                     seed)
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
                (len(index_array),) + self.image_shape,
                dtype=backend.floatx())
        # build batch of image data
        for i,j in enumerate(index_array):
            fname = self.filenames[j]
            img = nib.load(os.path.join(self.directory, fname))
            # TODO: pad/crop image to target_size here
            x = img.get_data()
            # TODO: extensible image augmentation applied here
            batch_x[i] = x

        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(backend.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                    (len(batch_x), self.num_classes),
                    dtype=backend.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


    # TODO: change some stuff to work with nibabel


class NIfTIImageDataGenerator(ImageDataGenerator):
    # TODO: change some stuff to work with nibabel


class NIfTINumpyArrayIterator(NumpyArrayIterator):
    # TODO: change some stuff to work with nibabel
