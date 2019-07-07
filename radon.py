import numpy as np
import chainer
from chainer import link
from warnings import warn
from abc import ABCMeta, abstractmethod

class Projector(link.Chain, metaclass=ABCMeta):
    """ Abstract class of the projector for 2D-images """

    @abstractmethod
    def forward(self, x):
        pass

    def _check_type_forward(self, x):

        assert x.ndim == 4, 'input tensor should be 4-D.'
        b, c, w, h = x.shape
        assert c == 1, 'input image should be grayscale.'

class ParallelProjector(Projector):

    def __init__(self, axis=2, keepdims=True):

        self._axis = axis
        self._keepdims = keepdims

    def forward(self, x):

        self._check_type_forward(x)
        return chainer.functions.sum(x, self._axis, self._keepdims)

class OrthogonalProjector(Projector):
    pass


class Radon(link.Chain):
    """ Radon transform of 2D-images given specified projection angles. """

    def __init__(self, theta=None):
        super(Radon, self).__init__()

        if theta is None:
            theta = np.arange(180)
        self._theta = theta

        self._projector = ParallelProjector(axis=2, keepdims=True)

    def _check_type_forward(self, x):

        assert x.ndim == 4, 'input tensor should be 4-D.'
        b, c, w, h = x.shape
        assert c == 1, 'input image should be grayscale.'
        assert w == h, 'input image should be square.'

    def _build_rotation(self, theta, batch_size):

        T = np.deg2rad(theta)
        R = np.array([[np.cos(T), np.sin(T), 0],
                        [-np.sin(T), np.cos(T), 0],
                        [0, 0, 1]])
        R = R[:-1,:].astype(np.float32)
        R = self.xp.asarray(R[np.newaxis])
        return chainer.functions.repeat(R, batch_size, axis=0)

    def forward(self, x):
        """Applies the radon transform.
        Args:
            x (~chainer.Variable): Batch of input images.
        Returns:
            ~chainer.Variable: Batch of output sinograms.
        """
        self._check_type_forward(x)

        b, c, w, h = x.shape

        ret = []

        for i, th in enumerate(self._theta):

            matrix = self._build_rotation(th, b)
            grid = chainer.functions.spatial_transformer_grid(matrix, (w,h))
            rotated = chainer.functions.spatial_transformer_sampler(x, grid)
            raysum = self._projector(rotated)

            ret.append(raysum)

        ret = chainer.functions.concat(ret, axis=2)

        return ret

def preprocess(image, circle=True):

    if circle:
        radius = min(image.shape) // 2
        c0, c1 = np.ogrid[0:image.shape[0], 0:image.shape[1]]
        reconstruction_circle = ((c0 - image.shape[0] // 2) ** 2
                                 + (c1 - image.shape[1] // 2) ** 2)
        reconstruction_circle = reconstruction_circle <= radius ** 2
        if not np.all(reconstruction_circle | (image == 0)):
            warn('Radon transform: image must be zero outside the '
                 'reconstruction circle')
        # crop image to make it square
        slices = []
        for d in (0, 1):
            if image.shape[d] > min(image.shape):
                excess = image.shape[d] - min(image.shape)
                slices.append(slice(int(np.ceil(excess / 2)),
                                    int(np.ceil(excess / 2)
                                        + min(image.shape))))
            else:
                slices.append(slice(None))
        slices = tuple(slices)
        padded_image = image[slices]
    else:
        diagonal = np.sqrt(2) * max(image.shape)
        pad = [int(np.ceil(diagonal - s)) for s in image.shape]
        new_center = [(s + p) // 2 for s, p in zip(image.shape, pad)]
        old_center = [s // 2 for s in image.shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = np.pad(image, pad_width, mode='constant',
                              constant_values=0)

    return padded_image

if __name__ == '__main__':


    import argparse
    import cv2

    parser = argparse.ArgumentParser(description='Radon Transfrom')
    parser.add_argument('--image', '-i', type=str, default='phantom.png')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--trial', '-t', type=int, default=20, help='number of trails')
    parser.add_argument('--angle', '-a', type=int, default=500, help='number of angles')
    parser.add_argument('--slice', '-s', type=int, default=1, help='number of slices, used for debugging')

    args = parser.parse_args()

    # setup an input slice
    image = cv2.imread(args.image)[:,:,0]
    image = preprocess(image)
    image = cv2.resize(image, (512,512))

    # convert to a volume for debugging.
    volume = image[:,:,np.newaxis]
    volume = np.repeat(volume, args.slice, axis=2)

    w, h, z = volume.shape

    # convert to a tensor
    b = c = 1
    x = volume.reshape(b,c,w,h,z).astype(np.float32)

    # reshape the tensor: [b(=1),c(=1),w,h,z] -> [b*z(=z),c(=1),w,h]
    x = x.transpose(0,4,1,2,3)
    x = x.reshape(b*z,c,w,h)

    # to gpu
    if args.gpu >= 0:
        import cupy as xp
        x = xp.asarray(x)
    x = chainer.Variable(x)
    print(x.shape)

    # do
    radon = Radon(theta=np.linspace(0,180,args.angle))

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        radon.to_gpu()

    import tqdm
    for _ in tqdm.tqdm(range(args.trial)):
        ret = radon(x)
    print(ret.shape)

    # visualize a graph
    import chainer.computational_graph as c
    g = c.build_computational_graph(ret)
    with open('graph.dot', 'w') as o:
        o.write(g.dump())

    # to cpu
    ret = ret.data
    if args.gpu >= 0:
        ret = ret.get()

    # visualize an sinogram respect to the first slice
    import matplotlib.pyplot as plt

    plt.figure(figsize=(19,6))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(ret[0,0], cmap='gray')
    plt.colorbar()
    plt.show()
