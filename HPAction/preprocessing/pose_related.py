import numpy as np

from .utils import Compose

class PreNormalize2D:
    """Normalize the range of keypoint values. """

    def __init__(self, img_shape=(1080, 1920), threshold=0.01, mode='fix', concatenate=True):
        self.threshold = threshold
        # Will skip points with score less than threshold
        self.img_shape = img_shape
        self.mode = mode
        self.concatenate = concatenate
        assert mode in ['fix', 'auto']

    def __call__(self, results):
        mask, maskout, keypoint_score, keypoint,  = None, None, None, results['keypoint'].astype(np.float32)
        if 'keypoint_score' in results:
            keypoint_score = results.pop('keypoint_score').astype(np.float32)
            if self.concatenate:
                keypoint = np.concatenate([keypoint, keypoint_score[..., None]], axis=-1)
                
        if keypoint.shape[-1] == 3:
            mask = keypoint[..., 2] > self.threshold
            maskout = keypoint[..., 2] <= self.threshold
        elif keypoint_score is not None:
            mask = keypoint_score > self.threshold
            maskout = keypoint_score <= self.threshold

        if self.mode == 'auto':
            if mask is not None:
                if np.sum(mask):
                    x_max, x_min = np.max(keypoint[mask, 0]), np.min(keypoint[mask, 0])
                    y_max, y_min = np.max(keypoint[mask, 1]), np.min(keypoint[mask, 1])
                else:
                    x_max, x_min, y_max, y_min = 0, 0, 0, 0
            else:
                x_max, x_min = np.max(keypoint[..., 0]), np.min(keypoint[..., 0])
                y_max, y_min = np.max(keypoint[..., 1]), np.min(keypoint[..., 1])
            if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                keypoint[..., 0] = (keypoint[..., 0] - (x_max + x_min) / 2) / (x_max - x_min) * 2
                keypoint[..., 1] = (keypoint[..., 1] - (y_max + y_min) / 2) / (y_max - y_min) * 2
        else:
            h, w = results.get('img_shape', self.img_shape)
            keypoint[..., 0] = (keypoint[..., 0] - (w / 2)) / (w / 2)
            keypoint[..., 1] = (keypoint[..., 1] - (h / 2)) / (h / 2)

        if maskout is not None:
            keypoint[..., 0][maskout] = 0
            keypoint[..., 1][maskout] = 0
        results['keypoint'] = keypoint

        return results
    
class GenSkeFeat:
    def __call__(self, results):
        if 'keypoint_score' in results and 'keypoint' in results:
            assert results['keypoint'].shape[-1] == 2, 'Only 2D keypoints have keypoint_score. '
            keypoint = results.pop('keypoint')
            keypoint_score = results.pop('keypoint_score')
            results['keypoint'] = np.concatenate([keypoint, keypoint_score[..., None]], -1)
        return results
        # return self.ops(results)

class PoseDecode:
    """Load and decode pose with given indices.

    Required keys are "keypoint", "frame_inds" (optional), "keypoint_score" (optional), added or modified keys are
    "keypoint", "keypoint_score" (if applicable).
    """

    @staticmethod
    def _load_kp(kp, frame_inds):
        return kp[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_kpscore(kpscore, frame_inds):
        return kpscore[:, frame_inds].astype(np.float32)

    def __call__(self, results):

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            results['keypoint_score'] = self._load_kpscore(results['keypoint_score'], frame_inds)

        if 'keypoint' in results:
            results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str

class FormatGCNInput:
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, mode='zero'):
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results['keypoint']
        if 'keypoint_score' in results:
            keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

        # M T V C
        if keypoint.shape[0] < self.num_person:
            pad_dim = self.num_person - keypoint.shape[0]
            pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and keypoint.shape[0] == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif keypoint.shape[0] > self.num_person:
            keypoint = keypoint[:self.num_person]

        M, T, V, C = keypoint.shape
        nc = results.get('num_clips', 1)
        assert T % nc == 0
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)
        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(num_person={self.num_person}, mode={self.mode})'
        return repr_str
    
class FormatGCNInputMV(FormatGCNInput):
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, mode='zero', num_view=3):
        super().__init__(num_person, mode)
        self.num_view = num_view

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results['keypoint']

        if 'keypoint_score' in results:
            keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

        n_in_person = keypoint.shape[0] // self.num_view

        # M T V C
        if n_in_person < self.num_person:
            pad_dim = self.num_person - n_in_person
            pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.insert(keypoint, list(range(0,pad_dim * self.num_view)), pad, axis=0)

        # TODO : check to collect only number of person per view
        elif keypoint.shape[0] > self.num_person*self.num_view:
            keypoint = keypoint[:self.num_person*self.num_view]

        M, T, V, C = keypoint.shape
        nc = results.get('num_clips', 1)
        assert T % nc == 0
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)
        results['keypoint'] = np.ascontiguousarray(keypoint)

        return results

class Coco2H36m:
    def __call__(self, results):
        '''
            Input: x (M x T x V x C)
            
            COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
            
            H36M:
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        '''

        x = results['keypoint']

        y = np.zeros(x.shape)
        y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
        y[:,:,1,:] = x[:,:,12,:]
        y[:,:,2,:] = x[:,:,14,:]
        y[:,:,3,:] = x[:,:,16,:]
        y[:,:,4,:] = x[:,:,11,:]
        y[:,:,5,:] = x[:,:,13,:]
        y[:,:,6,:] = x[:,:,15,:]
        y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
        y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
        y[:,:,9,:] = x[:,:,0,:]
        y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
        y[:,:,11,:] = x[:,:,5,:]
        y[:,:,12,:] = x[:,:,7,:]
        y[:,:,13,:] = x[:,:,9,:]
        y[:,:,14,:] = x[:,:,6,:]
        y[:,:,15,:] = x[:,:,8,:]
        y[:,:,16,:] = x[:,:,10,:]

        results['keypoint'] = y

        return results