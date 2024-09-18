import numpy as np

class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        seed (int): The random seed used during test time. Default: 255.
        start_index (int): Index where to start (originally set by the dataset in PYSKL)
    """

    def __init__(self,
                 clip_len,
                 num_clips=1,
                 p_interval=1,
                 seed=255,
                 start_index=0,
                 reproducible=False):

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.start_index = start_index
        self.p_interval = p_interval
        self.reproducible = reproducible
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)
        # if len(deprecated_kwargs):
        #     warning_r0('[UniformSampleFrames] The following args has been deprecated: ')
        #     for k, v in deprecated_kwargs.items():
        #         warning_r0(f'Arg name: {k}; Arg value: {v}')

    def _get_train_clips(self, num_frames, clip_len):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        for clip_idx in range(self.num_clips):
            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            inds = inds + off
            num_frames = old_num_frames

            allinds.append(inds)

        return np.concatenate(allinds)

    def _get_test_clips(self, num_frames, clip_len):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        np.random.seed(self.seed)

        all_inds = []

        for i in range(self.num_clips):

            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds + off)
            num_frames = old_num_frames

        return np.concatenate(all_inds)

    def __call__(self, results):
        num_frames = results['total_frames']

        if self.reproducible:
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = self.start_index
        inds = inds + start_index

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1], str(num_frames) + " vs " + str(kp.shape)
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(int)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results['frame_inds'] = inds.astype(int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'seed={self.seed})')
        return repr_str


class UniformSample(UniformSampleFrames):
    pass

class Resample:
    def __init__(self, target_len, replay=False, randomness=True):
        self.target_len = target_len
        self.replay = replay
        self.randomness = randomness

    def compute_sample_ids(self, ori_len):
        if self.replay:
            if ori_len > self.target_len:
                st = np.random.randint(ori_len-self.target_len)
                return range(st, st+self.target_len)  # Random clipping from sequence
            else:
                return np.array(range(self.target_len)) % ori_len  # Replay padding
        else:
            if self.randomness:
                even = np.linspace(0, ori_len, num=self.target_len, endpoint=False)
                if ori_len < self.target_len:
                    low = np.floor(even)
                    high = np.ceil(even)
                    sel = np.random.randint(2, size=even.shape)
                    result = np.sort(sel*low+(1-sel)*high)
                else:
                    interval = even[1] - even[0]
                    result = np.random.random(even.shape)*interval + even
                result = np.clip(result, a_min=0, a_max=ori_len-1).astype(np.uint32)
            else:
                result = np.linspace(0, ori_len, num=self.target_len, endpoint=False, dtype=int)
            return result

    def __call__(self, results):
        ori_len = results['total_frames']
        
        inds = self.compute_sample_ids(ori_len)

        results['frame_inds'] = inds.astype(int)
        results['clip_len'] = self.target_len
        results['frame_interval'] = None
        results['num_clips'] = 1
        return results