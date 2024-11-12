import torch, os, sys, shutil
import dill as pickle
from BERTeam.outcome import PlayerInfo
from BERTeam.utility.sorted_list import SortedTree


class LangReplayBuffer:
    storage_dir = None
    track_age = False

    def reset_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        self.reset_storage()

    def reset_storage(self):
        """
        resets internal buffer
        """
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.push(item)

    def add_outcome(self,
                    teams,
                    outcome,
                    filter=None,
                    ignore_observation=False,
                    chain_observation=True,
                    ):
        """
        adds result of Outcome.get_outcome to buffer
        Args:
            teams: list of teams
            outcome:
                list corresponding to teams
                    [
                        outcome score,
                        list corresponding to players of PlayerInfo(
                            obs_preembed=player observation (None or size (S,*) seq of observations);
                            obs_mask=observation mask (None or size (S,) boolean array of which items to mask;
                            )
                        list can be empty, this will correspond to an empty observation
                    ]
            filter: whether to add each outcome, bsed on score
        """

        for team, (score, playerinfos) in zip(teams, outcome):
            if filter is not None:
                if not filter(score):
                    continue
            playerinfo = PlayerInfo()

            if not ignore_observation and not chain_observation:
                # in this case, add one example for each player
                for pi in playerinfos:
                    obs_preembed, obs_mask = pi.get_data()
                    item = (score, obs_preembed, team, obs_mask)
                    self.push(item)
                continue
            # otherwise, we combine the observations (if not ignored)
            if chain_observation and not ignore_observation:
                for pi in playerinfos:
                    playerinfo = playerinfo.union_obs(other_player_info=pi)
            obs_preembed, obs_mask = playerinfo.get_data()
            item = (score, obs_preembed, team, obs_mask)
            self.push(item)

    def push(self, item, **kwargs):
        """
        pushes an item into replay buffer
        Args:
            item: item
        Returns: item that is displaced, or None if no such item
        """
        raise NotImplementedError

    def sample_one(self, **kwargs):
        raise NotImplementedError

    def clear(self):
        pass

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass

    def sample(self, batch, **kwargs):
        """
        sample an item
        Args:
            batch: number of items to sample
                None for 'special' sample, usually just spits out all items in buffer
            **kwargs:
        Returns:
        """
        if batch is None:
            for _ in range(self.__len__()):
                yield self.sample_one()
        else:
            for _ in range(batch):
                yield self.sample_one()

    # def __getitem__(self, item):
    #    raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ReplayBufferDiskStorage(LangReplayBuffer):
    def __init__(self,
                 storage_dir=None,
                 capacity=1e6,
                 device=None,
                 track_age=False,
                 count_capacity=1e12,
                 ):
        """
        Args:
            track_age: if True, when sampling, returns (item, age)
            track_order: whether to track the order of all elements to do some quantile shenanigans
                if true, requires all inputs to look like (scalar, ...). i.e. item[0] is a real number
        """
        self.idx = 0
        self.size = 0
        self.capacity = capacity
        self.device = device

        self.track_age = track_age
        self.time_cnt = 0
        self.parity = True
        self.reset_time = self.capacity
        self.count_capacity = count_capacity

        # count capacity should be MUCH bigger, but this should work as long as we dont tick too large
        assert self.count_capacity > self.capacity

        if storage_dir is not None:
            self.reset_storage_dir(storage_dir=storage_dir)

    def clear(self):
        super().clear()
        if self.storage_dir is not None:
            if os.path.exists(self.storage_dir):
                shutil.rmtree(self.storage_dir)

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        shutil.copytree(src=self.storage_dir, dst=save_dir)

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        self.clear()
        shutil.copytree(src=save_dir, dst=self.storage_dir)
        self.load_place(force=False)

    def reset_storage(self):
        self.clear()
        os.makedirs(self.storage_dir)
        self.size = 0
        self.idx = 0
        self.save_place()

    def save_place(self):
        """
        saves idx and size to files as well
        """
        pickle.dump(
            {
                'size': self.size,
                'idx': self.idx,
                'track_age': self.track_age,
                'time_cnt': self.time_cnt,
                'parity': self.parity,
            },
            open(self._get_file('info'), 'wb')
        )

    def load_place(self, force=False):
        info_file = self._get_file(name='info')
        if os.path.exists(info_file):
            dic = pickle.load(open(info_file, 'rb'))
            self.size = dic['size']
            self.idx = dic['idx']
            self.track_age = dic['track_age']
            self.time_cnt = dic['time_cnt']
            self.parity = dic['parity']
        else:
            if force:
                print('failed to load file:', info_file)
                print('resetting storage')
                self.reset_storage()
            else:
                raise Exception('failed to load file: ' + info_file)

    def _get_file(self, name):
        return os.path.join(self.storage_dir, str(name) + '.pkl')

    def push(self, item, age_tick=1):
        if self.size == self.capacity:
            disp = self.__getitem__(self.idx)
        else:
            disp = None
        item = self.change_item_to_new_youngest(item=item, age_tick=age_tick)
        pickle.dump(item, open(self._get_file(self.idx), 'wb'))

        self.size = max(self.idx + 1, self.size)
        self.idx = int((self.idx + 1)%self.capacity)

        self.save_place()
        return disp

    def _grab_item_by_idx(self, idx, change_device=True):
        item = pickle.load(open(self._get_file(name=idx), 'rb'))
        return self._convert_device(item=item, change_device=change_device)

    def _convert_device(self, item, change_device):
        if change_device:
            if type(item) == tuple:
                item = tuple(self._convert_device(t, change_device=change_device)
                             for t in item)
            elif torch.is_tensor(item):
                item = item.to(self.device)
        return item

    def sample_one(self, **kwargs):
        return self[torch.randint(0, self.size, (1,))]

    def sample(self, batch, **kwargs):
        """
        Args:
            batch: batch size, if None, returns all elements in random order
        Returns:
            iterable of items
        """
        if batch is None:
            for i in torch.randperm(self.__len__()):
                yield self[i]
        else:
            for _ in range(batch):
                yield self.sample_one()

    def __getitem__(self, item):
        if item >= self.size:
            raise IndexError
        return self.get_output_item(self._grab_item_by_idx(idx=int((self.idx + item)%self.size)))

    def __len__(self):
        return self.size

    def change_item_to_new_youngest(self, item, age_tick=1):
        """
        updates oldest and parity
        Returns:
            item, birthday storage tuple
            or item if not self.track_age
        """
        if self.track_age:
            self.time_cnt += age_tick
            if self.parity:
                item = item, (-1, self.time_cnt)
            else:
                item = item, (self.time_cnt, -1)
            if self.time_cnt >= self.count_capacity:
                # switch parity, reset age
                # do this to prevent overflow for particularly long experiments
                self.parity = not self.parity
                # time when we reset, usually self.count_capacity
                self.reset_time = self.time_cnt

                self.time_cnt = 0
            return item
        else:
            return item

    def get_output_item(self, item):
        """
        takes an item in storage, returns item to output
        if self.track_age is false, does nothing
        otherwise finds true age of item
        """
        if self.track_age:
            item, birth_tup = item
            if birth_tup[self.parity] == -1:
                # in this case, the age until the parity swap will be self.capacity-birth_tup[not self.parity]
                # we then add to this self.oldest
                age = self.time_cnt + (self.reset_time - birth_tup[not self.parity])
            else:
                age = self.time_cnt - birth_tup[self.parity]
            if age < 0:
                # this only happens if we made count capacity too low, and at least two swaps happened in recent memory
                # just add a bunch to age and call it a day
                age = 2*self.count_capacity
            return item, age
        else:
            return item


class GeneralBinnedReplayBuffer(LangReplayBuffer):
    """
    creates multiple disk replay buffers, each representing a 'bin' of data
    """

    def __init__(self,
                 storage_dir=None,
                 num_bins=1,
                 capacity=1e6,
                 device=None,
                 track_age=False,
                 independent_ages=True,
                 count_capacity=1e12,
                 ignore_zeros=True,
                 BinBufferClass=ReplayBufferDiskStorage,
                 ):
        """
        Args:
            num_bins: number of bins
            independent_ages: each bin has an independent age
            ignore_zeros: just pretend zeros do not happen, no reason to have zero weighted elements in dataset
            BinBufferClass: class to use for each bin
        """
        super().__init__()
        self.bins = [BinBufferClass(storage_dir=None,
                                    capacity=capacity,
                                    device=device,
                                    track_age=track_age,
                                    count_capacity=count_capacity,
                                    )
                     for _ in range(num_bins)
                     ]
        self.info = {
            'avgs': torch.zeros(num_bins),
            'size': 0,
            'buffer ages': [0 for _ in self.bins],  # keep track of how long ago each buffer was pushed to
        }
        if storage_dir is not None:
            self.reset_storage_dir(storage_dir=storage_dir)
        self.track_age = track_age
        self.independent_ages = independent_ages
        self.ignore_zeros = ignore_zeros
        self.num_bins = num_bins

    def reset_storage_dir(self, storage_dir):
        super().reset_storage_dir(storage_dir=storage_dir)
        for i, biin in enumerate(self.bins):
            biin.reset_storage_dir(storage_dir=os.path.join(storage_dir, 'bin_' + str(i)))

    def reset_storage(self):
        self.clear()
        os.makedirs(self.storage_dir)
        self.idx = 0

    def set_size(self, size):
        self.info['size'] = size

    @property
    def size(self):
        return self.info['size']

    @property
    def bounds(self):
        return self.info['bounds']

    @property
    def avgs(self):
        return self.info['avgs']

    def push(self, item, bindex=0):
        """
        pushes an item into replay buffer
        Args:
            item: item
        Returns: item that is displaced, or None if no such item
        """
        scalar = item[0]
        if self.ignore_zeros and scalar == 0:
            return
        if bindex is not None:
            if self.track_age:
                if self.independent_ages:
                    age_tick = 1
                else:
                    # we must increase age by how much time has passed since we increased this buffer
                    age_tick = 1 + self.info['buffer ages'][bindex]
                disp = self.bins[bindex].push(item, age_tick=age_tick)
            else:
                disp = self.bins[bindex].push(item)
            if disp is None:
                rem = 0
                self.set_size(self.size + 1)
            else:
                if self.track_age:
                    # in this case, disp is ((scalar,...),age)
                    rem = disp[0][0]
                else:
                    # (scalar,...)
                    rem = disp[0]
            self.avgs[bindex] = (self.avgs[bindex]*(len(self.bins[bindex]) - 1) + scalar - rem)/len(self.bins[bindex])
            self.tick_buffer_ages(new_push_idx=bindex)
            return disp

    def clear(self):
        super().clear()
        for guy in self.bins:
            guy.clear()
        if self.storage_dir is not None:
            if os.path.exists(self.storage_dir):
                shutil.rmtree(self.storage_dir)

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        f = open(os.path.join(save_dir, 'info.pkl'), 'wb')
        pickle.dump(self.info, f)
        f.close()

        for i, guy in enumerate(self.bins):
            guy.save(save_dir=os.path.join(save_dir, 'bin_' + str(i)))

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        f = open(os.path.join(save_dir, 'info.pkl'), 'rb')
        self.info.update(pickle.load(f))
        f.close()

        for i, guy in enumerate(self.bins):
            guy.load(save_dir=os.path.join(save_dir, 'bin_' + str(i)))

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        """
        weird but they call the index 'item'
        """
        if item >= self.size:
            raise IndexError
        for bindex, biin in enumerate(self.bins):
            if item < biin.size:
                return self.get_output_item(item=biin[item], bindex=bindex)
            item -= biin.size

    @property
    def bin_lens(self):
        return torch.tensor([len(bin) for bin in self.bins],
                            dtype=torch.float,
                            )

    def tick_buffer_ages(self, new_push_idx):
        if self.track_age:
            for c in range(len(self.bins)):
                # all others are added to
                self.info['buffer ages'][c] += 1
            # this one is updated to zero
            self.info['buffer ages'][new_push_idx] = 0

    def get_output_item(self, item, bindex):
        """
        gets true age of item by adding the buffer age to the age of item in buffer
         if independent ages, this just returns the age
        """
        if self.track_age:
            item, age = item
            if not self.independent_ages:
                age = age + self.info['buffer ages'][bindex]
            return (item, age)
        else:
            return item


class BinnedReplayBufferDiskStorage(GeneralBinnedReplayBuffer):
    """
    TODO: the name is old, we should change it along with coevolver
    creates multiple disk replay buffers, each representing a 'bin' of data
    items must begin with a scalar that represents how much it should show up in the data
    """

    def __init__(self,
                 storage_dir=None,
                 bounds=None,
                 capacity=1e6,
                 device=None,
                 track_age=False,
                 independent_ages=True,
                 count_capacity=1e12,
                 ignore_zeros=True,
                 BinBufferClass=ReplayBufferDiskStorage,
                 ):
        """
        Args:
            bounds: sorted list of numbers to divide the input into bins based on the scalar associated with it
                each bin keeps track of the average value
                upon sampling, an extra parameter can be added to determine with what
                    frequency elements from each bin appear

                values <=bounds[0] or >bounds[1] are ignored
                by default, uses [1/2, 1], corresponding to one bin of range (1/2,1]
                    This is intended for a 2 player game where we want to capture outcomes that are better than ties
                        Usually, in an n-player game, the bounds should range [1/n,...,1] to capture outcomes that are
                            better than ties
            ignore_zeros: just pretend zeros do not happen
        """
        if bounds is None:
            bounds = [1/2, 1]
        super().__init__(storage_dir=storage_dir,
                         num_bins=len(bounds) - 1,
                         capacity=capacity,
                         device=device,
                         track_age=track_age,
                         independent_ages=independent_ages,
                         count_capacity=count_capacity,
                         ignore_zeros=ignore_zeros,
                         BinBufferClass=BinBufferClass,
                         )
        # we should also track the bounds
        self.info['bounds'] = tuple(bounds)
        self.weights = None

    def bin_search(self, value, possible=None):
        """
        binary bin search
        Args:
            value: value to put into bin
            possible: possible bindices (if None, searches all)
        Returns:
            i such that self.bounds[i] < value<= self.bounds[i+1]
            or None if not possible
        """
        if possible is None:
            possible = (0, len(self.bounds) - 1)
        i, j = possible
        if i + 1 == j:
            if self.bounds[i] < value and value <= self.bounds[i + 1]:
                return i
            else:
                return None
        mid = (i + j)//2
        # i+1 <= mid <= j-1
        if self.bounds[mid] < value:
            return self.bin_search(value, (mid, j))
        if value < self.bounds[mid]:
            return self.bin_search(value, (i, mid))
        if value == self.bounds[mid]:
            # 0 <= i <= mid-1
            return mid - 1

    def push(self, item, **kwargs):
        scalar = item[0]
        bindex = self.bin_search(scalar)
        return super().push(item=item, bindex=bindex)

    def set_weights(self, values_to_weights):
        """
        sets weights of each  bin according to the average value of its elements
        Args:
        values_to_weights: a function (tensor -> tensor) ([0,1] -> R+), weights to give a bin with a particular value
            if None, uses weights of len(bin) for each bin (this corresponds to uniformly sampling an element)
        """
        self.weights = values_to_weights(self.avgs)

    def sample_one(self, **kwargs):
        """
        samples one according to weights set
        if weights are not set, samples uniformly from seen examples
        first samples a bin according to weights, then samples an element from bin
        """
        weights = self.weights
        if weights is None:
            weights = self.bin_lens
        # weight empty bins at 0
        weights = weights*(self.bin_lens > 0)

        bindex = torch.multinomial(weights, 1)
        return self.get_output_item(item=self.bins[bindex].sample_one(), bindex=bindex)

    def sample(self, batch, **kwargs):
        """
        samples a bin according to the average value of its elements, then samples elements from the bin
        Args:
            batch: number of elements to sample
            **kwargs:
        Returns:
            Iterable of samples
        """
        for item in super().sample(batch=batch, **kwargs):
            yield item


class QuantileReplayBuffer(GeneralBinnedReplayBuffer):
    """
    tracks bins that have certian quantile ranges, sampling must specify which quantile to sample
    items must begin with a scalar that represents how much it should show up in the data
    """

    def __init__(self,
                 storage_dir=None,
                 quantile_ranges=None,
                 capacity=1e6,
                 device=None,
                 track_age=False,
                 independent_ages=True,
                 count_capacity=1e12,
                 SorterClass=SortedTree,
                 sorter_capacity=float('inf'),
                 ignore_first=10,
                 ignore_zeros=True,
                 BinBufferClass=ReplayBufferDiskStorage,
                 ):
        """
        Args:
            quantile_ranges: list of (low, high), with low<high will use these to define bins to stash data in
                by default assumes open interval (low, high) unless low=0 or high=1, then [low,high) or (low,high]
                however, this can be changed by setting low or high to (float, 'c'/'o')for closed/open endpoints
            independent_ages: each bin has an independent age
            SorterClass: SorterClass() should return a Sorter object to insert/remove/search quantiles with
            ignore_first: ignores first n datapoints no matter what to get a better idea of the distribution
                 this prevents weird behavior like the first point always being accepted
            ignore_zeros: just pretend zeros do not happen
        """
        if quantile_ranges is None:
            quantile_ranges = [((.5, 'o'), (1, 'c'))]
        self.sorter = SorterClass(capacity=sorter_capacity)
        super().__init__(storage_dir=storage_dir,
                         num_bins=len(quantile_ranges),
                         capacity=capacity,
                         device=device,
                         track_age=track_age,
                         independent_ages=independent_ages,
                         count_capacity=count_capacity,
                         ignore_zeros=ignore_zeros,
                         BinBufferClass=BinBufferClass,
                         )
        self.quantile_ranges = []
        for low, high in quantile_ranges:
            if not type(low) == tuple:
                low = (low, 'c' if low == 0 else 'o')
            if not type(high) == tuple:
                high = (high, 'c' if high == 1 else 'o')
            self.quantile_ranges.append((low, high))
        # we should also track the bounds
        self.info['quantile_ranges'] = tuple(self.quantile_ranges)
        self.info['ignore_counter'] = ignore_first

    def _is_within(self, scalar, quant_rng):
        """
        checks if scalar is on the specified quantile range
        """
        (l_quant, l_inc), (h_quant, h_inc) = quant_rng
        low, high = self.sorter.quantile(q=l_quant), self.sorter.quantile(q=h_quant)
        if low < scalar and scalar < high:
            return True
        if low == scalar:
            # edge case: low==high==scalar, l_inc==h_inc=='o'
            # in this case, we should accept, as our dataset is essentially filled with a bunch of the same value
            # it is impossible to sort these, so just accept all of them
            return (l_inc == 'c') or (high == scalar)
        if high == scalar:
            return h_inc == 'c'
        return False

    def push(self, item, **kwargs):
        """
        pushes an item into replay buffer
        Args:
            item: item
        Returns: list(item that is displaced, index of bin it was removed from)
        """
        scalar = item[0]
        self.sorter.insert(scalar=scalar)
        if self.info['ignore_counter'] > 0:
            self.info['ignore_counter'] -= 1
        else:
            disp_items = []
            for bindex, quant_rng in enumerate(self.quantile_ranges):
                if self._is_within(scalar=scalar, quant_rng=quant_rng):
                    disp = super().push(item=item, bindex=bindex)
                    disp_items.append((disp, bindex))
            return disp_items

    def sample_one(self, bindex=0, **kwargs):
        """
        samples one according to weights set
        if weights are not set, samples uniformly from seen examples
        first samples a bin according to weights, then samples an element from bin
        Args:
            bindex: which quantile bin to sample from
        """
        return self.get_output_item(item=self.bins[bindex].sample_one(), bindex=bindex)

    def get_quantile_range(self, bindex=0):
        """
        returns the quantile rante at specified index
        Returns:
            (low, low_inclusion), (high, high_inclusion)
            *_inclusion is 'o'/'c' for open/closed endpoint
        """
        return self.quantile_ranges[bindex]

    def sample(self, batch, bindex=0, **kwargs):
        """
        samples a bin according to the average value of its elements, then samples elements from the bin
        Args:
            batch: number of elements to sample, if None returns all elemnts in random order
            bindex: index of quantile to use
            **kwargs:
        Returns:
            Iterable of samples
        """
        if batch is None:
            for item in self.bins[bindex].sample(batch=batch, kwargs=kwargs):
                yield self.get_output_item(item=item, bindex=bindex)
        else:
            for _ in range(batch):
                yield self.sample_one(bindex=bindex)

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        sort_dir = os.path.join(save_dir, 'sorter')
        self.sorter.save(save_dir=sort_dir)

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        sort_dir = os.path.join(save_dir, 'sorter')
        self.sorter.load(save_dir=sort_dir)

    def clear(self):
        super().clear()
        self.sorter.clear()


if __name__ == '__main__':

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
    storage_dir = os.path.join(DIR, 'temp')
    # normal test
    test = ReplayBufferDiskStorage(capacity=3,
                                   storage_dir=storage_dir,
                                   track_age=False,
                                   count_capacity=4
                                   )
    sring = '0123456789abcdefghi'
    test.extend(sring)
    stuff = list(test.sample(3))
    print(stuff)
    possible = []
    for age, item in enumerate(sring[::-1]):
        possible.append(item)
        if age == test.capacity:
            break
    for t in stuff:
        assert t in possible
    test.clear()

    # age test
    test = ReplayBufferDiskStorage(capacity=3,
                                   storage_dir=storage_dir,
                                   track_age=True,
                                   count_capacity=4
                                   )
    sring = '0123456789abcdefghi'
    test.extend(sring)
    stuff = list(test.sample(3))
    print(stuff)
    possible = []
    for age, item in enumerate(sring[::-1]):
        print((item, age), end=', ')
        possible.append((item, age))
        if age == test.capacity:
            print()
            break
    for t in stuff:
        assert t in possible
    test.clear()

    # normal binned test
    test = BinnedReplayBufferDiskStorage(capacity=30,
                                         storage_dir=storage_dir,
                                         track_age=False,
                                         count_capacity=1000,
                                         bounds=[-1, .25, .5, .75, 1],
                                         ignore_zeros=False,
                                         )
    sring = ['0123456789abcdefghij'[torch.randint(0, 10, ())] for _ in range(10000)]
    all_items = []
    for item in sring:
        item = (torch.rand(1).item(), item)
        test.push(item)
        all_items.append(item)
    stuff = list(test.sample(3))
    possible = []
    for age, item in enumerate(all_items[::-1]):
        possible.append(item)
    for t in stuff:
        assert t in possible
    test.clear()

    # binned age test

    test = BinnedReplayBufferDiskStorage(capacity=30,
                                         storage_dir=storage_dir,
                                         track_age=True,
                                         count_capacity=1000,
                                         bounds=[-1, .25, .5, .75, 1],
                                         independent_ages=False,
                                         ignore_zeros=False,
                                         )
    sring = ['0123456789abcdefghij'[torch.randint(0, 10, ())] for _ in range(10000)]
    all_items = []
    for item in sring:
        item = (torch.rand(1).item(), item)
        test.push(item)
        all_items.append(item)
    stuff = list(test.sample(3))
    possible = []
    for age, item in enumerate(all_items[::-1]):
        possible.append((item, age))
    for t in stuff:
        assert t in possible
    test.clear()

    # normal quantile test
    quantile_ranges = [
        (0, 1),
        (0, .5),
        (.5, 1),
        (.75, 1),
    ]
    test = QuantileReplayBuffer(capacity=420,
                                storage_dir=storage_dir,
                                track_age=False,
                                quantile_ranges=quantile_ranges,
                                ignore_zeros=False,
                                )

    for i in range(10000):
        item = (torch.rand(1).item(), i)
        test.push(item)
        all_items.append(item)
    for i, q in enumerate(quantile_ranges):
        batch = 100
        stuff = test.sample(batch=batch, bindex=i)
        mean = sum([t[0] for t in stuff])/batch
        print('quantile:', q, '\tmean:', mean, '\texpected:', sum(q)/2)
        # the expectation is the midpoint, since we draw uniformly from [0,1]

    test.clear()
