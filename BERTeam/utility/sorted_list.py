import itertools
import math, copy
import numpy as np
import os
import shutil
import pickle


class Sorter:
    """
    used to keep track of the utilities in a dataset
    important operations: insert, delete, quantile search

    general use case is this: we have a data point with utility x to add to a dataset
    we first would like to bin x into quantiles (say we care about if x is larger than the median)
    query the quantiles q we care about, and get the correct bin to stick x into
        O(1) quantile searches
    then we put in utility of x into the sorter
        1 insertion
    in the case we are in a replay buffer and we delete an element with utility y, we must remove it from sorter as well
        1 deletion (we know for sure y exists in the dataset)

    thus, we would like to minimize the time complexity of
        O(quantile search) + O(insertion) + O(deletion)
    """

    def __init__(self, scalars=None, capacity=int(1e6)):
        if scalars is None:
            scalars = []
        self.info = {
            'capacity': capacity
        }
        self.extend(scalars=scalars)

    @property
    def capacity(self):
        return self.info['capacity']

    def insert(self, scalar):
        raise NotImplementedError

    def extend(self, scalars):
        for scalar in scalars:
            self.insert(scalar)

    def remove(self, scalar):
        raise NotImplementedError

    def clear(self):
        """
        clears all elements
        by default just repeatedly deletes until no remaining elements
        """
        while self.__len__():
            self.remove(self.__getitem__(0))

    def quantile(self, q=0.5):
        """
        returns the scalar at the qth quantile
        Args:
            q: 0 <= q <= 1
        Returns:
            scalar at that quantile
        """
        float_idx = (self.__len__() - 1)*q
        bottom, top = math.floor(float_idx), math.ceil(float_idx)
        p = float_idx - bottom
        return (1 - p)*self.__getitem__(bottom) + p*self.__getitem__(top)

    def __getitem__(self, item):
        """
        gets ith item in order from self
        """
        raise NotImplementedError

    def __str__(self):
        return str(list(self.__iter__()))

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        """
        returns items in order
        """
        return (self.__getitem__(i) for i in range(self.__len__()))

    def save_info(self, info_path):
        f = open(info_path, 'wb')
        pickle.dump(self.info, f)
        f.close()

    def save(self, save_dir):
        """
        save data to folder
        by default, converts to list, and pickles it
        """
        filename = os.path.join(save_dir, 'scalars.pkl')
        f = open(filename, 'wb')
        pickle.dump(list(self.__iter__()), f)
        f.close()
        self.save_info(os.path.join(save_dir, 'info.pkl'))

    def load_info(self, info_path):
        f = open(info_path, 'rb')
        info = pickle.load(f)
        self.info.update(info)

    def load(self, save_dir):
        """
        loads data from folder
        by default, grabs item as iterable, clears self, then loads all data
        """
        self.load_info(os.path.join(save_dir, 'info.pkl'))

        filename = os.path.join(save_dir, 'scalars.pkl')
        f = open(filename, 'rb')
        scalars = pickle.load(f)
        f.close()
        self.clear()
        self.extend(scalars=scalars)


class SortedList(Sorter):
    """
    bad implementation of sorted list
    O(1) quantile search
    O(nlogn) insertion
    O(n) deletion
    overall complexity of quantile search + inseration + deletion is
    O(nlogn)
    """

    def __init__(self, scalars=None, capacity=int(1e6)):
        self.scalars = []
        super().__init__(scalars=scalars, capacity=capacity)

    def insert(self, scalar):
        self.scalars.append(scalar)
        self.scalars.sort()
        if self.__len__() > self.capacity:
            # remove a random item
            self.remove(self.__getitem__(np.random.randint(self.__len__())))

    def remove(self, scalar):
        self.scalars.remove(scalar)

    def __len__(self):
        return len(self.scalars)

    def __getitem__(self, item):
        return self.scalars[item]


class AVLNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1
        self.left_cnt = 0
        self.right_cnt = 0
        self.cnt = 1


class SortedTree(Sorter):
    """
    AVL tree https://www.geeksforgeeks.org/avl-tree-in-python/
        added extra data for O(log n) indexing and increased efficiency for repeat values
        we assert that there is at most one node representing each value at the end of each operation
    Note: n is the number of UNIQUE values
        for data where there are lots of repeats, n will be very small
        since we usually expect this for games (i.e. 1 for win, 0 for tie, -1 for loss), this is very useful
    O(log n) quantile search
    O(log n) insertion
    O(log n) deletion
    overall complexity of quantile search + inseration + deletion is
    O(log n)
    space complexity is O(n)
    """

    def __init__(self, scalars=None, capacity=float('inf')):
        """
        Args:
            scalars:
            capacity: maximum number of values
                in cases where there are a smallish number of possible outcomes, this can usually be infinite without any problems
                the only issues would be from literally having numbers too large for python to store, not sure if this is relevant
        """
        self.root = None
        self.num_elements = 0
        self.num_nodes = 0
        super().__init__(scalars=scalars, capacity=capacity)

    def height(self, node):
        if not node:
            return 0
        return node.height

    def _balance(self, node):
        if not node:
            return 0
        return self.height(node.right) - self.height(node.left)

    def _insert(self, root, value):
        # added support of duplicates
        if not root:
            self.num_elements += 1  # we added a node with 1 element
            self.num_nodes += 1
            return AVLNode(value)
        elif value < root.value:
            root.left = self._insert(root.left, value)
            root.left_cnt += 1
        elif value > root.value:
            root.right = self._insert(root.right, value)
            root.right_cnt += 1
        else:
            # this is a duplicate, just increase the count at root, return
            root.cnt += 1
            self.num_elements += 1  # we increased number of elements
            return root

        root.height = 1 + max(self.height(root.left), self.height(root.right))
        balance = self._balance(root)

        # Right rotation
        # if left heavy and we inserted to left of root.left
        # note that in this case we cannot have inserted AT root.left
        # this would make left tree height exactly 1, so no way for balance to be <-1
        if balance < -1 and value < root.left.value:
            return self._rotate_R(root)

        # Left rotation
        if balance > 1 and value > root.right.value:
            return self._rotate_L(root)

        # Left-Right rotation
        if balance < -1 and value > root.left.value:
            root.left = self._rotate_L(root.left)
            return self._rotate_R(root)

        # Right-Left rotation
        if balance > 1 and value < root.right.value:
            root.right = self._rotate_R(root.right)
            return self._rotate_L(root)

        return root

    def _delete(self, root, value, cnt=1):
        # if value is duplicated in the tree, we just remove any occurence of value, this method is unchanged
        if not root:
            raise Exception('deleting value not in tree')

        if value < root.value:
            root.left = self._delete(root.left, value, cnt=cnt)
            root.left_cnt -= cnt
        elif value > root.value:
            root.right = self._delete(root.right, value, cnt=cnt)
            root.right_cnt -= cnt
        else:  # value==root.value
            root.cnt -= cnt  # this should never make this <0
            if root.cnt > 0:
                # if we have nodes to spare, just delete one and tree structure stays the same
                self.num_elements -= cnt  # we remove elements
                return root
            # otherwise, root.cnt==0 and we must remove root
            #  if either side is empty, we can simply make the left (resp. right) the new root
            if not root.left:
                temp = root.right
                root = None
                self.num_elements -= cnt  # we remove only the root node
                self.num_nodes -= 1
                return temp
            elif not root.right:
                temp = root.left
                root = None
                self.num_elements -= cnt  # we remove only the root node
                self.num_nodes -= 1
                return temp
            # otherwise, we grab the minimum value node on right and make it the root
            temp = self._min_value_node(root.right)
            root.value = temp.value
            root.cnt = temp.cnt
            # we have removed cnt from root, and added temp.cnt
            # technically temp.cnt-(cnt+root.cnt), but root.cnt=0
            self.num_elements += temp.cnt - cnt
            # we must delete temp.cnt instances of temp.value from the right tree
            root.right = self._delete(root.right, temp.value, cnt=temp.cnt)
            root.right_cnt -= temp.cnt

        if not root:
            return root

        root.height = 1 + max(self.height(root.left), self.height(root.right))
        balance = self._balance(root)

        # Left rotation
        if balance < -1 and self._balance(root.left) <= 0:
            return self._rotate_R(root)

        # Right rotation
        if balance > 1 and self._balance(root.right) >= 0:
            return self._rotate_L(root)

        # Left-Right rotation
        if balance < -1 and self._balance(root.left) > 0:
            root.left = self._rotate_L(root.left)
            return self._rotate_R(root)

        # Right-Left rotation
        if balance > 1 and self._balance(root.right) < 0:
            root.right = self._rotate_R(root.right)
            return self._rotate_L(root)

        return root

    def _rotate_L(self, x):
        z = x.right
        T2 = z.left

        z.left = x
        x.right = T2

        x.height = 1 + max(self.height(x.left), self.height(x.right))
        z.height = 1 + max(self.height(z.left), self.height(z.right))

        # https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/AVL-simple-left_K.svg/194px-AVL-simple-left_K.svg.png
        # z's right and x's left do not change
        # z+z.right used to be right of x, and now they are not
        x.right_cnt -= z.cnt + z.right_cnt
        # x+x.left are now left of z
        z.left_cnt += x.cnt + x.left_cnt

        return z

    def _rotate_R(self, x):
        z = x.left
        T3 = z.right

        z.right = x
        x.left = T3

        x.height = 1 + max(self.height(x.left), self.height(x.right))
        z.height = 1 + max(self.height(z.left), self.height(z.right))

        # z+z.left used to be left of x and now are not
        x.left_cnt -= z.cnt + z.left_cnt
        # x+x.right are now right of z
        z.right_cnt += x.cnt + x.right_cnt

        return z

    def _min_value_node(self, root):
        current = root
        while current.left:
            current = current.left
        return current

    def search(self, root, value):
        if not root or root.value == value:
            return root
        if root.value < value:
            return self.search(root.right, value)
        return self.search(root.left, value)

    def insert(self, scalar):
        self.root = self._insert(self.root, scalar)
        if self.__len__() > self.capacity:
            # remove a random item
            self.remove(self.__getitem__(np.random.randint(self.__len__())))

    def remove(self, scalar):
        self.root = self._delete(self.root, scalar)

    def __getitem__(self, item):
        return self.search_idx(self.root, item)

    def __len__(self):
        return self.num_elements

    def search_idx(self, root, idx):
        if idx < root.left_cnt:
            return self.search_idx(root.left, idx)
        if idx < root.left_cnt + root.cnt:
            # root.left_cnt <= idx < root.left_cnt + root.cnt
            #  then we have landed on one of the root.cnt copies of root.value
            return root.value
        if idx >= root.left_cnt + root.cnt:
            return self.search_idx(root.right,
                                   idx - root.left_cnt - root.cnt,
                                   )  # -root.cnt because of the values in root

    def search_value(self, value):
        return self.search(self.root, value)

    def clear(self):
        self.root = None
        self.num_elements = 0

    def get_iterable(self, node):
        """
        gets in order traversal in O(n)
        Returns:
            iterable
        """
        if not node:
            return iter(())
        return itertools.chain(
            self.get_iterable(node.left),
            (node.value for _ in range(node.cnt)),
            self.get_iterable(node.right),
        )

    def __iter__(self):
        """
        returns items in order
        """
        return self.get_iterable(self.root)

    def save(self, save_dir):
        """
        surprised this works, looks like pickle grabs the entire tree for quick storage with O(n) space/time
        """
        self.save_info(os.path.join(save_dir, 'info.pkl'))
        filename = os.path.join(save_dir, 'root.pkl')
        f = open(filename, 'wb')
        pickle.dump((self.root, self.num_elements, self.num_nodes), f)
        f.close()

    def load(self, save_dir):
        self.load_info(os.path.join(save_dir, 'info.pkl'))
        filename = os.path.join(save_dir, 'root.pkl')
        self.clear()
        f = open(filename, 'rb')
        self.root, self.num_elements, self.num_nodes = pickle.load(f)
        f.close()


if __name__ == '__main__':
    import torch, time

    sl2 = SortedTree(scalars=[int(t.item()) for t in torch.rand(420)*420], capacity=100)
    print(len(sl2))

    sl1 = SortedList()
    sl2 = SortedTree()
    assert list(sl1) == list(sl2)

    n = 10000
    scalars = [int(t.item()) for t in torch.rand(n)*n]
    sl1 = SortedList(scalars=scalars)

    sl1.insert(69)
    sl1.remove(scalars[0])
    sl1.remove(scalars[3])

    sl2 = SortedTree(scalars=scalars)
    sl2.insert(69)
    sl2.remove(scalars[0])
    sl2.remove(scalars[3])
    assert list(sl1) == list(sl2)
    # save test
    save_dir = 'save_sorted_list_test'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    sl2.save(save_dir)
    sl2.clear()
    print(sl2)
    sl3 = SortedTree()
    sl3.load(save_dir)
    shutil.rmtree(save_dir)
    print(len(sl1), len(sl3))

    assert list(sl1) == list(sl3)

    # time complexity test
    # time of doing n insertions and removals, starting from empty list
    # it is quite fast for n>=10000, which is a good scale
    n = 50000
    m = 3  # number of unique values, decreaseing this should improve the tree implemnetation, as we love repeats
    scalars = [int(t.item()) for t in torch.rand(n)*m]
    perm = torch.randperm(n)
    for sl in SortedTree(), SortedList():
        tim = time.time()
        for t in scalars:
            sl.insert(t)
        for i in perm:
            sl.remove(scalars[i])
        print('time', time.time() - tim)


    # correctness test

    def check(Sorted1, Sorted2, n=100):
        """
        does n random tests that these are the same
        """
        assert len(Sorted1) == len(Sorted2)
        assert list(Sorted1) == list(Sorted2)
        if len(Sorted1) > 0:
            for q in torch.rand(n):
                assert Sorted1.quantile(q) == Sorted2.quantile(q)
            for q in [0, 1, .5]:
                assert Sorted1.quantile(q) == Sorted2.quantile(q)


    for i in range(420):
        # check only repeats of inserting 0 over and over
        scalars = [0 for t in range(i)]
        sl1 = SortedList(scalars=scalars)
        sl2 = SortedTree(scalars=scalars)
        check(sl1, sl2)
        # insertions:
        for _ in range(i):
            item = 0
            sl1.insert(item)
            sl2.insert(item)
        check(sl1, sl2)
        # removals:
        for _ in range(i):
            item = sl1[int(len(sl1)*torch.rand(1).item())]
            sl1.remove(item)
            sl2.remove(item)
        check(sl1, sl2)

    for i in range(1000):
        # allow repeats by rounding to int
        scalars = [int(t.item()) for t in torch.rand(i)*i]
        sl1 = SortedList(scalars=scalars)
        sl2 = SortedTree(scalars=scalars)
        check(sl1, sl2)
        # insertions:
        for t in torch.rand(i)*i:
            item = int(t.item())
            sl1.insert(item)
            sl2.insert(item)
        check(sl1, sl2)
        # removals:
        for _ in range(i):
            item = sl1[int(len(sl1)*torch.rand(1).item())]
            sl1.remove(item)
            sl2.remove(item)
        check(sl1, sl2)
    quit()
    # complexity test, should be O(n log n), so this line should be almost linear
    x = []
    y = []
    y_cum = []
    for n in [1e3, 2e3, 4e3, 7e3,
              1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4,
              1e5
              ]:
        n = int(n)
        perm = torch.randperm(n)
        sl = SortedTree()
        scalars = [int(t.item()) for t in torch.rand(n)*n]
        tim = time.time()
        for t in scalars:
            sl.insert(t)
        for i in perm:
            sl.remove(scalars[i])
        dur = time.time() - tim
        y.append(dur)
        y_cum.append(dur/n)
        x.append(n)
    import matplotlib.pyplot as plt

    print(x)
    print(y_cum)
    plt.plot(x, torch.tensor(y)/max(y), label='scaled time')
    plt.plot(x, torch.tensor(y_cum)/max(y_cum), label='scaled time/n')
    plt.legend()
    plt.show()
