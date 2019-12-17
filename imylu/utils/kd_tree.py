# -*- coding: utf-8 -*-
from ..utils.utils import get_euclidean_distance


class Node(object):
    def __init__(self):
        """Node class to build tree leaves.
        """

        self.father = None
        self.left = None
        self.right = None
        self.feature = None
        self.split = None

        self.class_num=None
        self.ratio=None
        self.num=None

        self.area=1331
        self.noempty=None
        self.x_min=None
        self.x_max=None
        self.y_min=None
        self.y_max=None
        self.z_min=None
        self.z_max=None
        self.visted=False


    def __str__(self):
        return "feature: %s, split: %s" % (str(self.feature), str(self.split))


    def brother(self):
        """Find the node's brother.

        Returns:
            node -- Brother node.
        """
        if not self.father:
            ret = None
        else:
            if self.father.left is self:
                ret = self.father.right
            else:
                ret = self.father.left
        return ret


class KDTree(object):
    def __init__(self,num,data_num,class_num):
        """KD Tree class to improve search efficiency in KNN.

        Attributes:
            root: the root node of KDTree.
        """
        self.root = Node()
        self.num=num
        self.root.num=data_num
        self.root.class_num=class_num


    def __str__(self):
        """Show the relationship of each node in the KD Tree.

        Returns:
            str -- KDTree Nodes information.
        """

        ret = []
        i = 0
        que = [(self.root, -1)]
        while que:
            nd, idx_father = que.pop(0)
            ret.append("%d -> %d: %s" % (idx_father, i, str(nd)))
            if nd.left:
                que.append((nd.left, i))
            if nd.right:
                que.append((nd.right, i))
            i += 1
        return "\n".join(ret)

    def _get_median_idx(self, X, idxs, feature):
        """Calculate the median of a column of data.

        Arguments:
            X {list} -- 2d list object with int or float.
            idxs {list} -- 1D list with int.
            feature {int} -- Feature number.
            sorted_idxs_2d {list} -- 2D list with int.

        Returns:
            list -- The row index corresponding to the median of this column.
        """

        n = len(idxs)
        # Ignoring the number of column elements is odd and even.
        k = n // 2
        # Get all the indexes and elements of column j as tuples.
        col = map(lambda i: (i, X[i][feature]), idxs)
        # Sort the tuples by the elements' values
        # and get the corresponding indexes.
        sorted_idxs = map(lambda x: x[0], sorted(col, key=lambda x: x[1]))
        # Search the median value.
        median_idx = list(sorted_idxs)[k]
        return median_idx

    def _get_variance(self, X, y, idxs, feature):
        """Calculate the variance of a column of data.

        Arguments:
            X {list} -- 2d list object with int or float.
            idxs {list} -- 1D list with int.
            feature {int} -- Feature number.

        Returns:
            float -- variance
        """

        n = len(idxs)
        col_sum = col_sum_sqr = 0

        for idx in idxs:
            xi = X[idx][feature]
            col_sum += xi
            col_sum_sqr += xi ** 2
        # D(X) = E{[X-E(X)]^2} = E(X^2)-[E(X)]^2

        return col_sum_sqr / n - (col_sum / n) ** 2

    def _choose_feature(self, X, y, idxs):
        """Choose the feature which has maximum variance.

        Arguments:
            X {list} -- 2d list object with int or float.
            idxs {list} -- 1D list with int.

        Returns:
            feature number {int}
        """

        m = len(X[0])
        variances = map(lambda j: (
            j, self._get_variance(X, y, idxs, j)), range(m))
        return max(variances, key=lambda x: x[1])[0]

    def _split_feature(self, X, idxs, feature, median_idx):
        """Split indexes into two arrays according to split point.

        Arguments:
            X {list} -- 2d list object with int or float.
            idx {list} -- Indexes, 1d list object with int.
            feature {int} -- Feature number.
            median_idx {float} -- Median index of the feature.

        Returns:
            list -- [left idx, right idx]
        """

        idxs_split = [[], []]
        split_val = X[median_idx][feature]

        for idx in idxs:
            # Keep the split point in current node.
            if idx == median_idx:
                continue
            # Split

            xi = X[idx][feature]
            if xi < split_val:
                idxs_split[0].append(idx)
            else:
                idxs_split[1].append(idx)
        return idxs_split

    def build_tree(self, X, y, node):
        """Build a KD Tree. The data should be scaled so as to calculate variances.

        Arguments:
            X {list} -- 2d list object with int or float.
            y {list} -- 1d list object with int or float.
        """

        # Initialize with node, indexes
        nd = self.root
        idxs = range(len(X))
        que = [(nd, idxs)]
        while que:
            nd, idxs = que.pop(0)
            n = len(idxs)
            # Stop split if there is only one element in this node
            if (n <= self.num):
                if(nd.class_num/nd.num<0.7):
                    0==0
                else:
                    node_one = []
                    for i in range(n):
                        node_one.append([X[idxs[i]], y[idxs[i]]])
                    node.append(node_one)
                    nd.split = (X[idxs[0]], y[idxs[0]])
                    continue
            # Split
            feature = self._choose_feature(X, y, idxs)
            median_idx = self._get_median_idx(X, idxs, feature)
            #print(feature)

            idxs_left, idxs_right = self._split_feature(
                X, idxs, feature, median_idx)
            # Update properties of current node
            nd.feature = feature
            nd.split = (X[median_idx], y[median_idx])
            # Put children of current node in que
            if(len(idxs_left)==0 or len(idxs_right)==0):
                node_one = []
                if(len(idxs_left)==0):
                    for k in range(len(idxs_right)):
                        node_one.append([X[idxs_right[k]], y[idxs_right[k]]])
                if(len(idxs_right)==0):
                    for k in range(len(idxs_left)):
                        node_one.append([X[idxs_left[k]], y[idxs_left[k]]])
                node.append(node_one)
                continue

            y_left=[]
            y_right=[]
            for i in idxs_left:
                if(y[i] not in y_left):
                    y_left.append(y[i])
            for i in idxs_right:
                if(y[i] not in y_right):
                    y_right.append(y[i])

            #print(idxs_left)
            if idxs_left != []:
                nd.left = Node()
                nd.left.father = nd
                nd.left.class_num=len(y_left)
                nd.left.num=len(idxs_left)
                que.append((nd.left, idxs_left))
            if idxs_right != []:
                nd.right = Node()
                nd.right.class_num=len(y_right)
                nd.right.father = nd
                nd.right.num=len(idxs_right)
                que.append((nd.right, idxs_right))
    def getAllleaves(self):
        nd=self.root
        queue=[]
        leaves=[]
        queue.append(nd)
        while(len(queue)!=0):
            front=queue[0]
            if(front.left!=None):
                queue.append(front.left)
            if(front.right!=None):
                queue.append(front.right)

            if(front.left!=None and front.right!=None):
                leaves.append(front)
            queue.pop(0)
        return leaves

    def _search(self, Xi, nd):
        """Search Xi from the KDTree until Xi is at an leafnode.

        Arguments:
            Xi {list} -- 1d list with int or float.

        Returns:
            node -- Leafnode.
        """

        while nd.left or nd.right:
            if not nd.left:
                nd = nd.right
            elif not nd.right:
                nd = nd.left
            else:
                if Xi[nd.feature] < nd.split[0][nd.feature]:
                    nd = nd.left
                else:
                    nd = nd.right
        return nd

    def _get_eu_dist(self, Xi, nd):
        """Calculate euclidean distance between Xi and node.

        Arguments:
            Xi {list} -- 1d list with int or float.
            nd {node}

        Returns:
            float -- Euclidean distance.
        """

        X0 = nd.split[0]
        return get_euclidean_distance(Xi, X0)

    def _get_hyper_plane_dist(self, Xi, nd):
        """Calculate euclidean distance between Xi and hyper plane.

        Arguments:
            Xi {list} -- 1d list with int or float.
            nd {node}

        Returns:
            float -- Euclidean distance.
        """

        j = nd.feature
        X0 = nd.split[0]
        return abs(Xi[j] - X0[j])

    def nearest_neighbour_search(self, Xi):
        """Nearest neighbour search and backtracking.

        Arguments:
            Xi {list} -- 1d list with int or float.

        Returns:
            node -- The nearest node to Xi.
        """

        # The leaf node after searching Xi.
        dist_best = float("inf")
        nd_best = self._search(Xi, self.root)
        que = [(self.root, nd_best)]
        while que:
            nd_root, nd_cur = que.pop(0)
            # Calculate distance between Xi and root node
            dist = self._get_eu_dist(Xi, nd_root)
            # Update best node and distance.
            if dist < dist_best:
                dist_best, nd_best = dist, nd_root
            while nd_cur is not nd_root:
                # Calculate distance between Xi and current node
                dist = self._get_eu_dist(Xi, nd_cur)
                # Update best node, distance and visit flag.
                if dist < dist_best:
                    dist_best, nd_best = dist, nd_cur
                # If it's necessary to visit brother node.
                if nd_cur.brother and dist > \
                        self._get_hyper_plane_dist(Xi, nd_cur.father):
                    _nd_best = self._search(Xi, nd_cur.brother)
                    que.append((nd_cur.brother, _nd_best))
                # Back track.
                nd_cur = nd_cur.father

        return nd_best
