from sklearn.base import BaseEstimator
import numpy as np
import tensorflow as tf
import os
import shutil


class Node:
    def __init__(self, path_prob, depth, n_classes, max_depth, lamda, gamma, name):
        self.path_prob = path_prob
        self.depth = depth
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.lamda = lamda
        self.gamma = gamma
        self.name = name
        self.is_leaf = depth == max_depth

        self.eps = 1e-8
        self.children = ()
        self.prob = None

    def forward(self, x):
        hidden_dim, activation = (self.n_classes, tf.nn.softmax) if self.is_leaf else (1, tf.nn.sigmoid)
        self.prob = tf.layers.dense(x, hidden_dim, activation=activation)
        if not self.is_leaf:
            parent_params = {k: v for k, v in self.__dict__.items() if k in ['n_classes', 'max_depth', 'lamda', 'gamma']}
            self.children = Node(self.path_prob * self.prob, self.depth + 1, name='{}/{}_left'.format(self.name, self.depth + 1), **parent_params), \
                            Node(self.path_prob * (1 - self.prob), self.depth + 1, name='{}/{}_right'.format(self.name, self.depth + 1), **parent_params)

    def get_loss(self, y):
        if self.is_leaf:
            y_oh = tf.one_hot(y, self.n_classes)
            return -tf.reduce_mean(self.path_prob * tf.log(tf.reduce_sum(y_oh * self.prob + self.eps, axis=1)))
        alpha = tf.reduce_mean(self.path_prob * self.prob + self.eps) / tf.reduce_mean(self.path_prob + self.eps)
        return -0.5 * self.lamda * (tf.log(alpha) + tf.log(1 - alpha)) * self.gamma ** self.depth


class Tree(BaseEstimator):
    """
    https://arxiv.org/pdf/1711.09784.pdf    
    """
    def __init__(self, sess, node_params, nn_params, weight=True, verbose=False):
        self.sess = sess
        self.x = tf.placeholder(tf.float32, (None, nn_params['n_features']), 'x')
        self.y = tf.placeholder(tf.uint8, (None,), 'y')
        self.loss = tf.Variable(0, dtype=tf.float32, name='loss')
        self.weight = weight
        self.verbose = verbose

        self.n_epochs = nn_params.get('n_epochs', 1)
        self.batch_size = nn_params.get('batch_size', 128)

        self.leaf_probs = []
        self.leaf_distribution = []
        nodes = [Node(path_prob=tf.constant(1, dtype=tf.float32, name='path_prob'), depth=0, name='root', **node_params)]
        for node in nodes:
            with tf.variable_scope(node.name):
                node.forward(self.x)
                self.loss = self.loss + node.get_loss(self.y)
            nodes.extend(node.children)
            if node.is_leaf:
                self.leaf_probs.append(node.prob)
                self.leaf_distribution.append(node.path_prob)
        self.leaf_probs = tf.reshape(tf.concat(self.leaf_probs, axis=1), [-1, len(self.leaf_distribution), node_params['n_classes']])
        self.leaf_distribution = tf.concat(self.leaf_distribution, axis=1)
        optimizer = tf.train.AdamOptimizer(learning_rate=nn_params.get('learning_rate', 0.01))
        self.train_op = optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists('graph'):
            os.makedirs('graph')
        shutil.rmtree('graph')
        self.writer = tf.summary.FileWriter('./graph', self.sess.graph)

    def fit(self, X, y):
        for i in range(self.n_epochs):
            start, end = 0, self.batch_size
            L = 0
            while start <= X.shape[0]:
                self.sess.run(self.train_op, feed_dict={self.x: X[start:end], self.y: y[start:end]})
                if self.verbose:
                    L += self.sess.run(self.loss, feed_dict={self.x: X[start:end], self.y: y[start:end]})
                start, end = end, end + self.batch_size
            if self.verbose:
                print('epoch: {}, loss: {}'.format(i, L))
        self.writer.close()

    def predict_proba(self, X):
        leaf_distribution, leaf_probs = self.sess.run([self.leaf_distribution, self.leaf_probs], feed_dict={self.x: X})
        if self.weight:
            return (leaf_probs * leaf_distribution[:, :, None]).sum(axis=1)
        else:
            leaf_indices = np.argmax(leaf_distribution, axis=1)
            return leaf_probs[range(X.shape[0]), leaf_indices]
