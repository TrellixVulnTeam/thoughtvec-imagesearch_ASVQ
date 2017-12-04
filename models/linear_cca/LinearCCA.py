import tensorflow as tf
import numpy as np

class LinearCCA():
    def __init__(self, nr_contrastive=50, image_dim=4096, sentence_dim=4800, projection_dim=1000):
        self.image_dim = image_dim
        self.sentence_dim = sentence_dim
        self.projection_dim = projection_dim

        self.x = tf.placeholder(tf.float32, [None, image_dim])
        self.y = tf.placeholder(tf.float32, [None, sentence_dim])

        self.x_contrastive = tf.placeholder(tf.float32, [None, image_dim])
        self.y_contrastive = tf.placeholder(tf.float32, [None, sentence_dim])

        self.Ux = self.getProjectionU(self.x)
        self.Vy = self.getProjectionV(self.y)

        self.Ux_contrastive = self.getProjectionU(self.x_contrastive, reuse=True)
        self.Vy_contrastive = self.getProjectionV(self.y_contrastive, reuse=True)

        self.loss = self.loss_with_contrastive_terms(self.Ux, self.Vy, self.Ux_contrastive, self.Vy_contrastive)

        self.opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.90).minimize(self.loss)

    def getProjectionU(self, input_, reuse=False):
        with tf.variable_scope("projectionU", reuse=reuse):
            U = tf.layers.dense(input_, units=self.projection_dim, activation=None)
        return U

    def getProjectionV(self, input_, reuse=False):
        with tf.variable_scope("projectionV", reuse=reuse):
            V = tf.layers.dense(input_, units=self.projection_dim, activation=None)
        return V

    def l2norm(self, vec):
        return tf.sqrt(tf.reduce_sum(vec * vec))

    def cosine_similarity(self, vecs1, vecs2):
        unit_vecs1 = tf.map_fn(lambda v: v / self.l2norm(v), vecs1)
        unit_vecs2 = tf.map_fn(lambda v: v / self.l2norm(v), vecs2)

        return tf.reduce_sum(unit_vecs1 * unit_vecs2, axis=1)

    def loss_with_contrastive_terms(self, Ux, Vy, Ux_contrastive, Vy_contrastive, alpha=0.20):

        sim_Ux_Vy = self.cosine_similarity(Ux, Vy)

        sim_Ux_Vy_contrastive = self.cosine_similarity(Ux, Vy_contrastive)

        sim_Ux_contrastive_Vy = self.cosine_similarity(Ux_contrastive, Vy)

        x_anchored_margin = alpha - sim_Ux_Vy + sim_Ux_Vy_contrastive
        x_anchored_margin = tf.maximum(0.0, x_anchored_margin)

        y_anchored_margin = alpha - sim_Ux_Vy + sim_Ux_contrastive_Vy
        y_anchored_margin = tf.maximum(0.0, y_anchored_margin)

        x_anchored_margin_sum_per_truepair = tf.reduce_sum(x_anchored_margin)
        y_anchored_margin_sum_per_truepair = tf.reduce_sum(y_anchored_margin)

        loss = x_anchored_margin_sum_per_truepair + y_anchored_margin_sum_per_truepair

        return loss

if __name__=='__main__':
    cca = LinearCCA(1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        vars = [(v.name, sess.run([v])) for v in tf.trainable_variables()]
        print(vars)
        sess.run(tf.global_variables_initializer())
        output = sess.run(cca.loss, feed_dict={cca.x: np.random.normal(loc=0.0, scale=1.0, size=(1, 4096)),
                                               cca.y: np.random.normal(loc=0.0, scale=1.0, size=(1, 4800)),
                                               cca.x_contrastive: np.random.normal(loc=0.0, scale=1.0, size=(50, 4096)),
                                               cca.y_contrastive: np.random.normal(loc=0.0, scale=1.0, size=(50, 4800))
                                               })
        print(output)