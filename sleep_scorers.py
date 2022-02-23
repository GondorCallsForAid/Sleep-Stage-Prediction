import tensorflow as tf
from attention import MultiHeadAttention


class CnnAttentionOnEpoch(tf.keras.Model):
    """
    Model to extract features from individual epochs (intra epoch learning)
    """

    def __init__(self):
        super(CnnAttentionOnEpoch, self).__init__()

        self.one_d_1 = tf.keras.layers.Conv1D(filters=32,
                                              kernel_size=200,
                                              strides=100,
                                              padding="valid"
                                              )

        self.mha_1 = MultiHeadAttention(d_model=32, num_heads=8)

    def call(self, inputs):
        # input_shape (Batch_size, 3000)
        x = self.one_d_1(inputs)

        # shape: (Batch_size, 29, 32)
        # do the attention stuff

        return x


