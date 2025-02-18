class HybridLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, ratio=0.9, **kwargs):
        super(HybridLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.ratio = ratio
        self.num_tanh = int(units * ratio)
        self.num_sigmoid = units - self.num_tanh
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1] + self.units, 4 * self.units),
            initializer='glorot_uniform',
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(4 * self.units,),
            initializer='zeros',
            name='bias'
        )
        self.built = True
        
    def call(self, inputs, states):
        h_tm1, c_tm1 = states
        inputs_h = tf.concat([inputs, h_tm1], axis=-1)
        gates = tf.matmul(inputs_h, self.kernel) + self.bias
        
        # Split gates into input, forget, output, candidate
        i, f, o, c_hat = tf.split(gates, num_or_size_splits=4, axis=1)
        
        # Apply activations to gates
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        
        # Split candidate and apply hybrid activations
        c_hat_tanh = c_hat[:, :self.num_tanh]
        c_hat_sigmoid = c_hat[:, self.num_tanh:]
        c_hat_tanh = tf.tanh(c_hat_tanh)
        c_hat_sigmoid = tf.sigmoid(c_hat_sigmoid)
        c_hat = tf.concat([c_hat_tanh, c_hat_sigmoid], axis=1)
        
        # Update cell state
        c = f * c_tm1 + i * c_hat
        
        # Split cell state for hidden state
        c_tanh = c[:, :self.num_tanh]
        c_sigmoid = c[:, self.num_tanh:]
        
        # Compute hidden state with hybrid activations
        h_tanh = tf.tanh(c_tanh)
        h_sigmoid = tf.sigmoid(c_sigmoid)
        h = o * tf.concat([h_tanh, h_sigmoid], axis=1)
        
        return h, [h, c]