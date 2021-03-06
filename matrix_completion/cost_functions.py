import tensorflow as tf

lamb = 0
lamb_tf=0
matrix_cost=0
graph_cost=0

def frobenius(A,tf_mask,WH,WTLW):
    global graph_cost
    global matrix_cost
    global lamb 
    lamb_tf = tf.constant(float(lamb))
    matrix_cost = tf.reduce_sum(tf.pow(tf.boolean_mask(A, tf_mask)
                         - tf.boolean_mask(WH, tf_mask), 2))
    graph_cost = tf.linalg.trace(tf.multiply(lamb_tf,WTLW))
    cost = tf.add(matrix_cost,graph_cost)
    return cost

def kl(x, y):
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)
    return tf.distributions.kl_divergence(X, Y)

def KL(A,tf_mask,WH,WTLW):
    lamb_tf = tf.constant(float(lamb))
    matrix_cost = kl(tf.boolean_mask(A+.00001, tf_mask),tf.boolean_mask(WH, tf_mask))
    graph_cost = tf.linalg.trace(tf.multiply(lamb_tf,WTLW))
    cost = tf.add(matrix_cost,graph_cost)
    return cost
