import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

# To use this DQN module you need to "pip install" the packets above in the Pycharm terminal
# When you input "pip install tensorflow", it will automatically install tensorflow with the version higher than 2.0
# But in the DQN code provided from the Internet, some functions are usable only when the version is lower than 2.0
# So in my code, there are many sentences like "tf.compat.v1". This is used to invoke functions of lower versions
class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_observations,
                 alpha=0.01,
                 gamma=0.9,
                 epsilon=0.9,
                 replace_target_iter=300,
                 buffer_size=500,
                 batch_size=32,
                 epsilon_increment=0.001,
                 output_graph=True
                 ):
        self.n_actions = n_actions                                                  # available actions (left, right)
        self.n_observations = n_observations                                        # available states (position, angle)
        self.alpha = alpha                                                          # learning rate
        self.gamma = gamma                                                          # reward decay
        self.epsilon_max = epsilon                                                  # maximum e-greedy allowed, which is for possible random steps
        self.replace_target_iter = replace_target_iter
        self.buffer_size = buffer_size
        self.buffer_counter = 0  # 统计目前进入过buffer的数量
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_increment                                  # increase e-greedy gradually until it reaches epsilon_max
        self.epsilon = 0 if epsilon_increment is not None else self.epsilon_max     # initial e-greedy, which means at first I move totally randomly

        self.learn_step_counter = 0                                                 # total learning step, increasing epsilon

        self.buffer = np.zeros((self.buffer_size, n_observations * 2 + 2))          # initialize zero buffer [s, a, r, s_]

        self.build_net()                                                            # consist of [target_net, evaluate_net]

        # Update all parameters in the eval network to the target network
        target_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        eval_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.compat.v1.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.compat.v1.assign(t, e) for t, e in zip(target_params, eval_params)]

        self.sess = tf.compat.v1.Session()

        # This is used for showing the graph of the neural network in tensorboard. You should first execute the code for once.
        # Then in the PyCharm terminal, input "tensorboard --logdir=logs/", then click http://localhost:6006/ to check the graph.
        if output_graph:
            tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.cost_his = [] # store the error at each step, which will be used in plot_cost()

    def build_net(self):
        tf.compat.v1.disable_eager_execution() # prevent "placeholder" for reporting errors
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_observations], name='s')
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_observations], name='s_')
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='q_target')

        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)
        with tf.compat.v1.variable_scope('eval_net'):
            c_names, n_l1 = ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], 10

            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_observations, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.name_scope('loss'):
            self.loss = tf.reduce_sum(tf.compat.v1.squared_difference(self.q_target, self.q_eval))
        with tf.name_scope('train'):
            self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.alpha).minimize(self.loss)

        with tf.compat.v1.variable_scope('target_net'):
            c_names = ['target_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]

            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_observations, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.buffer_counter % self.buffer_size
        self.buffer[index, :] = transition

        self.buffer_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        sample_index = np.random.choice(min(self.buffer_counter, self.buffer_size), size=self.batch_size)
        batch_buffer = self.buffer[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_buffer[:, -self.n_observations:],  # fixed params
                self.s: batch_buffer[:, :self.n_observations],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_buffer[:, self.n_observations].astype(int)
        reward = batch_buffer[:, self.n_observations + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_buffer[:, :self.n_observations],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return self.cost

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()