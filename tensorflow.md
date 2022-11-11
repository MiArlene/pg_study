# tensorflow



session 会话控制

variable 变量

placeholder 传入值

```python
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1*x_data + 0.3

#create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variable()
# create tensorflow structure end
sess = tf.Session()
sess.run(init)

## tensorflow 2.0 之后版本取消了 session 机制

for step in range(100):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases)) # 输出中间步骤的参数
   
```

## Session 会话机制

```python
import tensorflow as tf

matrix1 = tf.constant([[3,3]]) # 1行2列
matrix2 = tf.constant([[2],[2]]) # 2行一列
product = tf.matmul(matrix1, matrix2)  # matrix multiply

# method 1
sess = tf.Session()  ## session 是一个object，因此需要大写
result = sess.run(product)  # 每run 一次， tensorflow才会运行结构
print(result)
sess.close()  

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)  # 会自动执行 sess.close()
    print(result2)
```



## Variable 变量

```python
import tensorflow as tf
## 变量的定义一定要通过 tf.Variable()
state = tf.Variable(0, name = 'counter')
print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)  # 变量的更新，更新的是state

init = tf.initialize_all_variables() # 初始化所有的变量 才会激活变量
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(updata)
        print(sess.run(state))
```

## placeholder 传入值

placeholder 可以动态传入参数  和 feed_dict同时使用

```python
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict = {input1:[7.], input2:[2.]}))

```



## saver 保存读取

``` python
import tensorflow as tf
import numpy as np
## Save to file 
# remember to define the same dtype and shape when restore
W = tf.Variable([[1,2,3],[3,4,5]], dtype = tf.float32, name = 'weights')
b = tf.Variable([[1,2,3]], dtype = tf.float32, name = 'biase')

init = tf.initialize_all_variables()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Save to path:", save_path)

  

# restore variables
# redefine the same shape and same type for your variables

W = tf.Variables(np.arange(6).reshape((2,3)), dtype = tf.float32, name = 'weights')
b = tf.Variables(np.arange(3).reshape((1,3)), dtype = tf.float32, name = 'biases')

# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.restore(sess, "my_net/save_net.ckpt")
    print("weights", sess.run(W))
    print("biases", sess.run(b))
```



## RNN lstm 循环神经网络 （分类例子）

```python
import tensorflow as tf
from tensorflow.examples.turorials.minist import input_data

# this data
minist = input_data.read_data_sets('MNIST_data', one_hot = True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_inputs = 28  # MINST data  input(img shape: 28 28)
n_steps = 28 # time steps
n_hidden_units = 128  # neurons in hidden layer
n_class = 10 

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classs])

# Define weights
weights = {
    # 28 128
    'in':tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_class]))
}
biases = {
    'in':tf.Variable(tf.constant(0.1), shape = [n_hidden_units,]),
    'out':tf.Variable(tf.constant(0.1, shape  = [n_class,]))
}

def RNN(X, weights, biases):
    
    # hidden layer for input to cell
    # X 128 batch  28  steps 28 inputs
    X = tf.reshpe(X,[-1,n_input])
    X_in = tf.matmul(X, weights['in']) + biase['in']
    X_in = tf.reshape(X_in, [-1,n_steps, n_hidden_units])
     
    # cell
    
    
    # hidden layer for output as final results
    
    
    
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs, batch_ys = minist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict = {
            x:batch_xs,
            y:batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict = {
                x:batch_xs,
                y:batch_ys,
            }))
            step += 1
            


```



# 迁移学习

站在巨人的肩膀上俯瞰世界。

可以使用他人的模型和参数，或者用某些层的参数，后面接上自己的层，只训练增加层的参数，会大大缩短训练的时间。