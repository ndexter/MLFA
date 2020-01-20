import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

tf.executing_eagerly()

x_data = np.array([[[ 1.,  2.,  3.], [ 4.,  5.,  6.]],
                   [[ 7.,  8.,  9.], [10., 11., 12.]],
                   [[13., 14., 15.], [16., 17., 18.]]])

print x_data 

x = tf.convert_to_tensor(x_data, dtype = tf.float32)

print x

print tf.slice(x,[0,0,0],[3,2,3]) # the whole tensor
print tf.slice(x,[1,0,0],[2,2,3]) 
print tf.slice(x,[2,0,0],[1,2,3]) 
print tf.slice(x,[0,1,0],[3,1,3])
print tf.slice(x,[0,0,1],[3,2,1])
print tf.slice(x,[0,0,1],[3,2,2])

x_data = np.array([[1,2,3],[4,5,6],[7,8,9]])

print x_data 

x = tf.convert_to_tensor(x_data, dtype = tf.float32)

print x

print tf.slice(x,[0,0],[3,3]) # the whole tensor
print tf.slice(x,[0,0],[3,2]) # 2 cols from first col
print tf.slice(x,[0,1],[3,2]) # 2 cols from 2nd col
print tf.slice(x,[0,0],[3,1]) # first col
print tf.slice(x,[0,1],[3,1]) # 2nd col

print '\n\n x\'s 3rd column is'
print tf.slice(x,[0,2],[3,1]) # 3rd col

print '\n\n'
print 'computing the scalar product of matrix x with its 3rd col'
print x*tf.slice(x,[0,2],[3,1]) # scalar product of the elements of matrix x with its 3rd col
print 'computing the scalar product of matrix x with its 3rd col, then add 3rd col'
print x*tf.slice(x,[0,2],[3,1]) + tf.slice(x,[0,2],[3,1]) # scalar product of the elements of matrix x with its 3rd col

print 'matvec product of matrix x with its 3rd col'
print tf.matmul(x,tf.slice(x,[0,2],[3,1])) # matvec product of matrix x with its 3rd col

print 'matvec product of matrix x with its 3rd col, then add 3rd col'
print tf.matmul(x,tf.slice(x,[0,2],[3,1])) + tf.slice(x,[0,2],[3,1]) # matvec product of matrix x with its 3rd col
