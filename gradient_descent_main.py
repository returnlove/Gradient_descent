# -*- coding: utf-8 -*-

# y = mx + b

# ideally m= (y2-y1)/(x2-x1), once we know m, we can derive value of b
# but let's do gradient descent to find best parameters

from numpy import *



def compute_cost_given_points(b,m,points):
    total_error = 0
    for i in range(0,len(points)):
        x = points[i, 0] # for each row, first column is x
        y = points[i, 0] # and second column is y
        total_error += (y - (m*x + b)) **2 # formula: https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png
        #print(total_error)
    return total_error / float(len(points))
    
def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient= 0
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_current)
    return [new_b, new_m]

# some text
    

                
points =  genfromtxt("data.csv", delimiter = ",")
#print(points)     
initial_b = 0  # initial y-intercept guess
initial_m = 0.5 # initial slope guess
print(compute_cost_given_points(initial_b, initial_m, points))
    