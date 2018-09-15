import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sympy import *
from scipy.optimize import minimize
import csv

from IPython.display import display
import glob, os

E = 2e11  # Young's modulous
dict_node = {}
dict_element = {}
list_load_full = np.array([0, 0, 20000, 0, 0, -25000, 0, 0])
fixed_global_dof = [1, 2, 4, 7, 8]

class element:
    def __init__(self, from_node, to_node):
        from_node = int(from_node)
        to_node = int(to_node)
        self.from_node = from_node
        self.to_node = to_node

        x_from = dict_node[from_node].x
        y_from = dict_node[from_node].y
        x_to = dict_node[to_node].x
        y_to = dict_node[to_node].y
        self.length = np.sqrt((x_to - x_from)**2 + (y_to - y_from)**2)
        self.cos = (x_to - x_from)/self.length
        self.sin = (y_to - y_from)/self.length

    def global_dof_list(self):
        return [self.from_node*2-1, self.from_node*2, self.to_node*2-1, self.to_node*2]

    def print_element(self):  # for test 
        print(self.length, self.cos, self.sin)
    
class node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def read_csvs():
    # read node.csv
    with open('/Users/mikemac/Documents/vs_code/solab_HW_10_bar_truss/node_fake.csv', newline='') as csvfile:
        rows_node = csv.reader(csvfile, delimiter=',')

        count = 0
        for row in rows_node:
            number, x, y = row

            if number == 'node':
                count += 1
                continue

            x = float(x)
            y = float(y)
            dict_node[count] = node(x, y)  # The key is int
            count += 1

    # read element.csv
    with open('/Users/mikemac/Documents/vs_code/solab_HW_10_bar_truss/element_fake.csv', newline='') as csvfile:
        rows_element = csv.reader(csvfile, delimiter=',')

        count = 0
        for row in rows_element:
            number, from_node, to_node = row
            if number == 'element':
                count += 1
                continue
            dict_element[count] = element(from_node, to_node) # The key is int
            count += 1

    # read load.csv
    """
    with open('/Users/mikemac/Documents/vs_code/solab_HW_10_bar_truss/load_fake.csv', newline='') as csvfile:
        rows_load = csv.reader(csvfile, delimiter=',')
        for i in range(len(dict_node)):
            i += 1
            findit = False

            for row in rows_load:
                load_node, x_load, y_load = row
                print(i)

                if load_node == str(i):
                    list_load.extend([x_load, y_load])
                    findit = True
                    break
                    
            if findit == False:
                    list_load.extend([0, 0])
    """
            

def stiffness_matrix(i):  # i th element
    labels = dict_element[i].global_dof_list()
    indexes = dict_element[i].global_dof_list()
    c = dict_element[i].cos
    s = dict_element[i].sin

    arr=[[c**2, c*s, -c**2, -c*s],[c*s, s**2, -c*s, -s**2],
    [-c**2, -c*s, c**2, c*s], [-c*s, -s**2, c*s, s**2]]
    return pd.DataFrame(arr, index = indexes, columns = labels)

def main():
    read_csvs()

    num = len(dict_node)  # numbers of elements
    #K_matrix = np.zeros((num*2, num*2))
    K_matrix = pd.DataFrame(np.zeros((num*2, num*2)), index = list(range(1,num*2+1)), columns = list(range(1,num*2+1)))    
    
    for i in range(len(dict_element)):  # i th element
        i += 1
        for label in stiffness_matrix(i).columns:  # label is int
            for index in stiffness_matrix(i).index:  #index is int 
                K_matrix.iloc[index-1, label-1] += stiffness_matrix(i).loc[index, label]/dict_element[i].length

    # drop fixed global dof
    K_matrix = K_matrix.drop(fixed_global_dof, axis=0)
    K_matrix = K_matrix.drop(fixed_global_dof, axis=1)

    fixed_global_dof_minus1 = [x-1 for x in fixed_global_dof]
    list_load = np.delete(list_load_full, fixed_global_dof_minus1)

    # --- test only --- #
    K_matrix *= 295e8
    print(K_matrix)

    # Q = inv(K) * Fn
    Q = np.matmul(np.linalg.inv(K_matrix),list_load)
    print(Q)

    # max displacement
    max_displace = np.abs(np.amax(Q))
    print(max_displace)



if __name__ == '__main__':
    main()


