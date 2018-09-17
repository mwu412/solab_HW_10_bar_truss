import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sy
from scipy.optimize import minimize
import csv

from IPython.display import display
import glob, os

E = 2e11  # Young's modulous
Yield_Stress = 250e6  # unit: Pa
dict_node = {}
dict_element = {}
list_load_full = np.array([0, 0, 20000, 0, 0, -25000, 0, 0])
fixed_global_dof = [1, 2, 4, 7, 8]
element_stress = []

class element:
    def __init__(self, from_node, to_node, radius):
        from_node = int(from_node)
        to_node = int(to_node)
        self.from_node = from_node
        self.to_node = to_node
        self.radius = radius

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
    with open('/home/mike/Documents/vs_code/solab_HW_10_bar_truss/node_fake.csv', newline='') as csvfile:
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
    with open('/home/mike/Documents/vs_code/solab_HW_10_bar_truss/element_fake.csv', newline='') as csvfile:
        rows_element = csv.reader(csvfile, delimiter=',')

        count = 0
        for row in rows_element:
            number, from_node, to_node, radius = row
            if number == 'element':
                count += 1
                continue
            dict_element[count] = element(from_node, to_node, radius) # The key is int
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
    Compression = True
    critical_stress = 0
    critical_radius = 0



    read_csvs()

    num = len(dict_node)  # numbers of elements
    #K_matrix = np.zeros((num*2, num*2))
    K_matrix = pd.DataFrame(np.zeros((num*2, num*2)), index = list(range(1,num*2+1)), columns = list(range(1,num*2+1)))    
    
    for number, element in dict_element.items():  # i th element
        for label in stiffness_matrix(number).columns:  # label is int
            for index in stiffness_matrix(number).index:  #index is int 
                K_matrix.iloc[index-1, label-1] += stiffness_matrix(number).loc[index, label]/element.length


    # drop fixed global dof
    K_matrix = K_matrix.drop(fixed_global_dof, axis=0)
    K_matrix = K_matrix.drop(fixed_global_dof, axis=1)

    fixed_global_dof_minus1 = [x-1 for x in fixed_global_dof]
    list_load = np.delete(list_load_full, fixed_global_dof_minus1)

    # --- test only --- #
    K_matrix *= 295e5
    print(K_matrix)

    # Q = inv(K) * Fn
    Q_arr = np.matmul(np.linalg.inv(K_matrix),list_load)
    print(Q_arr)

    # max displacement
    max_displace = np.amax(np.abs(Q_arr))
    print(max_displace)

    # max stress
    Q_df = pd.DataFrame(Q_arr, index = K_matrix.index)
    print(Q_df)
    
    for n, element in dict_element.items():
        c = element.cos
        s = element.sin
        dof_list = element.global_dof_list()
        list_q = []
        for dof in dof_list:
            if dof in Q_df.index:
                list_q.extend(Q_df.loc[dof,:])
            else: list_q.extend([0])
        #print('q: ', list_q)

        dot = np.dot(np.array([-c, -s, c, s]), np.array(list_q))
        element_stress.append(295e5/element.length*dot)
        #-- element_stress.append(E/element.length*dot)

        #print('stress: ', element_stress)

    max_stress = np.amax(np.abs(np.array(element_stress)))
    print(max_stress)

    if max_stress in element_stress:
        # Compression = False
        critical_radius = dict_element[element_stress.index(max_stress)+1].radius
    

    # check buckling
    else:  # Compression = True
        max_stress_minus = -max_stress
        critical_radius = dict_element[element_stress.index(max_stress_minus)+1].radius
        #critical_stress = np.pi**3*E*critical_radius**4/4/dict_element[element_stress.index(max_stress_minus)+1].length

    print(critical_radius)

    # create object function

    def tuple_symbols():
        object_f = 0
        list_sym = [] 
        for n, element in dict_element.items():  
            r_th = sy.symbols('r' + str(element.radius))
            object_f += r_th**2
            list_sym.append(r_th)
        tuple_sym = tuple(list_sym)
        return object_f, tuple_sym

    object_f, syms = tuple_symbols()

    ###############################
    object_f_n = sy.lambdify(syms, object_f, modules='numpy')  # create for numerical 

    # scipy_optimization_minimize
    def f(x):         
        return object_f_n(lambda: x[n-1] for n in dict_element)

    

if __name__ == '__main__':
    main()





