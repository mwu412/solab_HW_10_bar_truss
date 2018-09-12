import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from IPython.display import display
import glob, os

dict_node = {}
dict_element = {}

class element:
    def __init__(self, from_node, to_node):

        from_node = int(from_node)
        to_node = int(to_node)

        x_from = dict_node[from_node].x
        y_from = dict_node[from_node].y
        x_to = dict_node[to_node].x
        y_to = dict_node[to_node].y
        self.length = np.sqrt((x_to - x_from)**2 + (y_to - y_from)**2)
        self.cos = (x_to - x_from)/self.length
        self.sin = (y_to - y_from)/self.length

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

    for key in dict_element:
        dict_element[key].print_element()

def main():
    read_csvs()

    for key in dict_element:
        



if __name__ == '__main__':
    main()


