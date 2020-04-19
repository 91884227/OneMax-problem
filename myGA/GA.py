#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tnrange, tqdm_notebook


# In[2]:


class Roulette_GA():
    def __init__(self, generations_ = 200):
        # initial parent:
        buf = np.random.randint(2, size = 60*500)
        self.parent_list = np.resize(buf, (500,60))
        self.chance_prop = []
        self.generations = generations_
        self.simulation_result = []
        
    def crossover(self, parent_1_, parent_2_):
        parent_len = len(parent_1_)
        ranfom_position = np.random.choice(parent_len, 1)[0]
        if( ranfom_position > parent_len/2 ):
            children_1 = np.hstack((parent_1_[:ranfom_position], parent_2_[ranfom_position:]))
            children_2 = np.hstack((parent_2_[:ranfom_position], parent_1_[ranfom_position:]))
        else :
            children_1 = np.hstack((parent_2_[:ranfom_position], parent_1_[ranfom_position:]))
            children_2 = np.hstack((parent_1_[:ranfom_position], parent_2_[ranfom_position:]))  
            
        return( (children_1, children_2))
    
    def Fitness(self, parent_):
        return( sum(parent_) )
    
    def Roulette(self):
        buf = list(map(self.Fitness, self.parent_list))
        self.chance_prop = np.asarray(buf)/sum(buf)
    
    def Recombination_2(self, _):
        buf = np.arange( len(self.parent_list) )
        select_index = np.random.choice( buf, 2, p = self.chance_prop, replace = False)
        parent_1 = self.parent_list[ select_index[0] ]
        parent_2 = self.parent_list[ select_index[1] ]
        return(  self.crossover( parent_1, parent_2) )
    
    def Recombination_all(self):
        parent_list_len = len(self.parent_list)
        one_parent_len = len(self.parent_list[0])
        buf = list(map(self.Recombination_2, np.arange( parent_list_len/2 )))
        self.parent_list = np.reshape(np.asarray(buf), ( parent_list_len, one_parent_len))  
    
    def oneRound(self):
        self.Roulette()
        self.Recombination_all()
        max_fitness =  max(list(map(self.Fitness, self.parent_list)))
        # print("max score: %d" % max(list(map(self.Fitness, self.parent_list))))
        return( max_fitness )
    
    def draw_plot(self):
        buf = np.array(self.simulation_result)
        buf = np.mean(buf, axis=0)
        print("average of the best fitness values: %.2f" % buf[-1])
        plt.plot(buf)
        plt.ylabel('Fitness')
        plt.show()
    
    def simulation_one(self, generations = 200):
        buf = []
        for i in  range(generations) :
            buf = buf + [self.oneRound()]
        self.simulation_result.append( buf )
    
    def __call__(self, simulation_times):
        for i in tqdm_notebook( range(simulation_times) ):
            self.simulation_one(self.generations)   
        self.draw_plot()      


# In[5]:


class Tournament_GA(Roulette_GA):
    def Recombination_2(self, _):
        buf = np.arange( len(self.parent_list) )
        select_index_list_1 = np.random.choice( buf, 2, replace = False)
        select_index_1 = max(select_index_list_1, key = lambda k: self.Fitness(self.parent_list[ k]))
        select_index_list_2 = np.random.choice( buf, 2, replace = False)
        select_index_2 = max(select_index_list_2, key = lambda k: self.Fitness(self.parent_list[ k]))
        select_index = [select_index_1, select_index_2]
        parent_1 = self.parent_list[ select_index[0] ]
        parent_2 = self.parent_list[ select_index[1] ]
        return(  self.crossover( parent_1, parent_2) )


# In[6]:


class Q4(Roulette_GA):
    def Fitness(self, parent_):
        return( sum(parent_) + 800 )    


# In[7]:


class Q7(Tournament_GA):
    def Fitness(self, parent_):
        return( sum(parent_) + 800 )    

