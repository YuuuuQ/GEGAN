#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import random
import argparse
import pandas as pd
import copy
import argparse
import os
import numpy as np
import math
import pandas as pd
import  torch
from    torch import nn, optim, autograd
from torch.autograd import Variable
from    torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import itertools

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(cuda)


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr_G_or", type=float, default=0.001)
parser.add_argument("--lr_G_final", type=float, default=0.001)
parser.add_argument("--lr_D", type=float, default=0.001)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--n_cpu", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--img_size", type=int, default=28)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--sample_interval", type=int, default=50)
parser.add_argument("--population_size", type=int, default=6)
parser.add_argument("--potence", type=float, default=0.8)
parser.add_argument("--fitness_up", type=float, default=0.8)
parser.add_argument("--fitness_down", type=float, default=0.4)
parser.add_argument("--interval",type=int,default=14)
parser.add_argument("--rate", type=int, default=12)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--num_dist", type=int, default=3)
parser.add_argument("--save_model", type=int, default=200)

opt = parser.parse_args([])
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


# In[3]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4  
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        C = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(C)
        return img,C
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 1, 3, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


# In[4]:



class Individual:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr_G_or, betas=(opt.b1, opt.b2))#用于优化generator和discriminator的参数
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))
        self.data = torch.empty(3, 4)#self.data和self.C是形状为(3, 4)的空的torch.Tensor
        self.C = torch.empty(3, 4)
        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.data = self.data.to(device)
            self.C = self.C.to(device)
            

        self.loss_G = 10000
        self.loss_D = 10000
        self.score = 0

    def update(self, z, real):
        self.data, self.C = self.generator(z)
        pf = self.discriminator(self.data)
        pr = self.discriminator(real)
        self.loss_G = -(adversarial_loss(pr, torch.ones_like(pr))+adversarial_loss(pf, torch.zeros_like(pf)))/2
        self.loss_D = (adversarial_loss(pr, torch.ones_like(pr))+adversarial_loss(pf, torch.zeros_like(pf)))/2

    def update_G(self, z, real):
        self.data, self.C = self.generator(z)
        pf = self.discriminator(self.data)
        pr = self.discriminator(real)
        self.optimizer_G.zero_grad()
        self.loss_G = -(adversarial_loss(pr, torch.ones_like(pr))+adversarial_loss(pf, torch.zeros_like(pf)))/2
        self.loss_G.backward()
        self.optimizer_G.step() 
    
    def update_D(self, z, real):
        self.data, self.C = self.generator(z)
        pf = self.discriminator(self.data)
        pr = self.discriminator(real)
        self.optimizer_D.zero_grad()
        self.loss_D = (adversarial_loss(pr, torch.ones_like(pr))+adversarial_loss(pf, torch.zeros_like(pf)))/2
        self.loss_D.backward()
        self.optimizer_D.step()
    def update_data(self, z):
        self.data, self.C = self.generator(z)
    def caculate_score(self):
        self.score = self.score + self.loss_D.item()
    def set_score(self):
        self.score = self.loss_D.item()


# In[5]:


def init_population(size):
    population = []
    for i in range(size):
        population.append(Individual())
    return population

def copy_population(population, z, real, lr):
    new_p = [None]*opt.population_size
    t_G_w = [None]*opt.population_size
    t_D_w = [None]*opt.population_size
    for n in range(opt.population_size):
        new_p[n] = Individual()
        t_G_w[n] = copy.deepcopy(population[n].generator.state_dict())
        t_D_w[n] = copy.deepcopy(population[n].discriminator.state_dict())
        new_p[n].generator.load_state_dict(t_G_w[n])
        new_p[n].discriminator.load_state_dict(t_D_w[n])
        new_p[n].optimizer_G = torch.optim.Adam(new_p[n].generator.parameters(), lr=lr, betas=(opt.b1, opt.b2))
        new_p[n].optimizer_D = torch.optim.Adam(new_p[n].discriminator.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))
        new_p[n].update(z, real)
        new_p[n].score = copy.deepcopy(population[n].score)
    return new_p

def copy_individual(individual, z, real):
    new = Individual()
    t_G_w = copy.deepcopy(individual.generator.state_dict())
    t_D_w[n] = copy.deepcopy(population[n].discriminator.state_dict())
    new_p.generator.load_state_dict(t_G_w[n])
    new_p[n].discriminator.load_state_dict(t_D_w[n])
    new_p.optimizer_G = torch.optim.Adam(new_p.generator.parameters(), lr=opt.lr_G, betas=(opt.b1, opt.b2))
    new_p[n].optimizer_D = torch.optim.Adam(new_p[n].discriminator.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))
    new_p.update(z, real)
    return new_p


def copy_fitness(population):
    fitness = [0]*opt.population_size
    n = 0
    for p in population:
        f = copy.deepcopy(p.score)
        fitness[n] = f
        n = n+1
    return fitness

def prob(cost, Fitness):
    sort_fitness = sorted(Fitness, reverse=False)
    min_fitness = sort_fitness[0]
    max_fitness = sort_fitness[-1]
    p = (cost-min_fitness)/(max_fitness-min_fitness + 10e-8)
    return p


# In[6]:



def fitness_sort(population):
    
    Fitness_sort = sorted(population, key=lambda x: x.score, reverse=False)
    best_individual = Fitness_sort[0]
    best_individual_fitness = best_individual.loss_D
    best_individual_data = best_individual.data
    return best_individual, best_individual_fitness, best_individual_data,Fitness_sort

def population_sort(population,offsprings):
    Fitness_sort = sorted(population + offsprings, key=lambda x: x.loss_D, reverse=False)  
    population_new = Fitness_sort[:opt.population_size]
    return population_new

def update_high_fitness(generator_individual,best):
    
    for i in range(opt.num_dist):
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
        generator_individual.update_data(z)
        best.update_data(z)
        loss = torch.dist(generator_individual.C, best.C.detach(), p=2)**2
        generator_individual.optimizer_G.zero_grad()
        loss.backward()
        generator_individual.optimizer_G.step()
    return generator_individual

def update_normal_fitness(g_individual,population):
    K = np.random.randint(0,len(population),2)
    individual1 = population[K[0]]
 
    for i in range(opt.num_dist):

        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
        individual1.update_data(z)

        g_individual.update_data(z)
        loss_norm = torch.dist(individual1.C.detach(), g_individual.C, p=2)**2
        g_individual.optimizer_G.zero_grad()
        loss_norm.backward()
        g_individual.optimizer_G.step()
    return g_individual

def update(g_individual,best_individual, population, z, Fitness, real_imgs):

    p=prob(g_individual.score,Fitness)

    if p < opt.fitness_up and p > opt.fitness_down:
      
        offspring = update_high_fitness(g_individual,best_individual)
    elif p > opt.fitness_up:
        offspring = update_normal_fitness(g_individual,population)
    else:
        offspring = g_individual
    offspring.update(z, real_imgs)

def choose_best_ID(best_new, best_old, Fitness, population):
    if best_old.score > best_new.score:
    
        return population.index(best_new)
    else:
        return population.index(best_old)

    


# In[7]:


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=0.8)


# In[8]:



adversarial_loss = torch.nn.BCELoss()

if cuda:
    adversarial_loss.cuda()
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import numpy as np

os.makedirs("data/mnist", exist_ok=True)


train_dataset = datasets.MNIST(
    "data/mnist",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
)

dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size
   
)
data_iter = itertools.cycle(dataloader)


# In[9]:


# os.makedirs("IM_fashion_v6_9_population6", exist_ok=True)
# os.makedirs("Model_fashion_v6_9_population6", exist_ok=True)
# os.makedirs("save_model_v6_9_population6", exist_ok=True)
# os.makedirs("CSV_fashion_v6_9_population6", exist_ok=True)
# os.makedirs("IM_savemodel_v6_9_population6", exist_ok=True)
# os.makedirs("IM_fashion_v6_9_population6/lrG%f_lrD%f_rate%d_fup%f_down%f_interval%d_potence%f_dist%d" % (opt.lr_G_or, opt.lr_D, opt.rate, opt.fitness_up, opt.fitness_down, opt.interval, opt.potence, opt.num_dist), exist_ok=True)
# os.makedirs("IM_savemodel_v6_9_population6/lrG%f_lrD%f_rate%d_fup%f_down%f_interval%d_potence%f_dist%d" % (opt.lr_G_or, opt.lr_D, opt.rate, opt.fitness_up, opt.fitness_down, opt.interval, opt.potence, opt.num_dist), exist_ok=True)


# In[10]:


os.makedirs("IMAGES", exist_ok=True)
os.makedirs("Model", exist_ok=True)
os.makedirs("IMAGES/lrG%f_lrD%f_rate%d_fup%f_down%f_interval%d_potence%f_dist%d" % (opt.lr_G_or, opt.lr_D, opt.rate, opt.fitness_up, opt.fitness_down, opt.interval, opt.potence, opt.num_dist), exist_ok=True)


# In[11]:


population = init_population(opt.population_size)
num_reject = [0]*opt.population_size
num_reject_liter = []
best_ID_new = random.randint(0, opt.population_size-1)
best_ID_var=0


for epoch in range(opt.max_iter):
    
    lr = opt.lr_G_or*np.exp(np.log(opt.lr_G_final/opt.lr_G_or)*epoch/(opt.max_iter-1))
    

    for step in range(opt.interval):
        x,_ = next(data_iter)
        xr = Variable(x.type(Tensor))
        z = Variable(Tensor(np.random.normal(0, 1, (xr.shape[0], opt.latent_dim))))
        population_or = copy_population(population, z, xr, lr)
        flag = False
    
        for g in population_or:
            g.update_G(z, xr)
        
        if epoch>0 and step == 0:
            flag = True
            Fitness = copy_fitness(population_or)

            best_ID_old = best_ID_new
            best_individual, best_individual_fitness, best_individual_imgs, Fitness_sort = fitness_sort(population_or)
            best = population_or[best_ID_old]
            best_ID_new = choose_best_ID(best_individual, best, Fitness, population_or)
            best = population_or[best_ID_new]
            for g_individual in population_or:
                if (g_individual is best) == False:
                    update(g_individual, best, population_or, z, Fitness, xr)
            if best_ID_old != best_ID_new:
                best_ID_var += 1
        
        for _ in range(opt.k):  
            
            x,_ = next(data_iter)
            xr = Variable(x.type(Tensor))  
            z = Variable(Tensor(np.random.normal(0, 1, (xr.shape[0], opt.latent_dim))))
            for d in population_or:
                d.update_D(z, xr)
       
        x,_ = next(data_iter)
        xr = Variable(x.type(Tensor))  
        z = Variable(Tensor(np.random.normal(0, 1, (xr.shape[0], opt.latent_dim))))
        for i in population_or:
            i.update(z, xr)

        train_step = epoch * opt.interval + step
        if train_step % opt.rate != 0 :
            for m in range(opt.population_size):
                if m == best_ID_new:
                    population[m] = population_or[m]
                elif population[m].loss_D > population_or[m].loss_D:
                    population[m] = population_or[m]
                else:
                    num_reject[m] += 1
        else:
            population = population_or

        for i in population:
            if flag:
                i.set_score()         
            else:
                i.caculate_score()
    
    test = population[random.randint(0, opt.population_size-1)]


    best = population[best_ID_new]
    print(
            "[Epoch %d/%d] [old loss: %f] [G loss: %f] [best score: %f] [best_ID: %d] [lr: %f]"
            % (epoch, opt.max_iter, best.loss_D.item(), best.loss_G.item(), best.score, best_ID_new, lr)
            )
    
    if epoch % opt.sample_interval == 0 or epoch == opt.max_iter-1:
        save_image(best.data[:100], "IMAGES/lrG%f_lrD%f_rate%d_fup%f_down%f_interval%d_potence%f_dist%d/best_%d.png" % (opt.lr_G_or, opt.lr_D, opt.rate, opt.fitness_up, opt.fitness_down, opt.interval, opt.potence, opt.num_dist,epoch), nrow=10, normalize=True)
        save_image(test.data[:100], "IMAGES/lrG%f_lrD%f_rate%d_fup%f_down%f_interval%d_potence%f_dist%d/random_%d.png" % (opt.lr_G_or, opt.lr_D, opt.rate, opt.fitness_up, opt.fitness_down, opt.interval, opt.potence, opt.num_dist, epoch), nrow=10, normalize=True)
        print(num_reject)
        num_reject_liter.append(copy.deepcopy(num_reject))
    
    if epoch % opt.save_model == 0:
        print(best_ID_new)
print(best_ID_var)


# In[12]:


torch.save(best.generator.state_dict(),"Model/GlrG%f_lrD%f_rate%d_fup%f_down%f_interval%d_potence%f_dist%d.pt" % (opt.lr_G_or, opt.lr_D, opt.rate, opt.fitness_up, opt.fitness_down, opt.interval, opt.potence, opt.num_dist))
torch.save(best.discriminator.state_dict(),"Model/DlrG%f_lrD%f_rate%d_fup%f_down%f_interval%d_potence%f_dist%d.pt" % (opt.lr_G_or, opt.lr_D, opt.rate, opt.fitness_up, opt.fitness_down, opt.interval, opt.potence, opt.num_dist))






