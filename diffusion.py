import sys,os
import numpy as np
from scipy.optimize  import minimize
import time
from random import randrange

class DiffusionModel:
    def __init__(self,dim,nsteps,noise_width,initial_sample_size,
    desired_sample_size,training_samples,var_guess,beta_1,beta_2):
        self.dim = dim
        self.nsteps = nsteps
        self.initial_sample_size = initial_sample_size
        self.desired_sample_size = desired_sample_size
        self.training_samples = training_samples
        self.var_guess = var_guess
        self.noise_width = noise_width
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def forward_diffusion(self,x0,t,dim_iter,alpha_bar):
        eps = np.random.normal(loc = self.noise_means[dim_iter],
        scale = self.noise_stds[dim_iter], size = len(x0))
        mean = ((alpha_bar[t]) ** 0.5) * x0
        var = 1-alpha_bar[t]
        noise_added = mean + (var ** 0.5) * eps
        return noise_added

    def fit(self):

        # Set width of noised distribution after forward diffusion process to be (noise_width)*(std of training samples)
        initial_samples = []
        self.noise_stds = []
        self.noise_means = []
        for i in range(0,self.dim):
            initial_samples.append(self.training_samples[:,i])
            self.noise_stds.append(self.noise_width*np.std(self.training_samples[:,i]))
            self.noise_means.append(np.mean(self.training_samples[:,i]))

        # Perform forward diffusion process
        beta = np.linspace(self.beta_1, self.beta_2, self.nsteps)
        alpha = 1-beta
        alpha_bar = np.cumprod(alpha)
        X_diffusion = []
        noised = initial_samples
        X_diffusion.append(noised)
        for t in range(0,self.nsteps):
            noised_temp = []
            for i in range(len(noised)):
                noised_temp.append(self.forward_diffusion(noised[i],t,i,alpha_bar))
            X_diffusion.append(noised_temp)
            noised = noised_temp

        # Learn reverse diffusion process
        vars = []
        initial_noise = []
        for dim_iter in range(self.dim):
            initial_noise.append(np.random.normal(loc =
            self.noise_means[dim_iter], scale = self.noise_stds[dim_iter],
            size = self.initial_sample_size))

        def loss(phis):
            denoised = np.array(initial_noise)
            for p in range(0,self.nsteps):
                t = self.nsteps-p-1
                for dim_iter in range(self.dim):
                    diff_temp = X_diffusion[t][dim_iter]-X_diffusion[t+1][dim_iter]
                    denoised[dim_iter] = denoised[dim_iter] + diff_temp*phis[p]

            loss_temp = np.sum((np.array(X_diffusion[0]) - denoised)**2)
            return loss_temp

        guess = self.var_guess
        sol = minimize(loss, guess, method='Nelder-Mead', tol=1e-10)
        vars = sol.x

        # Set correct number of samples to generate
        remainder = self.desired_sample_size % self.initial_sample_size
        generations = int((self.desired_sample_size - remainder)/ self.initial_sample_size)
        if generations == 0:
            generations = 1

        final_samples_temp = []
        for dim_iter in range(self.dim):
            final_samples_temp.append([])

        # Perform reverse diffusion process
        for gen in range(0,generations):
            for dim_iter in range(self.dim):
                denoised = np.random.normal(loc = self.noise_means[dim_iter],
                scale = self.noise_stds[dim_iter], size = self.initial_sample_size)

                for p in range(0,self.nsteps):
                    t = self.nsteps-p-1
                    diff_temp = X_diffusion[t][dim_iter]-X_diffusion[t+1][dim_iter]
                    denoised = denoised + diff_temp*vars[p]


                final_samples_temp[dim_iter] = np.append(final_samples_temp[dim_iter],denoised)

        final_samples = []
        for j in range(len(final_samples_temp[0])):
            final_samples_ind = []
            for i in range(len(final_samples_temp)):
                final_samples_ind.append(final_samples_temp[i][j])
            final_samples.append(final_samples_ind)

        diffusion_samples = np.array(final_samples)

        return diffusion_samples,vars
