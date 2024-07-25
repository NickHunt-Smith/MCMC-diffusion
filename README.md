# MCMC-diffusion (Optimized)
A Metropolis-Hastings MCMC sampler accelerated via diffusion models. Please cite https://arxiv.org/abs/2309.01454 if you use this code in your research.

Optimization ideas from Ewan Cameron at Telethon Kids Insitute:
"Replace line 158 with:
*a copy-paste of current lines 103-131*
Then
QMH = scipy.stats.multivariate_normal(theta_prime-theta,0,self.self.sigma)
Q_ratio = (self.diffusion_prob*Q+(1-self.diffusion_prob)*QMH)/(self.diffusion_prob*Q_prime+(1-self.diffusion_prob)*QMH)

a = min(1, self.log_likelihood(theta_prime,self.dim)self.log_likelihood(theta,self.dim)*Q_ratio)
 
Remove lines 116-117
 
Replace line 135 with:
 
QMH = scipy.stats.multivariate_normal(theta_prime-theta,0,self.self.sigma)
Q_ratio = (self.diffusion_prob*Q+(1-self.diffusion_prob)*QMH)/(self.diffusion_prob*Q_prime+(1-self.diffusion_prob)*QMH)"

Original run_MCMC_Gaussian:
Pure MH samples = 9999 Pure MH Acceptance efficiency =  0.2042
Diffusion MCMC Chain Started
Number of retrains = 100/100
Diffusion acceptance efficiency = 0.5
Metropolis acceptance efficiency = 0.25
Previous Retrain Time = 11.0 seconds 
Total Retrain Time = 735.0 seconds 

Modified:
Pure MH samples = 9999 Pure MH Acceptance efficiency =  0.1987
Diffusion MCMC Chain Started
Number of retrains = 100/100
Diffusion acceptance efficiency = 0.875
Metropolis acceptance efficiency = 0.38095238095238093
Previous Retrain Time = 12.0 seconds 
Total Retrain Time = 750.0 seconds 
