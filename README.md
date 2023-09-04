# MCMC-diffusion
A Metropolis-Hastings MCMC sampler accelerated via diffusion models. 

MCMC-diffusion contains 2 primary functions: `Algo_1.MH_Diffusion()`, which performs a Metropolis-Hastings MCMC chain combining a diffusion model proposal with a Gaussian proposal; and `Algo_1.PureMH()`, which performs a simple Metropolis-Hastings MCMC chain using only a Gaussian proposal. The user also has access to the `diffusion.DiffusionModel()` class, which can be used independently of any MCMC algorithm to generate diffusion samples resemble any target distribution.

The examples folder contains some files that will help you get started. The suggested order to run the examples is Gaussian, Himmelblau, EggBox, Rosenbrock, then the PDF physics example. The plotting files in the examples folder will help visualise the algorithm outputs. Copy the example files to the main directory first before running.

## Diffusion-accelerated MCMC algorithm
`Algo_1.MH_Diffusion(log_likelihood,dim,low_bound,high_bound,initial_samples,retrains,samples_per_retrain,**kwargs)`

Perform a Metropolis-Hastings MCMC chain combining a diffusion model proposal with a Gaussian proposal.

Function returns in order: an array of all samples in the chain, an array of only the accepted diffusion proposal samples, an array of only the accepted Gaussian proposal samples, an array of the acceptance rate of the diffusion proposal samples.

### Parameters:
`log_likelihood` - Your log-likelihood function to be sampled. Must be defined in terms of a 1D parameter array `x` and a number of dimensions `dim`. Some example log-likelihood functions are provided in the examples folder.

`dim` - Number of dimensions of the likelihood function.

`low_bound` - Array of lower bounds on each parameter.

`high_bound` - Array of upper bounds on each parameter.

`inital_samples` - Array of samples to be used as the starting point for the algorithm. For a likelihood function with widely separated modes, several samples from each mode must be supplied here to allow the diffusion model to jump between them. For functions with just one mode, it is not necessary to provide anything more than an initial sample as a starting point. See the examples folder for more details.

`retrains` - Number of times to retrain the diffusion model. More retrains will result in greater performance of the diffusion samples at the cost of increased runtime.

`samples_per_retrain` - Number of diffusion samples to generate before retraining the diffusion model. Total number of samples = "retrains" * "samples_per_retrain". 

### Keyword Arguments
`outdir` (default = `chain`) - Name of directory to save results.

`nsteps` (default = 20) - Number of noising steps in the forward/reverse diffusion process. If the diffusion model is failing to closely reproduce the target distribution (check "diffusion_check.pdf" in the relevant outdir), try increasing this value. If training is taking too long, try reducing this value.

`sigma` (default = 0.3) - Width of pure Metropolis-Hastings Gaussian proposal. If Metropolis acceptance efficiency is too low, try reducing this value. If algorithm is getting stuck in a local minimum or not exploring enough of the available parameter space, try increasing this value.

`diffusion_prob` (default = 0.5) - Probability of drawing a proposal sample from the diffusion model. Higher values are generally preferred for multi-modal functions where jumping between modes with the diffusion samples is required. 

`bins` (default = 20) - Number of bins used in 1D histograms of each parameter to calculate the Q proposal function weights. If diffusion acceptance rate remains very low over many samples, try increasing this value for greater resolution in the Q factor calculation, though doing so will increase retrain time.

`noise_width` (default = 0.05) - Width of noise after performing forward diffusion process. If diffusion model is failing to reproduce sharp modes of the target distribution (check "diffusion_check.pdf" in the relevant outdir), try reducing this value.

`beta_1` = How much noise to add in the first step of the forward diffusion process. We use a linear variance schedule, increasing from `beta_1` to `beta_2`. If diffusion model is failing to reproduce sharp modes of the target distribution (check "diffusion_check.pdf" in the relevant outdir), try increasing this value.

`beta_2` = How much noise to add at the last step of the forward diffusion process. We use a linear variance schedule, increasing from `beta_1` to `beta_2`. If diffusion model is failing to reproduce sharp modes of the target distribution (check "diffusion_check.pdf" in the relevant outdir), try increasing this value.

`plot_initial` (default = True) - Whether to plot "diffusion_check.pdf" in the outdir, set to false if not desired. If you get "ValueError: Contour levels must be increasing", set this to false.

## Simple Metropolis-Hastings algorithm
`Algo_1.PureMH(log_likelihood,nsamples,dim,low_bound,high_bound,**kwargs)`

Perform a Metropolis-Hastings MCMC chain using only a Gaussian proposal. Returns the samples generated from the overall chain.

### Parameters
`log_likelihood` - Your log-likelihood function to be sampled. Must be defined in terms of a 1D parameter array `x` and a number of dimensions `dim`. Some example log-likelihood functions are provided in the examples folder.

`nsamples` - Total number of MCMC samples to generate.

`dim` - Number of dimensions of the likelihood function.

`low_bound` - Array of lower bounds on each parameter.

`high_bound` - Array of upper bounds on each parameter.

### Keyword Arguments
`sigma` (default = 0.3) - Width of pure Metropolis-Hastings Gaussian proposal. If Metropolis acceptance efficiency is too low, try reducing this value. If algorithm is getting stuck in a local minimum or not exploring enough of the available parameter space, try increasing this value.

## Diffusion model class
`model = diffusion.DiffusionModel(training_samples,dim,nsteps,noise_width,initial_sample_size,desired_sample_size,var_guess)`

Initialise diffusion model. `model.fit()` generates unique diffusion samples approximately distributed according to a set of input data. Returns the diffusion samples along with the variance parameters.

### Parameters
`dim` - Number of dimensions of the input data.

`nsteps` - Number of noising steps in the forward/reverse diffusion process. If the diffusion model is failing to closely reproduce the target distribution, try increasing this value. If training is taking too long, try reducing this value.

`noise_width` - Width of noise after performing forward diffusion process. If diffusion model is failing to reproduce sharp modes of the target distribution, try reducing this value.

`initial_sample_size` - Length of input data array.

`desired_sample_size` - Number of diffusion samples to be generated.

`var_guess` - Starting point for variance parameters. Can be set to `np.ones(nsteps)` if unsure.

`beta_1` = How much noise to add in the first step of the forward diffusion process. We use a linear variance schedule, increasing from `beta_1` to `beta_2`. If diffusion model is failing to reproduce sharp modes of the target distribution, try increasing this value.

`beta_2` = How much noise to add at the last step of the forward diffusion process. We use a linear variance schedule, increasing from `beta_1` to `beta_2`. If diffusion model is failing to reproduce sharp modes of the target distribution, try increasing this value.
