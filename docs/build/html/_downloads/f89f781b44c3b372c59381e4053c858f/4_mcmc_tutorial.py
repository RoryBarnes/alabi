"""
4_mcmc_tutorial
===============

This Python script was automatically generated from the Jupyter notebook
4_mcmc_tutorial.ipynb.

You can run this script directly or copy sections into your own code.
"""

# %% [markdown]
# ## MCMC sampling
# 
# In this example, we show how to sample our trained surrogate model using both the `emcee` and `dynesty` samplers.

# %%
%matplotlib inline

from functools import partial

import alabi



from matplotlib import rcParams

# rcParams['font.family'] = 'serif'

# rcParams['text.usetex'] = True

# %% [markdown]
# First, we need to define the same function that was used when creating the cached model, then we can reload our results from the previous tutorial:

# %%
# Redefine in this notebook so that pickle file can be loaded

from scipy.optimize import rosen



def rosenbrock_fn(x):

    return -rosen(x)/100.0



bounds = [(-5,5), (-5,5)]

param_names = ['x1', 'x2']

# %%
import os

if not os.path.exists("results/rosenbrock_2d/surrogate_model.pkl"):

    sm_tmp = alabi.SurrogateModel(lnlike_fn=rosenbrock_fn, bounds=bounds,

                                  param_names=param_names,

                                  savedir="results/rosenbrock_2d", cache=True)

    sm_tmp.init_samples(ntrain=100, ntest=1000, sampler="sobol")

    sm_tmp.init_gp(kernel="ExpSquaredKernel", fit_amp=True, fit_mean=True,

                   white_noise=-12, gp_opt_method="l-bfgs-b")

    sm_tmp.active_train(niter=100, algorithm="bape", gp_opt_freq=20)

    del sm_tmp

sm = alabi.load_model_cache("results/rosenbrock_2d")

# %% [markdown]
# ### Sampling with `emcee`
# 
# First, let's run emcee with default settings (uniform prior):

# %%
import alabi.utility as ut



# In this example we will set up a uniform prior within the parameter bounds

prior_fn = partial(ut.lnprior_uniform, bounds=sm.bounds)  # Use bounds from loaded model



sm.run_emcee(

    like_fn=sm.surrogate_log_likelihood,    # use like_fn=sm.true_log_likelihood to sample the true function

    prior_fn=prior_fn,                      # if None, defaults to uniform prior within bounds

    nwalkers=4,

    nsteps=int(2e4),

    burn=int(1e3),

    multi_proc=True,

    min_ess=1000

)

# %%
sm.plot(plots=["emcee_corner"]);

# %% [markdown]
# ### Sampling with `dynesty`
# 
# Now let's use the `dynesty` nested sampler for comparison with default settings (uniform prior):

# %%
# Set up uniform prior - dynesty requires separate prior transform function

prior_transform = partial(ut.prior_transform_uniform, bounds=sm.bounds)  # Use bounds from loaded model



dynesty_sampler_kwargs = {"bound": "single",

                          "nlive": 100,

                          "sample": "auto"}



dynesty_run_kwargs = {"wt_kwargs": {'pfrac': 1.0},      # set weights to 100% posterior, 0% evidence

                      "stop_kwargs": {'pfrac': 1.0},

                      "maxiter": int(2e4),

                      "dlogz_init": 0.5}

    

sm.run_dynesty(like_fn=sm.surrogate_log_likelihood,    # use like_fn=sm.true_log_likelihood to sample the true function

               prior_transform=prior_transform,        # if None, defaults to uniform prior within bounds

               multi_proc=False,                       # optional parallelization (oftentimes not worth it for dynesty)

               sampler_kwargs=dynesty_sampler_kwargs, 

               run_kwargs=dynesty_run_kwargs,

               min_ess=1000)

# %%
sm.plot(plots=["dynesty_corner"]);

# %%
sm.plot(plots=["dynesty_traceplot"]);

# %%
sm.plot(plots=["dynesty_runplot"]);
