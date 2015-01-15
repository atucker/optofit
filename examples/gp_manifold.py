"""
We're going to classify neurons.

Step One: Generate data

First pick out two groups of functionally-similar neurons, put a prior over them, and then synthesize data using a fixed observation model

Step Two: Classify Data

Initialization:
pick a random set of conductance values according to the prior

2a: Compute the likelihood of the observations conditioned on the conductance values
2b: Select a new set of conductance values to investigate, then repeat step 2a until you're happy with how much you know
2c: Use the Gaussian Process representing the likelihood ratios along the prior to compute the posterior distribution.
"""

import numpy as np
# Set the random seed for reproducibility
seed = np.random.randint(2**16)
print "Seed: ", seed
np.random.seed(seed)

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from optofit.cneuron.compartment import Compartment, SquidCompartment
from optofit.cneuron.channels import LeakChannel, NaChannel, KdrChannel
from optofit.cneuron.simulate import forward_euler

from hips.inference.particle_mcmc import *
from optofit.cinference.pmcmc import *

# Make a simple compartment
hypers = {
            'C'      : 1.0,
            'V0'     : -60.0,
            'g_leak' : 0.3,
            'E_leak' : -65.0,
            'g_na'   : 120.0,
            'E_na'   : 50.0,
            'g_kdr'  : 36.0,
            'E_kdr'  : -77.0
         }

def make_hypers(g_leak = .3, g_na = 120.0, g_kdr = 36.0):
    params = {
            'C'      : 1.0,
            'V0'     : -60.0,
            'E_leak' : -65.0,
            'E_na'   : 50.0,
            'E_kdr'  : -77.0
         }

    params['g_leak'] = g_leak
    params['g_na']   = g_na
    params['g_kdr']  = g_kdr

    return params

def sample_model(params, plot=False):
    # # Add a few channels
    # body = Compartment(name='body', hypers=hypers)
    # leak = LeakChannel(name='leak', hypers=hypers)
    # na = NaChannel(name='na', hypers=hypers)
    # kdr = KdrChannel(name='kdr', hypers=hypers)
    #
    # body.add_child(leak)
    # body.add_child(na)
    # body.add_child(kdr)
    # Initialize the model
    # body.initialize_offsets()

    squid_body = SquidCompartment(name='body', hypers=params)

    # Initialize the model
    D, I = squid_body.initialize_offsets()

    # Set the recording duration
    t_start = 0
    t_stop = 500.
    dt = 0.01
    t = np.arange(t_start, t_stop, dt)
    T = len(t)

    # Make input with an injected current from 500-600ms
    inpt = np.zeros((T, I))
    inpt[20/dt:400/dt,:] = 9.
    inpt += np.random.randn(T, I)

    # Set the initial distribution to be Gaussian around the steady state
    z0 = np.zeros(D)
    squid_body.steady_state(z0)
    init = GaussianInitialDistribution(z0, 0.1**2 * np.eye(D))

    # Set the proposal distribution using Hodgkin Huxley dynamics
    # TODO: Fix the hack which requires us to know the number of particles
    N = 1000
    sigmas = 0.0001*np.ones(D)
    # Set the voltage transition dynamics to be a bit noisier
    sigmas[squid_body.x_offset] = 0.025
    
    prop = HodgkinHuxleyProposal(T, N, D, squid_body,  sigmas, t, inpt)

    # Set the observation model to observe only the voltage
    etas = np.ones(1)
    observed_dims = np.array([squid_body.x_offset]).astype(np.int32)
    lkhd = PartialGaussianLikelihood(observed_dims, etas)

    # Initialize the latent state matrix to sample N=1 particle
    z = np.zeros((T,N,D))
    z[0,0,:] = init.sample()
    # Initialize the output matrix
    x = np.zeros((T,D))

    # Sample the latent state sequence
    for i in np.arange(0,T-1):
        # The interface kinda sucks. We have to tell it that
        # the first particle is always its ancestor
        prop.sample_next(z, i, np.zeros((N), dtype=np.int32))#np.array([0], dtype=np.int32))

    # Sample observations
    for i in np.arange(0,T):
        lkhd.sample(z,x,i,0)

    # Extract the first (and in this case only) particle
    z = z[:,0,:].copy(order='C')

    if(plot):
        # Plot the first particle trajectory
        plt.ion()
        fig = plt.figure()
        # fig.add_subplot(111, aspect='equal')
        plt.plot(t, z[:,observed_dims[0]], 'k')
        plt.plot(t, x[:,0],  'r')
        plt.show()
        plt.pause(0.01)

    return t, z, x, init, prop, lkhd

def integrate_likelihood(t, z_curr, x,
                         init, prop, lkhd,
                         N_particles=100):
    T,D = z_curr.shape
    T,O = x.shape
    # import pdb; pdb.set_trace()
    pf = ParticleGibbsAncestorSampling(T, N_particles, D)
    pf.initialize(init, prop, lkhd, x, z_curr)

    weights = np.zeros((T, N_particles), dtype=np.float)
    pf.update_likelihoods()
    weights[:, :] = pf.weights[:, :]
    weights = weights.copy(order='C')[T-1, :]
    ans = np.mean(np.exp(weights))
    del weights

    return ans

# Now run the pMCMC inference
def sample_z_given_x(t, z_curr, x,
                     init, prop, lkhd,
                     N_particles=100,
                     plot=False):

    T,D = z_curr.shape
    T,O = x.shape
    # import pdb; pdb.set_trace()
    pf = ParticleGibbsAncestorSampling(T, N_particles, D)
    pf.initialize(init, prop, lkhd, x, z_curr)

    S = 100
    z_smpls = np.zeros((S,T,D))
    l = plt.plot(t, z_smpls[0,:,0], 'b')
    for s in range(S):
        print "Iteration %d" % s
        # Reinitialize with the previous particle
        pf.initialize(init, prop, lkhd, x, z_smpls[s,:,:])
        z_smpls[s,:,:] = pf.sample()
        l[0].set_data(t, z_smpls[s,:,0])
        plt.pause(0.01)

    z_mean = z_smpls.mean(axis=0)
    z_std = z_smpls.std(axis=0)
    z_env = np.zeros((T*2,2))

    z_env[:,0] = np.concatenate((t, t[::-1]))
    z_env[:,1] = np.concatenate((z_mean[:,0] + z_std[:,0], z_mean[::-1,0] - z_std[::-1,0]))

    if plot:
        plt.gca().add_patch(Polygon(z_env, facecolor='b', alpha=0.25, edgecolor='none'))
        plt.plot(t, z_mean[:,0], 'b', lw=1)


        # Plot a few random samples
        # for s in range(10):
        #     si = np.random.randint(S)
        #     plt.plot(t, z_smpls[si,:,0], '-b', lw=0.5)

        plt.ioff()
        plt.show()

    return z_smpls


def observations(X):
    t, z, obs, init, prop, lkhd = sample_model(make_hypers(g_leak=X[2], g_na=X[0], g_kdr=X[1]))
    return obs

def likelihood(X, obs):
    t, z, _, init, prop, lkhd = sample_model(make_hypers(g_leak=X[2], g_na=X[0], g_kdr=X[1]))
    return integrate_likelihood(t, z, obs, init, prop, lkhd)

def gen_weights(g_leak = .3, g_na = 120.0, g_kdr = 36.0, repeats = 10):
    samples = []
    for i in range(repeats):
        t, z, _, init, prop, lkhd = sample_model(make_hypers(g_leak, g_na, g_kdr))
        samples.append(integrate_likelihood(t, z, x, init, prop, lkhd))
        print "After %d: %f" % (i, np.mean(samples))
    print np.mean(samples)
    print np.std(samples)
    return samples

def make_X(na_num, kdr_num, leak):
    return np.hstack((
            np.dstack(np.meshgrid(
                np.linspace(70.0, 150.0, na_num),
                np.linspace(10.0, 50.0, kdr_num))
            ).reshape(-1, 2),
            leak * np.ones((na_num*kdr_num,1))
    ))

import pickle

def save_start(X, Y, obs):
    pickle.dump((X, Y, obs), open("gp_manifold_2.pkl", 'w'))

def get_data():
    try:
       return pickle.load(open("gp_manifold_2.pkl", 'r'))
    except:
       _, _, obs, _, _, _ = sample_model(make_hypers(g_kdr=22), plot=True)
       X = make_X(5, 5, .3)
       ys = []
       print "built X, len: %d" % len(X)
       for i in range(len(X)):
           print i
           ys.append(likelihood(X[i], obs))
       Y = np.array(ys).reshape((25, 1))
       print "starting pickle"
       save_start(X, Y, obs)
       print "done pickle"
       return X, Y, obs

X, Y, obs = get_data()

import GPy
import scipy.special as spec
import scipy.stats   as scistats
import time

def acquisition_function(mean, var, cur_max):
    #return .5 * (1 + spec.erf(np.divide((mean - (cur_max)), var)))
    return scistats.norm.cdf(np.divide((mean - cur_max), var)) - scistats.norm.cdf(np.divide((mean - cur_max - .1), var))

def rand_X(xdim, ydim, leak):
    return np.hstack((
        np.dstack(np.meshgrid(
            (150.0 - 70.0) * np.random.random(xdim) + 70.0,
            (50.0 - 10.0) * np.random.random(ydim) + 10.0
        )).reshape(-1, 2),
        leak * np.ones((xdim*ydim,1))
    ))

def weighted_draw(values, weights):
    total = np.sum(weights)
    dist  = np.random.random() * total

    acc = 0
    for v, w, i in zip(values, weights, range(len(values))):
        acc += w
        if acc > dist:
            return v, i

    return v, i

def update_gaussian():
    kernel = GPy.kern.Matern52(2, ARD=True, variance=.1, lengthscale=[15, 3])

    X, Y, obs = get_data()

    while(True):
        print "Current Best: ", np.max(Y)
        Ymod = (Y - np.mean(Y)) * 10
        model = GPy.models.GPRegression(X[:, :2], Ymod, kernel)
        model.plot()
        plt.show()
        plt.pause(10)
        time.sleep(.5)
    
        xdim, ydim = 500, 500
        new = rand_X(xdim, ydim, .3)
        mean, var, _, _ = model.predict(new[:, :2])
    
        weights = acquisition_function(mean, var, np.max(Ymod))
        x_next, i = weighted_draw(new, weights)
    
        print "Investigating: ", x_next
        print "With mean %f, var %f, and weight %f" % (mean[i], var[i], weights[i])

        y = likelihood(x_next, obs)
        print "Value: ", y

        X = np.vstack((X, x_next))
        Y = np.vstack((Y, y))

        print "Saving progress..."
        save_start(X, Y, obs)
        
