#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Rinaldo Betkiewicz
"""

NTHREADS = 1 #Number of threads to use in simulation

import itertools as it
import cPickle as pickle
from multiprocessing import Pool
from datetime import datetime

from brian.stdunits import pF,nA,nS,mV,ms,Hz
from sim_code import runsim, model_IF, Sinoid



### PARAMETERS ###

# Model Parameters
model_params = dict(
    # Neuron Parameters
    C  = 289.5*pF,
    gL = 28.95*nS,
    EL = -70*mV,
    VT = -57*mV,
    Vr = -70*mV,
    tau_ref = 5*ms,
    
    # Synaptic Parameters
    Ee = 0*mV,
    Ei = -75*mV,
    tau_syn_e = 2 *ms,
    tau_syn_i = 10*ms
        )


# Sim Parameters
sim_params = dict(
    # Simulation
    dt       = 0.1*ms,
    simtime  = 3000*ms,
    prerun   = 2000*ms,       
    
    # Monitors
    monitors = ['PN', 'LN', 'KC'],
    recvars  = []
        )


# Stimulation Parameters
stim_start   = 1000*ms
stim_stop    = 2000*ms
odorN        = 0

stimulation_params = dict(
    fstim       = Sinoid,
    
    r0_bg          = 20*Hz,
    r0_stim        = 40*Hz, 
    
    stim_starts    = [stim_start],
    stim_stops     = [stim_stop],
    stim_odors     = it.repeat(odorN),
    stim_amps      = it.repeat(1),
    stim_start_var = 20*ms
        )


# Network Parameters
net_params = dict(
    beeid     = 1,
    
    # Dimensions
    N_glu     = 35,
    N_KC      = 1000,
    
    ORNperGlu = 284,
    PNperKC   = 12,
    
    PN_I0     = -0.38*nA,
    LN_I0     = -0.38*nA
        )


# Network Weights
w0 = 1*nS
We = lambda wi: wi*0.04236111+1.00115741*nS
Wi = list(it.chain.from_iterable(50*[w0*i] for i in range(10)))

net_weights = dict(
    wi     = None,
    wORNLN = w0,
    wORNPN = None,
    wPNKC  = 5*w0
        )


# Merge the parameters into one directionary
runsim_params = dict()
runsim_params.update(sim_params)
runsim_params.update(stimulation_params)
runsim_params.update(net_params)
runsim_params.update(net_weights)


# Helper function for Pool.map
def runsim_helper(wLNPN):
    '''
    Simulates the network for a given value of the inhibitory weight wi
    wLNPN : synaptic weight between the LN and the PN populations
    
    returns simulation data
    '''
    wORNPN = We(wLNPN)
    runsim_params.update(wi=wLNPN, wORNPN=wORNPN)
    return runsim(
            neuron_model=model_IF(**model_params),
            V0min = model_params['Vr'],
            **runsim_params)

def run_sims():
    ### Run simulations
    print('Starting simulations. Number of threads: {}'.format(NTHREADS))
    p = Pool(NTHREADS)
    sims = p.map(runsim_helper, Wi)
    p.close();p.join()
    
    print('Simulation finished. Saving data... ')
    
    
    ### Save simulation data
    spikemon = [sim[0] for sim in sims]
    
    prefix    = 'IF_'
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename  = prefix+timestamp
    
    print('Writing '+filename)
    with open(filename+'.pkl', "wb") as thefile:
        pickle.dump(spikemon, thefile, protocol=-1)
    
    if len(runsim_params['recvars']) > 0:
        statemon = [sim[1] for sim in sims]
        with open(filename+'statemon.pkl', "wb") as thefile:
            pickle.dump(statemon, thefile, protocol=-1)
            
    print('Done')

if __name__ == "__main__":
    run_sims()