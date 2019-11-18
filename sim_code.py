#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Rinaldo Betkiewicz
"""

import numpy as np
from brian import define_default_clock, PoissonGroup, NeuronGroup, Connection, Network, SpikeMonitor, MultiStateMonitor, run, StringReset
from brian.library.IF import MembraneEquation, IonicCurrent
from brian.library.ionic_currents import leak_current, Current
from brian.library.synapses import exp_conductance

def runsim(neuron_model, 
           # sim params
           dt, simtime, prerun, monitors, recvars,
           # stimulation params
           fstim, r0_bg, r0_stim, stim_starts, stim_stops, stim_odors, stim_amps, stim_start_var,
           # network params
           beeid, N_glu, N_KC, ORNperGlu, PNperKC, PN_I0, LN_I0,
           # network weights
           wi, wORNLN, wORNPN, wPNKC,
           # default params
           V0min, inh_struct=None, Winh=None, timestep=500, report=None):

    np.random.seed() #needed for numpy/brian when runing parallel sims
    define_default_clock(dt=dt)    
    
    inh_on_off = 0 if (wi == 0) or (wi is None) or (wORNLN is None) else 1    
    
    
    
    #########################     NEURONGROUPS     #########################
    NG = dict()

    # ORN Input
    
    # For each glumerolus, random temporal response jitter can be added.
    # The jitter is added to the response onset. Maximum jitter is given by stim_start_var.
    # stim_start_jittered is a vector containing the jittered stim start tims
    
    # orn_activation returns a booolean vector of stim presence given time t
    
    # Total ORN rate: Baseline componenent equal for all units,
    # and individual activationa.
    
    jitter = np.random.uniform(0,stim_start_var,N_glu)
    
    stim_tun       = lambda odorN: fstim(N_glu=N_glu, odorN=odorN) * r0_stim
    orn_activation = lambda t: np.sum([
                     a*stim_tun(odorN=o)*np.logical_and(np.greater(t,prerun+stim_start+jitter), np.less(t,prerun+stim_stop))
                     for stim_start,stim_stop,o,a in zip(stim_starts, stim_stops, stim_odors, stim_amps)], 0)                         
    orn_rates      = lambda t: np.repeat(r0_bg + orn_activation(t),repeats = ORNperGlu)
    
    NG['ORN'] = PoissonGroup(ORNperGlu*N_glu, rates=orn_rates)
    NG['PN'] = NeuronGroup(N_glu, **neuron_model)
    NG['LN'] = NeuronGroup(N_glu*inh_on_off, **neuron_model)
    if 'KC' in monitors: NG['KC'] = NeuronGroup(N_KC, **neuron_model)

    #########################     CONNECTIONS       #########################
    c = dict()
    
    c['ORNPN'] = Connection(NG['ORN'],NG['PN'],'ge')
    
    for i in np.arange(N_glu): c['ORNPN'].connect_full(NG['ORN'].subgroup(ORNperGlu),NG['PN'][i],weight=wORNPN)

    if inh_on_off:
        print('-- inhibiting --',wi)
        
        c['ORNLN'] = Connection(NG['ORN'],NG['LN'],'ge')
        c['LNPN'] = Connection(NG['LN'],NG['PN'],'gi',weight=(wi*35)/N_glu)
        
        for i in np.arange(N_glu):
            c['ORNLN'].connect_full(NG['ORN'][ i*ORNperGlu : (i+1)*ORNperGlu ],
                                NG['LN'][i],
                                weight = wORNLN)
        if inh_struct: c['LNPN'].connect(NG['LN'],NG['PN'],Winh)
    
    if 'KC' in monitors:
        c['KC'] = Connection(NG['PN'],NG['KC'],'ge')
        c['KC'].connect_random(NG['PN'],NG['KC'],p=PNperKC/float(N_glu),weight=wPNKC,seed=beeid)
    
    #########################     INITIAL VALUES     #########################
    VT = neuron_model['threshold']
    
    NG['PN'].vm    = np.random.uniform(V0min,VT,size=len(NG['PN']))
    if inh_on_off:
        NG['LN'].vm= np.random.uniform(V0min,VT,size=len(NG['LN']))
    if 'KC' in monitors:
        NG['KC'].vm= np.random.uniform(V0min,VT,size=len(NG['KC']))
    
    net = Network(NG.values(), c.values())
    
    #### Compensation currents ###
    NG['PN'].I0 = PN_I0
    NG['LN'].I0 = LN_I0
    ##########################################################################

    #########################         PRE-RUN        #########################    
    net.run(prerun)
    #########################     MONITORS     #########################
    spmons = [SpikeMonitor(NG[mon], record=True) for mon in monitors]
    net.add(spmons)
    
    if len(recvars) > 0:
        mons = [MultiStateMonitor(NG[mon], vars=recvars, record=True, timestep=timestep) for mon in monitors]
        net.add(mons)
    else:
        mons = None
    #########################           RUN          #########################
    net = run(simtime, report=report)
    

    out_spikes = dict( (monitors[i],np.array(sm.spikes)) for i,sm in enumerate(spmons) )
    
    if mons is not None:
        out_mons = dict( (mon,dict((var,statemon.values) for var,statemon in m.iteritems())) for mon,m in zip(monitors,mons))
    else:
        out_mons = None

    #subtract the prerun from spike times, if there are any
    for spikes in out_spikes.itervalues():
        if len(spikes) != 0:
            spikes[:,1] -= prerun
    
    return out_spikes, out_mons


def model_IF(C,gL,EL,Ee,tau_syn_e,Ei,tau_syn_i,VT,Vr,tau_ref):
    '''
    returns a neuron model that can be passed to NeuronGroup
    '''
    ## ------- Model ------- ##
    neuron_model = dict()
    neuron_model['model'] = MembraneEquation(C=C) + leak_current(gl=gL,El=EL) 
    neuron_model['model'] += Current('I0: amp')
    neuron_model['model'] += exp_conductance('ge', Ee, tau_syn_e)
    neuron_model['model'] += exp_conductance('gi', Ei, tau_syn_i)

    neuron_model['threshold']  = VT
    neuron_model['reset']      = Vr

    # ´vm´ _is_ clamped to ´Vr´ during the refractory period.
    # This is default behaviour in Brain1.
    neuron_model['refractory'] = tau_ref

    return neuron_model

def model_saIF(C,gL,EL,Ee,tau_syn_e,Ei,tau_syn_i,VT,Vr,tau_ref,
               a,b,tauw,D):
    '''
    returns a neuron model that can be passed to NeuronGroup
    '''
    
    sigma = np.sqrt(2*D)*b
    
    ## ------- Model ------- ##
    neuron_model = dict()
    neuron_model['model'] = MembraneEquation(C=C) + leak_current(gl=gL,El=EL) 
    neuron_model['model'] += IonicCurrent('dw/dt=(a*(vm-EL)-w)/tauw + sigma*xi/tauw**.5 :amp', a=a, EL=EL, tauw=tauw, sigma=sigma)    
    neuron_model['model'] += Current('I0: amp')
    neuron_model['model'] += exp_conductance('ge', Ee, tau_syn_e)
    neuron_model['model'] += exp_conductance('gi', Ei, tau_syn_i)

    neuron_model['threshold'] = VT
    neuron_model['reset']     = StringReset('vm  = Vr; w  += b')
    
    # Note that vm is not clamped during the refractory period because more than
    # one variable is modified during reset. This is default behaviour in Brain1.
    neuron_model['refractory'] = tau_ref
    
    return neuron_model

def Sinoid(odorN=0, N_glu=35, N_a=14):
    if (N_glu-N_a)%2 != 0:
        N_a -= 1
    units = np.sin(np.linspace(0,np.pi,N_a))
    units = np.pad(units, (N_glu-N_a)//2, 'constant')
    return np.roll(units, odorN)