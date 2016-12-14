from __future__ import print_function
from cl_osnn import snn
import pickle
import numpy as np
import matplotlib.pyplot as plt

def prg_plasticTest():
    s = snn('default')
    
    s.nneus = [2, 1]
    s.makeSyns = [(0, 0, 0)]
    s.connections = [(0,0,0,1,10.)]
    s.plasticity = True
    s.plasticity_params['WFactor'] = 40# 200000. #200000. #40
    s.norm_weights = False
    n1, n2, n11 = (0, 0), (0, 1), (1, 0)

    # test spike before train:
    s.spikeTm[4.0] = [n1]
    s.resetAt.append(8.0)
    # train:
    for st in np.linspace(10.,50.,5): 
        s.spikeTm[st] = [n1]
        s.spikeTm[st + 1.] = [n11]
        s.spikeTm[st + 1.2] = [n2]
        s.resetAt.append(st + 2.)
    # test:
    s.spikeTm[60.] = [n1]
    print(s.spikeTm)
    s.stoptime = 80.0
    
    s.nTrace = [n1, n2, n11]  # [(layer <0..> ,indx <0..>)]
    s.enable_Log = (True, False, True, True)
    
    s.init()
    s.run()
    if np.any(s.spikeRaster):
        n,itr = np.where(s.spikeRaster)
#        for nn,ii in np.where(s.spikeRaster.T[
        print("Last spikes on N%d at %f time." % (n[-1],itr[-1]*s.dt))
        np.save('pltimes', sorted(zip(n.tolist(),(itr*s.dt).tolist()), key = lambda x: x[1])) or print("saved to pltimes.")
    else:
        print("Nothing in raster")
    plt.show()
    
def prg_mesh():
    np.random.seed(10)
    s = snn('default')
    s.nneus = [100, 20] #200
    # s.delays = [0.] #,5.,7.]
    s.makeSyns = [(0,0,0)]
    s.randLayerConnections = [(0,0,.5,-10,5.),(0,1,.7,5.,1.)]
    s.randCurrent = (10,20)
    s.relaxAt = 50.
    
    s.enable_Log = (False, False, True, False)
    s.stoptime, s.dt = 100., .01
    s.extraplots = 1
    s.init()
    s.run()
    
    rwnd = 100
    axr = s.fig.add_subplot(s.subplotIndx+1, sharex=s.axes[0])    
        
    spall = np.zeros((0,2))
    for l in range(len(s.nneusUser)):
        start = sum(s.nneusUser[:l])        
        spikeRates = np.asarray([(i,np.sum(s.spikeRaster[start:(start+s.nneus[l]),i:(i+rwnd)])/s.nneusUser[l])
                                     for i in range(rwnd,s.iterations-rwnd)])
        axr.plot(spikeRates[:,0], spikeRates[:,1], label = 'Layer %d' % l)
        spall = np.vstack((spall,spikeRates))
    axr.legend()
    axr.set_xlim(0,s.iterations)
    axr.set_title('Average f.rates')
    np.save('sprates', spall)
    s.axes.append(axr)
    plt.show()
    
def prg_2mesh():
    np.random.seed(10)
    s = snn('default')
    s.delays = [] #[1.,3.,5.]
    s.nneus = [20,40,40,40]
    s.defWt = 40.
    #s.randLayerConnections = [(0,1,.7,10.,1.),(1,1,1.,10.,10.),(1,0,1.,20.,.1)]#]#,(2,0,.6,10.,5.),(0,0,.4,12.,2.)]
    s.randLayerConnections = [(0,1,.5,80.,1),(1,1,.3,50.,10.),(1,0,.4,20,5.),
                              (0,2,.5,80.,1),(2,2,.3,50.,10.),(2,0,.4,20,5.),
                              (0,3,.5,80.,1),(3,3,.3,50.,10.),(3,0,.4,20,5.),]
    s.randCurrent = [(1.,20.)]#,(30.,10.),(50.,10.),(70.,10.)]
    #s.relaxAt = 50.
        
    #s.showSyns = True
    s.plasticity = True
    s.plasticity_params['WFactor'] = 0.
    
    s.enable_Log = (False, False, True, False)
    s.nTrace = [(0,0),(0,1),(1,4)]
    s.stoptime, s.dt = 500., .1
    s.extraplots = 1
    s.init()
    s.run()
    
    rwnd = 100
    axr = s.fig.add_subplot(s.subplotIndx+1, sharex=s.axes[0])    
        
    spall = np.zeros((0,2))
    try:
        for l in range(len(s.nneusUser)):
            start = sum(s.nneusUser[:l])        
            spikeRates = np.asarray([(i,np.sum(s.spikeRaster[start:(start+s.nneus[l]),i:(i+rwnd)])/s.nneusUser[l])
                                         for i in range(rwnd,s.iterations-rwnd)])
            print(spikeRates.shape)
            axr.plot(spikeRates[:,0], spikeRates[:,1], label = 'Layer %d' % l)
            spall = np.vstack((spall,spikeRates))
        axr.legend()
        axr.set_xlim(0,s.iterations)
        axr.set_title('Average f.rates')
        np.save('sprates', spall)
        s.axes.append(axr)    
    except IndexError:
        print("Could not plot spike rates")
    #plt.savefig('result.png')
    
def prg_Big():
    np.random.seed(10)
    s = snn('default')
    s.nneus = [100] * 150
    #s.makeSyns = [(0,0,0)]
    s.randLayerConnections = [(sL,sL+1,100,1.,.5) for sL in range(len(s.nneus)-2)]
    s.randCurrent = (5.,1)
    
    s.enable_Log = (False, False, False, False)
    s.stoptime, s.dt = 200., .1
    s.extraplots = 1
    s.init()
    s.run()
    
def prg_2mesh2():
    #np.random.seed(10)
    s = snn('default')
    s.nneus = [2,1]
    s.defWt = 1.
    '''
    s.randLayerConnections = [(0,1,.2,1.*s.defWt,s.defWt/10.),
                              (1,0,.2,-0.1*s.defWt,s.defWt/10.)]
    '''                          
    s.connections = [(0,0,1,0,.1),(1,0,0,1,-.01)]
    #s.makeSyns = [(0,0,0)]
    s.currentAt = dict([((0,i),50.) for i in range(s.nneus[0])])

    s.enable_Log = (True, False, True, False)
    
    s.nTrace = [(0,0),(0,1),(1,0)]
    #s.nTrace = [(0,1)]
    s.stoptime, s.dt = 100., .01
    s.init()
    s.run()
    
    plt.show()
    
def prg_large():
    s = snn('default')
        
    s.nneus = [200,200,10]
    s.randSyns = 10 # no. of outward connections
    s.randCurrent = (100,20)
    
    # [(layer <0..> ,indx <0..>)]
    s.nTrace = zip([1]*10, [1])
    s.plasticity = False
    s.enable_Log = (False, False, True, False)
    s.stoptime = 20.0
    #s.K_DEFN['DISPSPK'] = str(0)
    s.norm_weights = True
    s.defWt = 10.
    s.dt = .01
    
    s.init()
    with open('last_syns','w') as f:
        pickle.dump(s.syns,f)
    
    s.run()
    
def prg_small():
    s = snn('default')
    
    s.nneus = [1, 2, 2]
    #s.delays = [1.,2.,3.]
    s.defWt = 100.
    # s.makeSyns = [(0, 0, 0)] #, (1, 0, 0)] #, (0, 0, 1), (0, 1, 1), (1, 1, 0)]
    # s.connections <- [(srcLayer, srcN, destLayer, destN, weight)]
    # s.connections = [(0,0,1,0),(1,0,1,1)] #,(1,1,1,0),(2,0,2,1),(2,1,2,0)]
    s.connections = [(0,0,1,0),(0,0,2,0),(1,0,1,1),(1,1,1,0, -25.)]
                     #(1,0,1,1),(1,1,1,0),
                     #(2,0,2,1),(2,1,2,0)]
    
    s.plasticity = False
    n0, n1, n10, n11, n20, n21 = (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)

    s.currentAt = {n0: 30.}
    s.stoptime = 10.
    s.dt = 0.01
    
    s.nTrace = [n0,n10,n11,n20] #,n20,n21]  # [(layer <0..> ,indx <0..>)]
    s.enable_Log = ("S", False, True, False)
    s.enable_RunStats = False
    
    s.init()    
    s.run()
    #with open('last_spkout','w') as f:        pickle.dump(s.spikeOuts,f)
    
def prg_small_delays():
    s = snn('default')
    
    s.nneus = [3, 1, 1]
    s.delays = [1.,1.5,3.]#, [1.,2.] #[1.,2.] #,2.,3.]
    s.defWt = 40.
    s.makeSyns = [(0, 0, 0),(0,1,0)]
    s.plasticity = False
    n0, n1 = (0, 0), (1, 0)
    
    #s.spikeTm[1.] = [n0]
    s.currentAt = {n0: 20.}
    s.stoptime = 20.
    s.dt = 0.01
    
    s.nTrace = [n0,n1,(2,0)]  # [(layer <0..> ,indx <0..>)]
    s.enable_Log = (True, False, True, False)
    s.init()
    s.run()
    #with open('last_spkout','w') as f:        pickle.dump(s.spikeOuts,f)
    
if __name__== '__main__':
    import os
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
    os.environ['PYOPENCL_CTX'] = '0'
    #os.environ['PYOPENCL_CTX'] = '0:0'

    prg_2mesh()
    plt.show()
