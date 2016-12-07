from __future__ import print_function
import pyopencl as cl
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
# from itertools import permutations as perm
from time import clock, sleep
from threading import Timer
from inspect import getouterframes, currentframe
from sys import stdout
import sys
import pdb
from collections import defaultdict
import pickle

def timeDisplay(interval):
    stime = clock()
    val = {'stp': 0, 'timer': []}
    
    def timedExec():
        '''Exec at intervals'''
        print('Elapsed %1d sec.' % (clock() - stime))
        val['timer'] = Timer(interval, timedExec)
        val['timer'].start()

    def reset():
        try:
            val['timer'].cancel()
        except AttributeError: pass

    def step(str):
        val['stp'] += 1
        ln = getouterframes(currentframe())[1]
        print('%1d> Line %1d: elapsed %1d sec. %s' %
              (val['stp'], ln[2], clock() - stime, str))
    return (step, timedExec, reset)

def offsets2(nneus, syns):
    nOffsets = np.array(
        [sum(nneus[0:i]) for i in range(len(nneus))] + [sum(nneus)], dtype=np.int32)
    (sPreNs, sWeights, sPreOffs, sOffsets, maxPreN) = ([], [], [0], [], 0)
    (post4mSyn, pre4mSyn, wt4mSyn) = (
        lambda x: x[1], lambda x: x[0], lambda x: x[2])
    import itertools
    for i, syn in enumerate(syns):
        # groupby postNs
        syn.sort(key=lambda x: x[1])
        gps = itertools.groupby(syn, key=post4mSyn)
        try:
            maxPn = nneus[i + 1]
        except IndexError:
            print('Check neuron count list. Ignoring further synapses.')
            break

        def get_gps():
            l = 0
            repeat = False
            while(l < maxPn):
                try:
                    z = gps.next()
                except StopIteration:
                    if (l < maxPn):
                        repeat = True
                while (z[0] > l) or repeat:
                    if (l > maxPn):
                        raise StopIteration
                    yield (l, [])
                    l += 1
                l = z[0] + 1
                yield z[0], list(z[1])
        sPreOffs, sPreNs, sWeights, maxPreN, postNCount =                 \
            reduce(lambda (offs, ns, wts, maxPreN, postCount), (postN, pres):
                   (lambda pres=list(pres):
                    (offs + [len(pres) + offs[-1]],
                     ns + map(pre4mSyn, pres),
                     wts + map(wt4mSyn, pres),
                     max([len(pres), maxPreN]),
                     postCount + 1))(),
                   get_gps(), (sPreOffs, sPreNs, sWeights, maxPreN, 0))
        #gps = list(itertools.groupby(syn, key = post4mSyn))
        #sPreOff = [len(list(gps[i][1])) for i in range(syn[-1][1]) if i in map(lambda x:x[0], gps)]
        # sPreOffs.extend(sPreOff)
        sOffsets.append(postNCount)
    #sPreOffs = np.array([sum(sPreOffs[0:i]) for i in range(len(sPreOffs))], dtype = np.int32)
    sPreOffs = np.array(sPreOffs, dtype=np.int32)
    sOffsets = np.array([sum(sOffsets[0:i])
                         for i in range(len(sOffsets))], dtype=np.int32)
    sPreNs = np.array(sPreNs, dtype=np.int32)
    sWeights = np.array(sWeights, dtype=np.float32)
    return (sPreNs, sPreOffs, sOffsets, sWeights, nOffsets, maxPreN)

class snn:
    'Class for creating and simulating SNNs with opencl'
    def __init__(self, prog='default', nplat=0, ndev=0):
        s = self
        
        #platform = cl.get_platforms()[nplat]
        #s.device = platform.get_devices()[ndev]
        s.context = cl.create_some_context()
        print("Context : %s" % s.context)
        s.device = s.context.devices[0]
        print("Device : %s" % s.device)
        s.queue = cl.CommandQueue(s.context)

        s.absToNeu = lambda x,(l,p):  (s.absToNeu((x-1),(l,p+1)) if (p < s.nneus[l]-1) else s.absToNeu((x-1),(l+1,0))) if (x > 0) else (l,p)
        s.spikeTm = {}  # {time: (layr,neu)}
        s.makeSyns = []  # [(preLayr, preN, postN)]
        s.makeSynsWeight = [] # [(preLayr, preN, postN, weight)]
        s.connections = []
        s.randLayerConnections = []
        s.delays = []
        s.delay_buff_len = 100
        
        s.randSyns = False
        s.showSyns = False
        s.currentAt = {}  # {(layr,neu): mA}
        s.randCurrent = False
        s.relaxAt = None
        s.plasticity_params = {'A2minus': 7.0E-3, 'A3minus': 2.3E-4, 'A2plus': 5.0E-10, 'A3plus': 6.2E-3, 'WFactor': 10.0, 'TAU_R1': 50.0, 'TAU_R2': 20.0, 'TAU_O1': 10.0, 'TAU_O2': 500.0}        
        #Pfister2006a, Table 4
        s.plasticity_params = {'A2minus': 3.0E-3, 'A3minus': 7.5E-9, 'A2plus': 4.6E-3, 'A3plus': 9.1E-3, 'WFactor': 1.0, 'TAU_R1': .168, 'TAU_R2': 5.75, 'TAU_O1': .337, 'TAU_O2': .47}        
        s.plasticity = False
        s.timed = True
        # potential, current, spiked, weight
        s.enable_Log = (True, False, False, False) # potn_like, xx, spike plot, weights

        s.multiple_runs = False
        s.plot_lines = []
        s.plot_int = False
        s.logBuffers = False
        s.enable_RunStats = True
        s.autoCropStart = True

        s.stoptime, s.dt = 20.0, 0.1
        s.outNeu = [-1]
        s.singleItr = False

        s.K_DEFN = {}
        s.resetAt = []
        s.defWt = 10.
        s.nTrace = []
        s.norm_weights = True
        s.printBufs = False
        s.extraplots = 0

        if prog == 'classify_test':
            # TODO: mark
            s.nneus = [4,15,3]#[4,5]
            s.spikeTm = {10: [(2, 1)]}
            s.spikeTm = {0.833:[(0,3)], 1.187: [(0,2)], 3.89: [(0,0)], 7.5: [(0,1)], 10.0: [(2,0)], 25.0: [(2,1)], 25.0: [(2,2)]}
            outtimes = [10.0, 25.0, 25.1]
            s.spikeTm =  dict([(t,[s.absToNeu(n,(0,0))]) for (n,t) in zip(range(sum(s.nneus),0,-1),reversed(outtimes))])
            intimes = [5, 7, 9, 3]
            s.spikeTm.update(dict([(t,[s.absToNeu(n,(0,0))]) for (n,t) in zip(range(sum(s.nneus)),intimes)]))
            s.nTrace = zip([1]*15,range(15))
            s.nTrace = [(2,0),(2,1),(2,2)]
            s.randSyns = 100 # no. of outward connections
            s.randCurrent = False
            s.plasticity = False
            s.enable_Log = (True, False, False, True)
            s.stoptime = 20.0
            s.K_DEFN['DISPSPK'] = str(0)
            s.K_DEFN['DEBOUT'] = str(0)
            s.defWt = 20.0
            s.singleItr = False
        
        elif prog == 'default':
            return
        else:
            raise RuntimeError('Unknown program !')

    def init(self):
        s = self
        
        s.nneusUser = list(s.nneus)
        # layer is [0,..] indx is [0,..]
        s.absPos = lambda (layer, indx): sum(s.nneus[0:layer]) + indx
        def getRelativePos(apos) :  # apos is 0 based
            ss = 0
            for l,n in enumerate(s.nneusUser + [0]):
                ss += n
                if ss > apos: return (l,apos-(ss-n))
        s.relPos = getRelativePos
        
        # for random connxns between layers
        # s.randLayerConnections <- [(srcLayer, destLayer, count, weightMean, weightSD)]
        for srcLayer, destLayer, count, weightMean, weightSD in s.randLayerConnections:
            allConn = np.random.rand(s.nneusUser[srcLayer],s.nneusUser[destLayer])
            count = (count > 1) and (float(count) / np.prod(allConn.shape)) or count
            allConn[:] = allConn[:] < count
            srcNArrs, destNArrs = np.where(allConn)
            weightArrs = np.random.randn(len(srcNArrs)) * weightSD + weightMean
            (p,) = np.where(weightArrs * weightMean < 0.)
            weightArrs[p] *= -1.
            connOut = np.empty((len(srcNArrs),), 
                               dtype = [('srcL','i4'),('srcN','i4'),('destL','i4'),('destN','i4'),('wt','f4')])            
            connOut['srcL'], connOut['destL'] = srcLayer, destLayer
            connOut['srcN'], connOut['destN'] = srcNArrs, destNArrs
            connOut['wt'] = weightArrs
            s.connections += connOut.tolist()
            np.save('randLayerConn_%d' % srcLayer, connOut)
            
        # for connxns to other layers or to same layer
        from itertools import groupby
        from collections import defaultdict
        # s.connections <- [(srcLayer, srcN, destLayer, destN[, weight])]
        s.postNOffsets = {}
        s.connections = [(x[0],x[1],x[2],x[3],(((len(x) == 5) and x[4]) or s.defWt)) for x in s.connections]
        s.connections = sorted(sorted(s.connections, key = lambda x: x[-2]), key = lambda x: x[-3])
        assert(all(map(lambda a: len(a) == 5,s.connections)))
        
        # if any connxn starts at the end layer, then put another layer on top
        if s.connections and (np.asarray(s.connections)[:,0] == (len(s.nneus)-1)).any():
            s.nneus += [1]
        
        # select source layer
        for layer in range(len(s.nneus)):
            # groupby destination
            for ((destL,destN),connx) in groupby(
                                            filter(lambda xx: xx[0] == layer, s.connections), 
                                            key = lambda xx: (xx[-3],xx[-2])):
                connx = list(connx)
                srcs = map(lambda x: x[1], connx)                
                wts = map(lambda x: x[-1], connx)                
                
                if (destL == (layer+1)):
                    s.makeSynsWeight += [(layer,src,destN,wt) for src, wt in zip(srcs,wts)]
                else:
                    s.makeSynsWeight += [(layer,src,s.nneus[layer+1],wt) for src,wt in zip(srcs,wts)]
                    s.postNOffsets[(layer+1,s.nneus[layer+1])] = (destL,destN)
                    s.nneus[layer+1] += 1
            
        s.postNOffsetsArray = np.zeros((sum(s.nneus),), dtype = np.int16)
        for ((srcL,srcN),(destL,destN)) in s.postNOffsets.iteritems():
            s.postNOffsetsArray[s.absPos((srcL,srcN))] = s.absPos((destL,destN)) - s.absPos((srcL,srcN))
        
        #print("Make syns: %s" % str(s.makeSynsWeight))
        #print("Post N Offsets array: %s" % str(s.postNOffsetsArray))
        
        s.timeOfItr = lambda itr: s.dt * itr

        if s.timed:  s.tstep, s.ttimed, s.reset = timeDisplay(10)
        else:        s.tstep, s.ttimed, s.reset = lambda x: (), lambda: (), lambda: ()

        if s.singleItr:
            s.stoptime = s.dt
        s.iterations = int(np.ceil(s.stoptime / s.dt))

        s.logNs = (any(s.enable_Log[:2]) and map(s.absPos, s.nTrace)) or []
        s.logNs = sorted(s.logNs)
        
        s.nlegend = map(lambda x: "N %d.%d" % x, s.nTrace)
        syns = []

        import random
        if s.randSyns:
            for i in range(1, len(s.nneus)):
                syns.append([(x, y, s.defWt)
                                for x in range(s.nneus[i - 1])
                                    for y in np.sort(random.sample(
                                                        range(s.nneus[i]),
                                                        min([s.randSyns,
                                                             s.nneus[i]])))])  # (src,des) all to any 1
        else:
            syns = [[]] * (len(s.nneus) - 1)

        # 1. connect (2+1)th neu from 1st layer to (1+1)th neu in 2nd
        
        for preLayer, src, dest in s.makeSyns:
            if not (src, dest) in syns[preLayer]:
                syns[preLayer].append((src, dest, s.defWt))
        for preLayer, src, dest, weight in s.makeSynsWeight:
            if not (src, dest) in syns[preLayer]:
                syns[preLayer].append((src, dest, weight))
        s.syns = syns
        np.save('synapses',syns)
        np.save('postNoff',s.postNOffsetsArray)
        
        if s.showSyns:
            for l,ls in enumerate(syns):
                print("Layer %d" % l)
                for cs in ls:
                    print("%d -> %d  (%s) " % (cs[0],cs[1],
                                               str(getRelativePos(s.absPos((l+1,cs[1])) + 
                                                                  s.postNOffsetsArray[s.absPos((l+1,cs[1]))]))))

        s.profiling = False

        s.resetAt = map(lambda x: int(np.floor(x / s.dt)), s.resetAt)
        print("Reset at iterations ", s.resetAt)

        s.initBuffers()
        s.build(True)

    def initBuffers(self):
        s = self
        s.tstep('Started initBuffers.')
        # TODO: if opencl supports -Inf (IEEE float), consider using that ?
        spks = -1 * np.inf * np.ones(sum(s.nneus), dtype= np.float32)
        spksOut = spks.copy()
        potn = np.ones_like(spks) * -65.0
        recv = np.zeros_like(potn, dtype=np.float32)
        current = np.zeros_like(potn)
        inscurrent = np.zeros_like(potn)
        
        if s.randCurrent:
            if 'tuple' in str(type(s.randCurrent)):
                current[0:s.nneus[0]] = s.randCurrent[0] + np.random.rand(s.nneus[0]) * s.randCurrent[1]
            if 'list' in str(type(s.randCurrent)):
                for l,(m,sd) in enumerate(s.randCurrent):
                    if l == 0: startInd = 0
                    else: startInd = sum(s.nneus[0:(l-1)])
                    stopInd = startInd + s.nneusUser[l]
                    current[startInd:stopInd] = m + np.random.rand(stopInd - startInd) * sd
                    
        for p in s.currentAt.keys():
            current[s.absPos(p)] = s.currentAt[p]
        #print("Currents: %s" % str(current))

        ctime = np.array([0], dtype=np.float32)
        s.logNs.sort()
        logIndices = np.zeros_like(potn, dtype=np.int32)
        for c, x in enumerate(s.logNs):
            logIndices[x] = c + 1
        logSz = np.array([len(s.logNs)], dtype=np.int32)
        logPot = np.zeros(s.iterations * len(s.logNs), dtype=np.float32)
        logcurrnt = np.copy(logPot)
        spkOffsets = [int(np.sum(
            map(lambda x: np.ceil(x / 32.0), s.nneus)[0:i])) for i in range(len(s.nneus))]
        spkOffsets += [spkOffsets[-1] + int(np.ceil(s.nneus[-1] / 32.0))]
        spkOffsets = np.array(spkOffsets, dtype=np.int32)
        logSpks = np.array(
            [0] * spkOffsets[-1] * s.iterations, dtype=np.uint32)
        logSpksOut = logSpks.copy()
        print(logSpks.shape)
        #pdb.set_trace()
        hasSpkd = np.zeros_like(potn, dtype=np.int8)

        (sPreNs, sPreOffs, sOffsets, sWeights, nOffsets, s.maxPreN) = offsets2(s.nneus, s.syns)
        #print(sWeights)
        #print(sPreNs, sPreOffs, sOffsets, sWeights, nOffsets, s.maxPreN)
                
        wd1 = s.device.max_work_item_sizes[0]
        #next2powr = lambda x :2 ** int(np.ceil(np.log2(x)))

        s.neuSize = ((wd1, (1 + max(s.nneus) / wd1), len(s.nneus)), None)
        s.neuToOff = lambda (l, n): (n % wd1, int(np.floor(float(n) / wd1)), l)

        s.propSize = None
        if len(s.nneus) > 1 :
            # no propagation for single layer
            # propSize: (preN, postN, layer) -> (preN, 1, 1)
            s.propSize = (
                (1 + max(s.nneus[0:-1]), max(s.nneus[1:]), len(s.nneus)), 
                (1 + max(s.nneus[0:-1]), 1, 1))
            s.propSize = ((wd1, max(s.nneus[1:]), len(s.nneus) - 1), (wd1, 1, 1))
            s.propSize = ((s.maxPreN, max(s.nneus[1:]), len(s.nneus)-1), (s.maxPreN, 1, 1))
        
        #s.propSize = ((1, max(s.nneus[1:]), len(s.nneus) - 1), (wd1, 1, 1))
        #s.propSize = ((max(s.nneus[1:])/256, 256, 1), (wd1,1,1))

        #propSize = ((1024, 1+max(nneus)/1024, len(nneus)-1), (64, 1+max(nneus)/64, 1))
        if ('AMD' in str(s.device)):
            s.spkSize = [(s.neuSize[0][0], s.neuSize[0][1], len(s.nneus)),None]
        else:
            s.spkSize = [(1 + s.neuSize[0][0]/8, 1 + s.neuSize[0][1]/8, len(s.nneus)),None]
        s.spkSize[1] = (s.spkSize[0][0],s.spkSize[0][1],1)
        
        s.dispWS = lambda: print('Neuron WG,WI: %s; \nSynapse WG,WI: %s; \nSpikeLog WG,WI: %s'
                               % (str(s.neuSize), str(s.propSize), str(s.spkSize)))
        s.dispWS()
        s.K_DEFN['MAX_WT_IND'] = reduce(np.multiply, sWeights.shape)
        s.wtSize = ((wd1,wd1,int(1+np.floor(s.K_DEFN['MAX_WT_IND']/(wd1*wd1)))), (1,1,1))
        
        print('MaxPreN = %d, Iterations = %d, nOffsets = %s' %
              (s.maxPreN, s.iterations, str(nOffsets)))
        #print('logIndices = %s' % str(logIndices))
        s.tstep('Allocating buffers on device.')

        s.cl_potn = cl.Buffer(
            s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=potn)
        s.cl_recov = cl.Buffer(
            s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=recv)
        s.cl_currnt = cl.Buffer(
            s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=current)
        s.cl_inscurrnt = cl.Buffer(
            s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=inscurrent)        
        s.cl_spks = cl.Buffer(
            s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=spks)
        s.cl_spksOut = cl.Buffer(
            s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=spksOut)
                
        if s.delays:
            spRegister = -1 * np.inf * np.ones((sum(s.nneus), s.delay_buff_len),
                                               dtype = np.float32)
            spInIndex  = np.zeros(s.nneus, dtype = np.int32)
            spOutIndex = np.zeros(s.nneus, dtype = np.int32)
            spDelaysBase = np.vstack([np.asarray(s.delays, dtype = np.float32).reshape(1,-1)] * sum(s.nneus)).reshape(-1,)

            s.cl_spRegister = cl.Buffer(
                s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=spRegister);
            s.cl_spInIndex  = cl.Buffer(
                s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=spInIndex);
            s.cl_spOutIndex  = cl.Buffer(
                s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=spOutIndex);
            s.cl_spDelaysBase  = cl.Buffer(
                s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=spDelaysBase);        
            print("Sp delays base: ")
            print(spDelaysBase)
        else:
            s.cl_spRegister,s.cl_spInIndex,s.cl_spOutIndex,s.cl_spDelaysBase = \
                None, None, None, None
            
        if s.printBufs:
            print('nOffsets: %s' % str(nOffsets))
        s.cl_nOffsets = cl.Buffer(
            s.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=nOffsets)
        
        cl.enqueue_read_buffer(s.queue, s.cl_nOffsets, nOffsets).wait()
        
        s.cl_ctime = cl.Buffer(
            s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=ctime)

        if s.plasticity:
            detectors_0 = np.zeros_like(potn)
            detectors_1 = np.zeros_like(potn)
            detectors_2 = np.zeros_like(potn)
            detectors_3 = np.zeros_like(potn)
            s.cl_d_r1 = cl.Buffer(
                s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=detectors_0)
            s.cl_d_r2 = cl.Buffer(
                s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=detectors_1)
            s.cl_d_o1 = cl.Buffer(
                s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=detectors_2)
            s.cl_d_o2 = cl.Buffer(
                s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=detectors_3)

        else:
            s.cl_d_r1, s.cl_d_r2, s.cl_d_o1, s.cl_d_o2 = None, None, None, None

        if s.printBufs:
            print('sPreNs: %s' % str(sPreNs))
            
        # if synapse is needed
        if s.propSize:
            s.cl_sPreNs = cl.Buffer(
                s.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=sPreNs)
            s.cl_sPreOffs = cl.Buffer(
                s.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=sPreOffs)
            s.cl_sOffsets = cl.Buffer(
                s.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=sOffsets)
            s.cl_postNOffsetsArray = cl.Buffer(
                s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=s.postNOffsetsArray)
            s.cl_sWeights = cl.Buffer(
                s.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=sWeights)
        s.sWeightInit = sWeights
        
        if s.printBufs:
            print('sPreOffs: %s' % str(sPreOffs))
        
        if s.printBufs:
            print('sOffsets: %s' % str(sOffsets))
        
        s.defWeights = sWeights.copy()
        if s.printBufs:
            print('sWeights: %s' % str(sWeights))
        
        s.cl_itr = cl.Buffer(
            s.context, cl.mem_flags.READ_WRITE, np.array([0], dtype=np.int32).nbytes)
        s.cl_hasSpiked = cl.Buffer(
            s.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=hasSpkd)

        if s.enable_Log[0] or s.enable_Log[1]:
            s.cl_logInd = cl.Buffer(
                s.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=logIndices)
            s.cl_logSz = cl.Buffer(
                s.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=logSz)
        if s.enable_Log[0]:
            s.cl_logPot = cl.Buffer(
                s.context, cl.mem_flags.WRITE_ONLY, logPot.nbytes)
        if s.enable_Log[1]:
            s.cl_logcur = cl.Buffer(
                s.context, cl.mem_flags.WRITE_ONLY, logcurrnt.nbytes)
        if s.enable_Log[3]:
            logWt = np.empty((sWeights.shape[0] * s.iterations,), dtype=np.float32)
            s.cl_logWt = cl.Buffer(
                s.context, cl.mem_flags.WRITE_ONLY, logWt.nbytes)
            s.cl_logWtSz = cl.Buffer(s.context, cl.mem_flags.READ_ONLY |
                                     cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([sWeights.shape[0]]))
            s.logWt = logWt
        if s.enable_Log[2]:
            s.cl_logSpk = cl.Buffer(
                s.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=logSpks)
            s.cl_logSpkOut = cl.Buffer(
                s.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=logSpksOut)
            s.cl_logSpkOffset = cl.Buffer(
                s.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=spkOffsets)
            s.cl_logSpkItrOffset = cl.Buffer(
                s.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([spkOffsets[-1]], dtype=np.int32))

        if s.logBuffers:
            # TODO: buf should only be numpy.array not cl.Buffer
            for fname, buf in [('cl_potn', potn), ('cl_recov', potn), ('cl_currnt', current), ('cl_spks', spks), ('cl_nOffsets', nOffsets), ('cl_ctime', ctime),('cl_sPreNs', sPreNs), ('cl_sPreOffs', sPreOffs), ('cl_sOffsets',sOffsets), ('cl_sWeights', sWeights),  ('cl_logInd', logIndices), ('cl_logSz', logSz), ('cl_logPot', logPot), ('cl_logSpk', s.cl_logSpk), ('cl_logSpkOffset', s.cl_logSpkOffset),('cl_logSpkItrOffset', s.cl_logSpkItrOffset), ('cl_hasSpiked', s.cl_hasSpiked)]:
                try:
                    np.savetxt('./logs/' + fname + '.csv', buf, fmt='%1d')
                except:
                    #print('Could not save %s. Shape of buffer: %s' % (fname, (buf.shape,)))
                    print('Could not save %s.' % fname)
                else:
                    print('Size of %s: %d' % (fname, buf.size))
        s.logPot = logPot
        s.logcurrnt = logcurrnt
        s.logSpks = logSpks
        s.logSpksOut = logSpksOut
        s.spkOffsets = spkOffsets
        s.tstep('Allocation complete.')
        s.wtsize = sWeights.shape[0]
        s.spks = spks
        s.hasSpkd = hasSpkd
        s.potn = potn
        #pdb.set_trace()

    def build(self, displayCode=True):
        s = self
        s.tstep('Building opencl kernels.')
        # load kernel code and add defines
        kDefs = {}
        kDefs['DT'] = "%fF" % s.dt
        kDefs['LEN_DELAYS'] = len(s.delays)
        kDefs['LEN_BUFFER_DELAYS'] = s.delay_buff_len
        kDefs['MAX_PRE_N'] = '%d' % s.maxPreN
        kDefs['MAX_LAYER'] = '%d' % len(s.nneus)
        kDefs['MAX_N'] = '%d' % max(s.nneus)
        kDefs['SILENT_FORCE'] = str(1)
        # kDefs['DEBOUT'] = str(1)        # Seg faults 
        kDefs['PRINT_AVAILABLE'] = str(1 * ('NVIDIA' not in str(s.device)))
        kDefs['USING_AMD'] = str(1 * ('AMD' in str(s.device)))
        kDefs['DISPDW'] = str(0)
        kDefs['DISPSPK'] = str(0)
        kDefs['DEF_W'] = '%fF' % (s.defWt*1.0)
        kDefs['DEBNETWTOUT'] = str(0)
        kDefs['TOTAL_N'] = str(sum(s.nneus))
        for k in s.plasticity_params:
            s.plasticity_params[k] = '%fF' % s.plasticity_params[k]
        kDefs.update(s.plasticity_params)
        kDefs.update(s.K_DEFN)

        defines = reduce(lambda t, kv: t + ("#define %s %s \n" % kv),
                         kDefs.iteritems(), '')

        kernelstr = defines + open('neuSyn1.cl', mode='r').read()
        
        if not int(kDefs['PRINT_AVAILABLE']):
            print("Replacing printfs.")
            regExPrintf = r"printf\(\"[^\"]+\"[^)]+\)"
            from re import sub
            kernelstr = sub(regExPrintf,"0",kernelstr);
        
        if displayCode:
            code_save_file = 'kernel_code_out.cl'
            with open(code_save_file,'w') as f:
                f.write('/*  Do not modify, will be overwritten.\nneuSyn1.cl is the one to modify.\n  */\n\n')
                f.write(kernelstr)       
            print("Kernels written to %s" % code_save_file)     
        try:
            build = cl.Program(s.context,kernelstr).build()
        except cl.RuntimeError:
            print('Failed to compile opencl code.')
            raise
        except:
            raise
        s.iterate_RS = build.iterate_RS
        s.transfer_spikes = build.transfer_spikes
        s.use_force_spike2 = True
        if s.use_force_spike2:
            s.force_spike = build.force_spike2
        else:
            s.force_spike = build.force_spike
        s.ping_spiked = build.ping_spiked
        s.propagate = build.synPost1_Pre0 #__test
        s.detect_upd = build.detectors
        s.weight_upd = build.triplet_stdp
        s.reset_states = build.reset_run_state
        s.reset_weights = build.reset_weights
        s.synWtNorm = build.synWtNorm
        
        s.logPotKernel = build.log_var
        s.logCurKernel = build.log_var
        s.logSpiked = build.spikedLog
        s.logSpiked2 = build.spikedLog2
        s.logWeights = build.log_all
        s.incTime = build.increment_time
        s.tstep('Kernels built.')

    def profiledRun(self):
        from cProfile import runctx
        self.profiling = True
        runctx('self.run()', globals(), locals())

    def resetWts(self, changeValue = None):
        #cl.enqueue_copy(self.queue, self.cl_sWeights, self.defWeights) 
        #print(self.wtSize)
        if not changeValue:
            self.reset_weights(self.queue, self.wtSize[0], self.wtSize[1], self.cl_sWeights, np.float32(s.defWt))
        else:
            self.reset_weights(self.queue, self.wtSize[0], self.wtSize[1], self.cl_sWeights, np.float32(changeValue))
        
    def run(self):
        s = self
        s.tstep('Running kernels.')

        #s.dispWS()
        s.ttimed()
        treset = s.reset
        stdout.flush()

        if s.enable_Log[0]:        
            if s.enable_Log[0] == "S":     cl_potn_like = s.cl_inscurrnt
            else: cl_potn_like = s.cl_potn

        #cl.enqueue_fill_buffer(s.queue, s.cl_ctime, np.array([0.0]))
        cl.enqueue_copy(s.queue, s.cl_ctime, np.array([0.0], dtype = np.float32))
        cl.enqueue_copy(s.queue, s.cl_itr, np.array([0], dtype = np.int32))
        #cl.enqueue_write_buffer(s.queue, s.cl_ctime, np.array([0.0]))
        s.resetAt.append(0)

        s.spikeAt = {}
        for itr in s.spikeTm.keys():
            s.spikeAt[int(np.ceil(itr / s.dt))] = s.spikeTm[itr]

        pzeros = np.zeros_like(s.potn, dtype = np.float32)
        if s.relaxAt: s.relaxAt /= s.dt
#        print(s.propSize)
        try:
            clk_start = clock()
            print(r'\\\\')
            for itr in range(s.iterations):
                #if itr > 0: break
                if s.relaxAt and itr == s.relaxAt:
                    print("Relax")
                    cl.enqueue_copy(s.queue, s.cl_currnt, pzeros)
                #'''
                if s.iterations > 10 and (itr % (s.iterations//10) == 0):
                    print("Completed %d of %d iterations." % (itr, s.iterations))
                    
                if itr in s.resetAt:
                    s.reset_states(s.queue, s.neuSize[0], s.neuSize[1],
                                       np.int32(s.plasticity),
                                       s.cl_potn, s.cl_recov,
                                       s.cl_d_r1, s.cl_d_r2, s.cl_d_o1, s.cl_d_o2,
                                       s.cl_spks, s.cl_hasSpiked, s.cl_nOffsets, s.cl_ctime, s.cl_itr)

                #'''
                #'''
                if s.use_force_spike2 and itr in s.spikeAt.keys():
                    for n in s.spikeAt[itr]:
                        wrkr = s.force_spike(s.queue, (1, 1, 1), None,
                                              s.cl_potn, s.cl_recov, s.cl_spks, s.cl_hasSpiked,
                                              s.cl_nOffsets, s.cl_ctime,
                                              global_offset=s.neuToOff(n))
                        
                wrkr = s.iterate_RS(s.queue, s.neuSize[0], s.neuSize[1],
                                     s.cl_potn, s.cl_recov, s.cl_currnt, s.cl_inscurrnt,
                                     s.cl_spks, s.cl_hasSpiked, s.cl_spksOut,
                                     s.cl_nOffsets, s.cl_ctime,
                                     s.cl_spRegister,s.cl_spInIndex,s.cl_spOutIndex,
                                     s.cl_spDelaysBase)
                
                if s.delays:
                    wrkr = s.transfer_spikes(s.queue, s.neuSize[0], s.neuSize[1],
                                             s.cl_spks, s.cl_hasSpiked,
                                             s.cl_nOffsets, s.cl_ctime,
                                             s.cl_spRegister, s.cl_spInIndex, s.cl_spOutIndex,
                                             s.cl_spDelaysBase)
                #'''
                #'''
                if not s.use_force_spike2 and itr in s.spikeAt.keys():
                    for n in s.spikeAt[itr]:
                        wrkr = s.force_spike(s.queue, (1, 1, 1), None,
                                              s.cl_potn, s.cl_recov, s.cl_spks, s.cl_hasSpiked,
                                              s.cl_nOffsets, s.cl_ctime,
                                              global_offset=s.neuToOff(n))
                #'''
                #'''
                wrkr = s.ping_spiked(s.queue, s.neuSize[0], s.neuSize[1],
                                     s.cl_spks, s.cl_hasSpiked,
                                     s.cl_nOffsets, s.cl_ctime)
                #'''
                #'''
                if s.enable_Log[0]:
                     wrkr = s.logPotKernel(s.queue, s.neuSize[0], s.neuSize[1],
                                           cl_potn_like, s.cl_nOffsets, s.cl_itr,
                                           s.cl_logInd, s.cl_logSz, s.cl_logPot)
                if s.enable_Log[1] and False:
                     wrkr = s.logCurKernel(s.queue, (max(s.nneus), len(s.nneus), 1), None,
                                           s.cl_currnt, s.cl_nOffsets, s.cl_itr,
                                           s.cl_logInd, s.cl_logSz, s.cl_logcur)
                if s.enable_Log[2]:
                     wrkr = s.logSpiked2(s.queue, s.spkSize[0], s.spkSize[1],
                                         s.cl_spks, s.cl_hasSpiked, s.cl_spksOut, s.cl_nOffsets,
                                         s.cl_logSpk, s.cl_logSpkOut,
                                         s.cl_logSpkOffset, s.cl_logSpkItrOffset,
                                         s.cl_itr, s.cl_ctime)                
                #'''
                #'''
                if s.propSize:
                    if s.enable_Log[3]:
                        wrkr = s.logWeights(s.queue, (s.wtsize,), None,
                                         s.cl_sWeights, s.cl_itr, s.cl_logWtSz, s.cl_logWt)
                    #'''
                    wrkr = s.propagate(s.queue, s.propSize[0], s.propSize[1],
                                        s.cl_sPreNs, s.cl_sPreOffs, s.cl_sOffsets, s.cl_nOffsets,
                                        s.cl_postNOffsetsArray,
                                        s.cl_sWeights, s.cl_inscurrnt,
                                        s.cl_spks, s.cl_hasSpiked, s.cl_ctime)
                    #'''
                    #'''
                
                    if s.plasticity:
                        s.weight_upd(s.queue, s.propSize[0], s.propSize[1],
                                     s.cl_sPreNs, s.cl_sPreOffs, s.cl_sOffsets, s.cl_nOffsets,
                                     s.cl_postNOffsetsArray,
                                     s.cl_sWeights, s.cl_d_r1, s.cl_d_r2, s.cl_d_o1, s.cl_d_o2,
                                     s.cl_hasSpiked, s.cl_spks, s.cl_ctime)
                        if s.norm_weights:
                            s.synWtNorm(s.queue, s.propSize[0], s.propSize[1],
                                        s.cl_sPreNs, s.cl_sPreOffs, s.cl_sOffsets, s.cl_nOffsets,
                                        s.cl_sWeights)
                        s.detect_upd(s.queue, s.neuSize[0], s.neuSize[1],
                                     s.cl_d_r1, s.cl_d_r2, s.cl_d_o1, s.cl_d_o2,
                                     s.cl_hasSpiked, s.cl_nOffsets, s.cl_spks, s.cl_ctime)
                #'''
                
                wrkr = s.incTime(s.queue, (1, 1, 1), None, s.cl_ctime, s.cl_itr)
                wrkr.wait()
                                
            print(r'////')
            if s.enable_RunStats:
                print("Neurons: " + str(s.nneus))
                if s.propSize:
                    print("Average synapses/layer, Net: " +
                            str(sum(map(len, s.syns)) / len(s.syns)) +
                            ", " + str(sum(map(len, s.syns))))
                print("Runtime: %0.3f, Speed: %0.2fX" %
                      (clock() - clk_start, s.stoptime / (clock() - clk_start)))
                print("Kernel Sizes: " + str((s.neuSize, s.propSize, s.spkSize)))
        except cl.LogicError, e:
            print(e)
            s.dispWS()
            raise
        except RuntimeError, e:
            print('Iteration: %d\nError!' % itr)
            #print(open(sys.argv[0], 'rb').readlines()[sys.exc_traceback.tb_lineno])
            print(e)
            d = s.device
            print("%s %s %s\n\r" % (d.vendor, d.name, d.version))
            print("Max workitem size: %d,%d,%d" % (
                d.max_work_item_sizes[0], d.max_work_item_sizes[1], d.max_work_item_sizes[2]))
            print("Max workgroup size: %d" % (d.max_work_group_size))
            raise
        else:
            print("Reading buffer")
            cl.enqueue_read_buffer(s.queue, s.cl_spks, s.spks)
            cl.enqueue_read_buffer(s.queue, s.cl_hasSpiked, s.hasSpkd).wait()
            if s.relaxAt:
                curr = np.zeros_like(s.potn)
                cl.enqueue_read_buffer(s.queue, s.cl_currnt, curr).wait()
                if np.sum(np.abs(curr)) == 0:
                    print("Zeroed current.")
            '''
            ibuf = np.empty(sum(s.nneus), dtype = np.int32)
            cl.enqueue_read_buffer(s.queue, s.cl_spInIndex, ibuf).wait()
            print(ibuf)
            print("out buffer")
            ibuf = np.empty(sum(s.nneus), dtype = np.int32)
            cl.enqueue_read_buffer(s.queue, s.cl_spOutIndex, ibuf).wait()
            print(ibuf)                        
            
            sWeights = np.zeros_like(s.sWeightInit)
            cl.enqueue_read_buffer(s.queue, s.cl_sWeights, sWeights).wait()
            print(sWeights)
            print(s.sWeightInit)
            
            d = np.zeros_like(s.potn)
            cl.enqueue_read_buffer(s.queue, s.cl_d_r1, d).wait()
            print(d)
            cl.enqueue_read_buffer(s.queue, s.cl_d_r2, d).wait()
            print(d)
            cl.enqueue_read_buffer(s.queue, s.cl_d_o2, d).wait()
            print(d)
            cl.enqueue_read_buffer(s.queue, s.cl_d_o1, d).wait()
            print(d)
            '''
            np.save('hasSpkd',s.hasSpkd)
            if np.any(s.hasSpkd):print("Some spiked")

            outTimes = [(s.hasSpkd[on] > 0 and s.spks[on]) or -1 for on in s.outNeu]

            axs = []
            if any(s.enable_Log):
                print("Start plotting")                
                fig = plt.figure()
                subplotIndx = 100 * (sum(map(lambda x: not (not x),s.enable_Log)) + s.extraplots) + 10
                              
                from matplotlib import ticker
                ticks = ticker.FuncFormatter(lambda x, pos: '%.1f' % (x * s.dt))
                
            if s.enable_Log[0]:
                print("Potential plot")
                cl.enqueue_read_buffer(s.queue, s.cl_logPot, s.logPot).wait()
                potns = np.reshape(
                    s.logPot, (s.iterations, len(s.logNs)), order='C')  # [1:]
                subplotIndx += 1
                ax1 = fig.add_subplot(subplotIndx)
                pot_lines = ax1.plot(potns)
                ax1.legend(s.nlegend)
                if s.enable_Log[0] == "P": ax1.set_ylim(-85,40)
                ax1.set_title('Potentials (mV)')
                ax1.xaxis.set_major_formatter(ticks)
                axs = [ax1]
            else:
                pot_lines = ()
            if s.enable_Log[1] and False: # removed
                print("Current plot")
                cl.enqueue_read_buffer(
                    s.queue, s.cl_logcur, s.logcurrnt).wait()
                potns = np.reshape(
                    s.logcurrnt, (s.iterations, len(s.logNs)), order='C')  # [1:]
                subplotIndx += 1
                if axs == []:
                    ax2 = fig.add_subplot(subplotIndx)
                else:
                    ax2 = fig.add_subplot(subplotIndx, sharex=axs[0])
                cur_lines = ax2.plot(potns)
                ax2.legend(s.nlegend)
                ax2.xaxis.set_major_formatter(ticks)
                axs.append(ax2)
            else:
                cur_lines = ()
            if s.enable_Log[3]:
                print("Weight plot")
                cl.enqueue_read_buffer(s.queue, s.cl_logWt, s.logWt).wait()
                potns = np.reshape(
                    s.logWt, (s.iterations, s.wtsize), order='C')  # [1:]
                subplotIndx += 1
                if axs == []:
                    ax4 = fig.add_subplot(subplotIndx)
                else:
                    ax4 = fig.add_subplot(subplotIndx, sharex=axs[0])
                wt_lines = ax4.plot(potns)
                # ax1.legend(s.nlegend)
                # ax1.set_ylim(-85,40)
                ax4.set_title('Weights')
                ax4.xaxis.set_major_formatter(ticks)
                axs.append(ax4)
            else:
                wt_lines = ()
            if s.enable_Log[2]:
                print("Spike plot")
                cl.enqueue_read_buffer(s.queue, s.cl_logSpk, s.logSpks).wait()
                #bt = lambda s: int(s.replace(' ',''), base=2)
                np.save('logSpks',s.logSpks) or print('Saved spk.')
                chunk = int(s.spkOffsets[-1])
                x = np.zeros((sum(s.nneus), s.iterations))
                s.spikeOuts = defaultdict(list)
                
                for itr in range(1, s.iterations):
                    b = s.logSpks[chunk * (itr - 1):(chunk * itr)]
                    a = map(lambda s: '{0:032b}'.format(s), b)
                    a = map(lambda (f, t): a[f:t], zip(s.spkOffsets, s.spkOffsets[1:]))
                    n = map(lambda (x, l):
                                map(int, reduce(lambda x1, x2: x1 + x2, x)[:l]),
                                zip(a, s.nneus))
                    #n = map(int,reduce(lambda x,y:x+y,a))
                    c = reduce(lambda x, y: x + y, n)
                    if any(c):
                        for layr,ns in enumerate(n):
                            for n,spkd in enumerate(ns):
                                if spkd:
                                    s.spikeOuts[(layr,n)] += [itr]                            
                    x[:, (itr - 1)] = c
                #map(lambda (itr,s): map(lambda (n,t): print("Itr %d, N %d" % (itr,n)) if t == 1 else None, enumerate(s)), enumerate(x))
                # remove the pseudo neurons
                xx = np.empty((sum(s.nneusUser), s.iterations), dtype = x.dtype)
                i = 0
                for layer,(netNeus,actNeus) in enumerate(zip(s.nneus,s.nneusUser)):
                    xx[i:(i+actNeus)] = x[sum(s.nneus[:layer]):(sum(s.nneus[:layer])+actNeus)]
                    i += actNeus
                np.save('log_spkPlot',xx) or print('Saved spike raster to log_spkPlot')
                s.spikeRaster = xx
                
                subplotIndx += 1
                if axs == []:
                    ax3 = fig.add_subplot(subplotIndx)
                else:
                    ax3 = fig.add_subplot(subplotIndx, sharex=axs[0])
                ax3.imshow(xx, aspect='auto', cmap = 'hot').set_interpolation('none')                
                ax3.xaxis.set_major_formatter(ticks)
                ax3.set_title('Plasticity: %s' % s.plasticity)
                axs.append(ax3)   
                try:
                    firstSpikeItr = np.where(xx > 0)[1][0]
                except IndexError:
                    print("No spikes :(")
                else:
                    ax3.set_xlim([firstSpikeItr - 10,s.iterations])
                
            if any(s.enable_Log):
                try:
                    plt.savefig('result.png')
                except RuntimeError, e:
                    print("File save error")
                    print(e)
                else:
                    print("Saved to result.png")
            s.axes = axs
            s.subplotIndx = subplotIndx
            s.fig = fig
        finally:
            treset()
            
        if any(s.enable_Log) and not s.profiling:
            if (s.multiple_runs):
                s.plot_lines += [(pot_lines,cur_lines,wt_lines)]
            else:
                a = {'xscale': np.max(axs[0].get_xlim()) / 2, 'xmaxx': np.max(axs[0].get_xlim())}

                def onZoom(event):
                    if event.button == 'up':
                        a['xscale'] *= 10
                    elif event.button == 'down':
                        a['xscale'] /= 10
                    onMove(event)

                def onMove(event):
                    xmin = np.max([0, event.x - a['xscale']])
                    xmax = np.min([a['xmaxx'], event.x + a['xscale']])
                    axs[0].set_xlim(xmin, xmax)
                    axs[0].figure.canvas.draw()
                    #print("x_lim %d %d\n" % (event.x-20.0,event.x+20.0))

                if s.plot_int:
                    for ax in axs:
                        ax.figure.canvas.mpl_connect('motion_notify_event', onMove)
                        ax.figure.canvas.mpl_connect('scroll_event', onZoom)
                plt.savefig('output.svg')                

        return outTimes

    def animAll(self):
        # TODO:
        import matplotlib.animation as animation
        plot_lines = self.plot_lines

        class AnimGraphs(animation.TimedAnimation):
            def __init__(self):
                print('Init called')
                self.fig = plt.figure()
                self.axs = [self.fig.add_subplot(311),self.fig.add_subplot(312),self.fig.add_subplot(313)]
                #self.figs = [plt.figure(),plt.figure(),plt.figure()]
                self.showLineNum(0)
                animation.TimedAnimation.__init__(self,self.fig,1,blit=False)
            def showLineNum(self,n):
                print('Show Line %d' % n)
                ln = plot_lines[n]
                map(lambda ax: ax.cla(), self.axs)
                map(lambda (ax,l): l and map(lambda lseg: ax.add_line(lseg),l), zip(self.axs,ln))
            def _draw_frame(self,framedata):
                print('_draw_frame')
                self.showLineNum(framedata)
            def new_frame_seq(self):
                print('_new_frame_seq')
                return iter(range(1,len(plot_lines)))
        AnimGraphs()
        plt.show()

    def drawNet(self):
        import networkx as nx
        G = nx.DiGraph()
        edg = reduce(lambda x, y: x + y,
                     map(lambda (ln, ls):
                         map(lambda (s, d, w):
                             ('N%d.%d' % (ln, s), 'N%d.%d' % (ln + 1, d), {'weight': w}), ls),
                         enumerate(self.syns)))

        #G.add_edges_from([('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])
        G.add_edges_from(edg)

        #val_map = {'A': 1.0,'D': 0.5714285714285714,'H': 0.0}

        #values = [val_map.get(node, 0.25) for node in G.nodes()]

        # Specify the edges you want here
        #red_edges = [('A', 'C'), ('E', 'C')]
        # edge_colours = ['black' if not edge in red_edges else 'red'
        #                for edge in G.edges()]
        #black_edges = [edge for edge in G.edges() if edge not in red_edges]

        # Need to create a layout when doing
        # separate calls to draw nodes and edges
        pos = nx.spring_layout(G)
        #nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color = values)
        #nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
        #nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
        #nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw(G)
        plt.savefig('output_graph.svg')
        plt.show()

        return pos

if __name__ == '__main__':
    s = snn()
    s.init()
    s.run()

    # s.drawNet()
    #for i in range(2):
    #    print('Runtime %d' % (i+1))
    #    print(s.run())
    
'''
TODO:
    issues:-
        1. clEnqueueNDRangeKernel failed: out of resources @ 336 on lab pc for nvidia card
'''


def offsets(nneus, syns):
    '''
    nneus = [9,5,2]
    syn1 = zip([0,1,5,6,8],[0]*5,[1]*5) + zip([2,3,5],[1]*3,[2]*3) + zip([1,4,7],[2]*3,[2]*3)
    syn2 = zip([1,2],[0]*2,[1]*2) + zip([2,4],[1]*2,[1]*2)
    syn3 = zip([0,1]*2,[0]*2,[0]*2) '''

    nOffsets = np.array(
        [sum(nneus[0:i]) for i in range(len(nneus))] + [sum(nneus)], dtype=np.int32)
    sPreNs = []
    sWeights = []
    sPreOffs = []
    sOffsets = []
    preNCounts = []

    for syn in syns:
        postNs = np.unique(map(lambda x: x[1], syn)).tolist()
        for postN in postNs:
            preNs = filter(lambda x: x[1] == postN, syn)
            sPreNs += map(lambda x: x[0], preNs)
            sWeights += map(lambda x: x[2], preNs)
            sPreOffs.append(len(preNs))
            preNCounts.append(len(preNs))
        sOffsets.append(len(postNs))

    sPreOffs = np.array([sum(sPreOffs[0:i])
                         for i in range(len(sPreOffs))], dtype=np.int32)
    sOffsets = np.array([sum(sOffsets[0:i])
                         for i in range(len(sOffsets))], dtype=np.int32)
    sPreNs = np.array(sPreNs, dtype=np.int32)
    sWeights = np.array(sWeights, dtype=np.float32)
    return (sPreNs, sPreOffs, sOffsets, sWeights, nOffsets, max(preNCounts))

def tests(what):
    if what == 'offset':
        nneus = [9, 5, 2]
        syn1 = zip([0, 1, 5, 6, 8], [0] * 5, [1] * 5) + zip([2, 3, 5],
                                                            [1] * 3, [2] * 3) + zip([1, 4, 7], [2] * 3, [2] * 3)
        syn2 = zip([1, 2], [0] * 2, [1] * 2) + zip([2, 4], [1] * 2, [1] * 2)
        syn3 = zip([0, 1] * 2, [0] * 2, [0] * 2)
        print(offsets2(nneus, [syn1, syn2, syn3]))
    elif what == 'timer':
        step, auto, reset = timeDisplay(3)
        auto()
        sleep(10)
        reset()
