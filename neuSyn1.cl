#if PRINT_AVAILABLE == 1
    #pragma OPENCL EXTENSION cl_amd_printf : enable    
#endif

// should already be defined
// these are stubs
#ifndef DT
#define DT 0.01F
// MAX_PRE_N should be < biggest local_dim(1)*local_dim(0)    
//#define MAX_PRE_N 10+1
//#define MAX_N 100
#define DEBOUT 1/0
#define DEBNETWTOUT 0
#define DISPDW 0
#define DISPSPK 0
#define MAX_WT_IND 1/0
#define LEN_DELAYS 0
#endif 

#ifndef NODEB
#define DEBPRINT  { if (*ctime < 100.0) printf("\n%.2f: %s", *ctime, __func__); }
#define DEBPRINT2 printf("\nXXXX: %s", __func__); 
#define DEBPRINT_EXIT  { if (*ctime < 100.0) printf("\n%.2f: %s (exit)", *ctime, __func__); }
#endif
#ifdef NODEB
#define DEBPRINT 0;
#define DEBPRINT2 0;
#define DEBPRINT_EXIT 0;
#endif

inline float vdash(float v, float rec, float cur) {
    return 0.04F*v*v+5*v+140-rec+cur;
}

inline float udash_RS(float v, float rec) {
    return 0.02F*(0.2F*v-rec);
}

inline float udash_IB(float v, float rec) {
    return 0.02F*(0.2F*v-rec);
}

inline float2 x_RK_RS(float x, float u, float cur, float dt) {
    float k1 = vdash(x,u,cur);
    float k1_rec = udash_RS(x,u);
    float k2 = vdash(x+0.5F*dt*k1, u+0.5F*dt*k1_rec,cur);
    float k2_rec = udash_RS(x+0.5F*dt*k1, u+0.5F*dt*k1_rec);
    float k3 = vdash(x+0.5F*dt*k2, u+0.5F*dt*k2_rec,cur);
    float k3_rec = udash_RS(x+0.5F*dt*k2, u+0.5F*dt*k2_rec);
    float k4 = vdash(x+dt*k3, u+dt*k3_rec,cur);
    float k4_rec = udash_RS(x+dt*k3, u+dt*k3_rec);
    float2 xx;
    xx.x = x + dt*(k1+2*k2+2*k3+k4)/6;
    xx.y = u + dt*(k1_rec+2*k2_rec+2*k3_rec+k4_rec)/6;
    return xx;
}

#define _gf  __global float
#define _gb  __global bool
#define _gi  __global int
#define _gi2 __global int2
#define _gc  __global uchar
#define _gs  __global short

__kernel void
spikedLog3(_gf *spkd, _gc *hasSpkd, _gf *spksOut, _gi *offsets, 
           _gi *logSpk, _gi *logSpkOut,
           _gi *logLayrOffset, _gi *logItrOffset, _gi *itr, _gf *ctime)
{
    /* WI[0], WI[1]: neuron number ;       WI[2]: layer number 
       WG[0] = 32; WG[1] = 1; WG[2] = 1;
       Each work item should correspond to a neuron and each work group to group of 32 neurons
    */

    size_t x = get_global_id(0) + get_global_id(1) + get_global_id(2);    
#ifdef DEBOUT  
    if (x == 0) DEBPRINT    
#endif
    int layr = get_global_id(2);
    
    size_t nNum = get_global_size(0)*get_global_id(1)+get_global_id(0);
    unsigned int absPos = offsets[layr] + nNum;

    __local unsigned int spikedInt, spkoutInt;
    spikedInt = 0; 
    spkoutInt = 0;

    bool isSpkd = 0, isSpkOut = 0;

    if (absPos < offsets[layr+1]) {
        if (hasSpkd[absPos] > 0) {
            isSpkd = fabs(spkd[absPos] - *ctime) < (0.5F*DT);
            isSpkOut = fabs(spksOut[absPos] - *ctime) < (0.5F*DT);		   
        }
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);

    unsigned int sd = (isSpkd) << (31 - (nNum % 32));
    unsigned int so = (isSpkOut) << (31 - (nNum % 32));

    atomic_or(&spikedInt, sd);
    atomic_or(&spkoutInt, so);

    unsigned int globalIndex = logLayrOffset[layr] + (*logItrOffset)*(*itr);  

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((nNum % 32) == 0) {
        if (absPos < offsets[layr+1]) {
//            logSpk[globalIndex + nNum] = 0xFF; //spikedInt;
//            logSpkOut[globalIndex + nNum] = 0xFF; //spkoutInt;
//            atomic_max(&logSpk[globalIndex+nNum],get_global_id(0));
            logSpk[globalIndex+ (nNum >> 5)] = spikedInt;
            logSpkOut[globalIndex+ (nNum >> 5)] = spkoutInt;
//            logSpkOut[globalIndex+1] = 1;
        }
    }
}

/*
TODO:       get_global_id(2) (8 bits):      <neuron indx. 0> [8-k bits] | <layr indx> [k bits]
*/

__kernel void
spikedLog2(_gf *spkd, _gc *hasSpkd, _gf *spksOut, _gi *offsets, 
           _gi *logSpk, _gi *logSpkOut,
           _gi *logLayrOffset, _gi *logItrOffset, _gi *itr, _gf *ctime)
{
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);    
#ifdef DEBOUT  
    if (x == 0) DEBPRINT    
#endif
    int layr = get_global_id(2);
    
#if USING_AMD == 0   
    unsigned int nind = (get_global_size(0)*get_global_id(1)+get_global_id(0)) * 8;  /* nind restarts for every layer */
    unsigned int absPos = offsets[layr] + nind;
        
    unsigned int bI = (nind/32) + logLayrOffset[layr] + (*logItrOffset)*(*itr);    
    __global unsigned char* logSpkByte = &logSpk[bI];
    __global unsigned char* logSpkOutByte = &logSpkOut[bI];
    int b = 0, c = 0;
    unsigned char zz = 0, zy = 0;
    unsigned int maxN;
    const unsigned int eight = 8;

    maxN = max(eight, offsets[layr+1]-absPos);
	
    for (int i = 0; i < maxN && i >= 0; i++) {
        if (hasSpkd[absPos+i] > 0) {
            b = fabs(spkd[absPos+i] - *ctime) < (0.5F*DT);		
            zz |= b << i;
            c = fabs(spksOut[absPos+i] - *ctime) < (0.5F*DT);		
            zy |= b << i;
        }        
    }	
    logSpkByte[(nind >> 3) & 3] = zz;
    logSpkOutByte[(nind >> 3) & 3] = zy;
    
#else /* USING_AMD */
    int nind = get_global_size(0)*get_global_id(1)+get_global_id(0);  /* nind restarts for every layer */
    int absPos = offsets[layr] + nind;
    
    unsigned int localPos = get_local_id(0);
    // get_local_size(0) should be 256
    __local bool locByte[256];
    __local bool locOutByte[256];
    
    bool b = hasSpkd[absPos] && (fabs(spkd[absPos] - *ctime) < (0.5F*DT));
    bool c = hasSpkd[absPos] && (fabs(spksOut[absPos] - *ctime) < (0.5F*DT));
    
    locByte[localPos] = b;
    locOutByte[localPos] = c;
    
    //if (b) printf("Spiked %d @ %f; %d!\n",nind, *ctime, localPos);    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    unsigned int aa = 0, bb = 0;
    unsigned int bI = floor((float)nind/32) + logLayrOffset[layr] + (*logItrOffset)*(*itr);    

    if ((localPos % 32 == 0) && (absPos < offsets[layr+1])) {
        //if (*ctime < 3.0)  printf("%d, %d\n",bI,*itr);
        //printf("%d ", localPos);
        int i = localPos;
        int xx = floor((float)i/32);
        do {
            aa = aa | (locByte[i] << (31 - (i - 32*xx)));
            bb = bb | (locOutByte[i] << (31 - (i - 32*xx)));
            i++;
        } while (i % 32 > 0);
        logSpk[bI] = aa;
        logSpkOut[bI] = bb;
    }
#endif
}

#ifdef PRINT_AVAILABLE
 
__kernel void
ping_spiked(_gf *spkd, _gc *hasSpkd, _gi *offsets, _gf *ctime) {    
#if DISPSPK
    int layr = get_global_id(2);
    int nind = get_global_size(0)*get_global_id(1)+get_global_id(0);
    int absPos = offsets[layr] + nind;

    if (absPos < offsets[layr+1]) 
        if ((spkd[absPos] == *ctime) && (hasSpkd[absPos])) 
            printf("^%d@%.3f\n", absPos, *ctime);    
#endif
}

#endif

/* spRegister:
        [absPos*lenBufferDelays] : [t0,t1,t2,t3,t4,t5,t6,X,...]
                                  |                \-- in next index
                                  \--  out next index
 * delays:      [absPos*lenDelays]
 * spInIndex:
        [absPos] :  in next index
 
 * spOutIndex:
        [absPos] : out next index
*/

__kernel void
transfer_spikes(_gf *spkd, _gc *hasSpkd,            
           _gi *offsets, _gf *ctime,
           _gf *spRegister, _gi *spInIndex, _gi *spOutIndex,
           _gf *delaysBase) {
    
    int layr = get_global_id(2);
    int nind = get_global_size(0)*get_global_id(1)+get_global_id(0);
    int absPos = offsets[layr] + nind;
    
    __global float* spRegBaseAbs = &spRegister[absPos * LEN_DELAYS];
    __global int* spRegOutIndex = spOutIndex + absPos;
    __global int* spRegInIndex  = spInIndex + absPos;
    
    if (absPos < offsets[layr+1]) {
        int outind = *spRegOutIndex % LEN_BUFFER_DELAYS;
        float otime = spRegBaseAbs[outind];
        if (fabs(otime - *ctime) < DT) {
            spkd[absPos] = *ctime;
            hasSpkd[absPos] = 1;
		}
		if (otime < *ctime) {
            outind = (outind + 1) % LEN_BUFFER_DELAYS;
            if (outind != (*spRegInIndex % LEN_BUFFER_DELAYS)) {
                // if next output pointer is not same as input pointer (buffer not filled)
                *spRegOutIndex = outind;
            } else {
                spRegBaseAbs[outind] = 0.;
            }
        }
    }
}

__kernel void
iterate_RS(_gf *pot, _gf *rec, _gf *cur, _gf *inscur,
           _gf *spkd, _gc *hasSpkd, _gf *spksOut,           
           _gi *offsets, _gf *ctime,
           _gf *spRegister, _gi *spInIndex, _gi *spOutIndex,
           _gf *delaysBase) {    
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
#ifdef DEBOUT         
    if (x == 0) DEBPRINT
#endif    
    int layr = get_global_id(2);
    int nind = get_global_size(0)*get_global_id(1)+get_global_id(0);
    int absPos = offsets[layr] + nind;
    float p,r,c;
    p = pot[absPos];
    r = rec[absPos];
    float ii = inscur[absPos], tk;
    c = cur[absPos];    
#if LEN_DELAYS > 0    
    __global float* spRegBaseAbs = &spRegister[absPos * LEN_BUFFER_DELAYS];
    __global float* delays = &delaysBase[absPos * LEN_DELAYS];
    __global int* spRegInIndex = spInIndex + absPos;
    __global int* spRegOutIndex = spOutIndex + absPos;
    int spInInd, spInInd2, spOutInd, i;
#endif
    // TODO: really need to iterate layer-wise ?
    if (absPos < offsets[layr+1]) {
        //if (absPos == 0) printf("@%d %f %f\n",absPos,c,ii);
        inscur[absPos] = 0;
        
        float2 uv = x_RK_RS(p,r, c + ii, DT);
        if (p >= 30) {
            //printf("N %d\n", absPos);
                spksOut[absPos] = *ctime;
#if LEN_DELAYS == 0            
                spkd[absPos] = *ctime;
                hasSpkd[absPos] = 1;            
#else
                spInInd  = *spRegInIndex % LEN_BUFFER_DELAYS;
                spOutInd = *spRegOutIndex % LEN_BUFFER_DELAYS;
            
                if ((spOutInd == spInInd - 1) &&
                    (spRegBaseAbs[spOutInd] < 0.5f*DT)) {
                    // buffer is empty, set out index to input position
                    *spRegOutIndex = (*spRegOutIndex) + 1;
                }
                
                for (i = 0; i < LEN_DELAYS; i++) {
                    tk = (*ctime) + delays[i];
                    spRegBaseAbs[spInInd] = tk;
                    //printf("%d: %0.2f -> %0.2f (%0.2f)\n", absPos, *ctime, tk, delays[i]);
                    //printf("%0.2f \n", delays[i]);
                    spInInd2 = (spInInd+1) % LEN_BUFFER_DELAYS;                    
                    if (spInInd2 == spOutInd) {
                        break;    // buffer is full        
                    }
                    spInInd = spInInd2;
                }
                *spRegInIndex = spInInd;
#endif   
            p = -65.0F;
            r = r + 8.0F;            
        } else {
            if (uv.x >= -82.F) {
                p = uv.x;
                r = uv.y;
            }
        }
        pot[absPos] = p;
        rec[absPos] = r;
    }
}

__kernel void
reset_weights(_gf *wt, float value) {
    unsigned int x = get_global_size(0)*get_global_size(1)*get_global_id(2) + get_global_id(1)*get_global_size(0) + get_global_id(0);
#ifdef DEBOUT
    if (x == 0) DEBPRINT2
#endif
    
    if (x < MAX_WT_IND) wt[x] = value;
}

__kernel void
reset_run_state(int detects, _gf *pot, _gf *rec, 
                _gf *d_r1, _gf *d_r2, _gf *d_o1, _gf *d_o2, 
                _gf *spkd, _gc *hasSpkd, _gi *offsets, 
                _gf *ctime, _gi *itr) {
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
#ifdef DEBOUT    
    if (x == 0) DEBPRINT
#endif
    int layr = get_global_id(2);
    int nind = get_global_size(0)*get_global_id(1)+get_global_id(0);
    int absPos = offsets[layr] + nind;  
    // TODO: really need to iterate layer-wise ?
    if (absPos < offsets[layr+1]) {
        spkd[absPos] = 0.0F;
        pot[absPos] = -65.0F;
        rec[absPos] = 0.0F;
        hasSpkd[absPos] = 0; 
        //printf("R%d\n",absPos);
        if (detects)  {
            d_r1[absPos] = 0.0F;
            d_r2[absPos] = 0.0F;
            d_o1[absPos] = 0.0F;
            d_o2[absPos] = 0.0F;
        }
    }    
}

__kernel void
force_spike(_gf *pot, _gf *rec, _gf *spkd, _gc *hasSpkd, _gi *offsets, _gf *ctime) {
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
#ifdef DEBOUT        
    if (x == 0) DEBPRINT
#endif
    int layr = get_global_id(2);
    int nind = get_global_size(0)*get_global_id(1)+get_global_id(0);
    int absPos = offsets[layr] + nind;

    spkd[absPos] = *ctime;
#ifndef SILENT_FORCE
    pot[absPos] = -65.0F;
    rec[absPos] = rec[absPos] + 8.0F;
#endif
    hasSpkd[absPos] = 1;
}

__kernel void
force_spike2(_gf *pot, _gf *rec, _gf *spkd, _gc *hasSpkd, _gi *offsets, _gf *ctime) {
    // Call this just before iterate
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
#ifdef DEBOUT        
    if (x == 0) DEBPRINT
#endif
    int layr = get_global_id(2);
    int nind = get_global_size(0)*get_global_id(1)+get_global_id(0);
    int absPos = offsets[layr] + nind;

    pot[absPos] = 30.;
}

/*
#ifndef TAU_R1
// tau+
#define TAU_R1 50.0F
// tauX
#define TAU_R2 20.0F
// tau-
#define TAU_O1 10.0F
// tauY
#define TAU_O2 500.0F
#endif
*/

__kernel void
detectors(_gf *d_r1, _gf *d_r2, _gf *d_o1, _gf *d_o2, _gc *hasSpkd, _gi *offsets, _gf *spkd, _gf *ctime) {
    // call this after weight update
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
#ifdef DEBOUT
    if (x == 0) DEBPRINT
#endif

    int layr = get_global_id(2);
    int nind = get_global_size(0)*get_global_id(1)+get_global_id(0);
    int absPos = offsets[layr] + nind;
    if (absPos < offsets[layr+1]) {
        if (spkd[absPos] == *ctime) {
            d_r1[absPos] += 1.0F * hasSpkd[absPos];
            d_r2[absPos] += 1.0F * hasSpkd[absPos];
            d_o1[absPos] += 1.0F * hasSpkd[absPos];
            d_o2[absPos] += 1.0F * hasSpkd[absPos];
        } else {
            d_r1[absPos] *= exp(-DT/TAU_R1);
            d_r2[absPos] *= exp(-DT/TAU_R2);
            d_o1[absPos] *= exp(-DT/TAU_O1);
            d_o2[absPos] *= exp(-DT/TAU_O2);
        }
    }
}

void printBin(int);

/*
#ifndef A2minus
#define A2minus 7.0E-3F
#define A3minus 2.3E-4F
#define A2plus 5.0E-10F
#define A3plus 6.2E-3F
#define WFactor 10.0F
#endif

#define A2minus 1.0F
#define A3minus 0.5F
#define A2plus 0.2F
#define A3plus 0.3F
#define WFactor 10.0F
*/

__kernel void
synWtNorm(_gi *sPreNs, _gi *sPreOffs, _gi *sOffsets, _gi *nOffsets, _gf *weights) {
    int postN = get_global_id(1); //1
    int preNIndx  = get_local_id(0); //0
    int sLayr = get_global_id(2); //0
    
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
#ifdef DEBOUT
    if (x == 0) DEBPRINT2
#endif

    bool validPostN = (nOffsets[sLayr+2] - nOffsets[sLayr+1]) > postN;
    int ofs = sOffsets[sLayr] + postN;
    int preNCount = sPreOffs[ofs + 1] - sPreOffs[ofs];
    bool validSyn = preNIndx < preNCount;
    /* unused
    int preN;
    int absPostNIndx = nOffsets[sLayr+1] + postN; */
    
    __local float preNWt[MAX_PRE_N + 1];

    if (validPostN && validSyn) {
        /* unused
        preN = nOffsets[sLayr] + sPreNs[ofs + preNIndx]; */
        preNWt[preNIndx] = weights[ofs+preNIndx];       
//        printf("\n%d->%d:%f(%d)", preNIndx,postN,weights[ofs+preNIndx],ofs+preNIndx);
//        printf("\n%d,%f", ofs+preNIndx, weights[ofs+preNIndx]);        
    }
    if (validPostN && !validSyn) {
        preNWt[preNIndx] = 0.0F;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    __local float netWt;

    if (validPostN) {
        if (preNIndx == preNCount) {            
            netWt = 0.0F;
            for (int i=0; i < preNCount; i++) {
                netWt += preNWt[i];      
#ifdef DEBOUT
                if (DEBNETWTOUT) {
                    printf("\n%.3f",preNWt[i]);
                }                
#endif
            }            
//            printf("\n%d,%d: %.3f",postN,sLayr,netWt);            
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (validPostN && validSyn) {
        float x = (weights[ofs+preNIndx]/netWt)*(DEF_W * preNCount);
//        printf("\n%d,%d: %.3f -> %.3f",postN,sLayr,weights[ofs+preNIndx],x);
        weights[ofs+preNIndx] = x;
    }   
}

__kernel void
triplet_stdp(_gi *sPreNs, _gi *sPreOffs, _gi *sOffsets, _gi *nOffsets, 
             _gs *neuronOffset,
             _gf *weights, 
             _gf *d_r1, _gf *d_r2, _gf *d_o1, _gf *d_o2, _gc *hasSpkd, _gf *spkd, _gf *ctime)
{
    // just like the synapse function, loop over all preN and postN pairs
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
#ifdef DEBOUT
    if (x == 0) DEBPRINT
#endif

    int postN = get_global_id(1); 
    int preNIndx  = get_local_id(0); 
    int sLayr = get_global_id(2); 
    bool validPostN = (nOffsets[sLayr+2] - nOffsets[sLayr+1]) > postN; //(nOffsets[sLayr+1] - nOffsets[sLayr]) > postN;

    int ofs = sOffsets[sLayr] + postN;
    int preNCount = sPreOffs[ofs + 1] - sPreOffs[ofs];
    bool validSyn = preNIndx < preNCount;
    int preN;
                          
    int absPostNIndx = nOffsets[sLayr+1] + postN;
    absPostNIndx = absPostNIndx + neuronOffset[absPostNIndx];
    float w;
    
    __local float postNWeights[MAX_PRE_N + 1];
        
    if (validSyn && validPostN) {    
        preN = nOffsets[sLayr] + sPreNs[ofs + preNIndx + neuronOffset[ofs + preNIndx]];
        // if preN spiked then only change weight_(preN -> postN)
        if (hasSpkd[preN] && (fabs(spkd[preN] - *ctime) < DT)) {
            w = WFactor * hasSpkd[preN] * d_o1[absPostNIndx] * (A2minus + A3minus * d_r2[preN]);
            weights[ofs+preNIndx + neuronOffset[ofs + preNIndx]] -= WFactor * w;        // error for nvidia 
        }
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    if (validSyn && validPostN) {        
        // if postN spiked then change weight_(all preN -> postN)
        if (hasSpkd[postN] && (fabs(spkd[absPostNIndx] - *ctime) < DT)) {            
            float dw = hasSpkd[absPostNIndx] * d_r1[preN] * (A2plus  + A3plus  * d_o2[absPostNIndx]);
            weights[ofs+preNIndx + neuronOffset[ofs + preNIndx]] += WFactor * dw;
            postNWeights[preNIndx] = WFactor * dw;            
        }
        else { 
            postNWeights[preNIndx] = 0.0F; 
        }        
    }
    if (validPostN && !validSyn) {
        postNWeights[preNIndx] = 0.0F;        
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    float netWt = 0.0F;
    if (validPostN) 
        if (preNIndx == preNCount) 
            for (int i=0; i < preNCount; i++)
                netWt += postNWeights[i];   
#if DISPDW
    if (netWt > 0.0F) {
        unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
        printf("dW @N%d, %0.3f = %0.3f\n", x, *ctime, netWt);
    }  
#endif
#ifdef DEBOUT   
    //if (DEBOUT && (x == 0)) DEBPRINT
#endif
}

__kernel void 
synPost1_Pre0(_gi *sPreNs, _gi *sPreOffs, _gi *sOffsets, _gi *nOffsets, 
              _gs *postNeuronOffset,
              _gf *weights, _gf *inscurrent, 
              _gf *spkd, _gc *hasSpkd, 
              _gf *ctime)
{
	
	int postN = get_global_id(1); //1
    // int postN = get_global_id(1) + get_global_id(0)* get_global_size(1);
    int preNIndx  = get_local_id(0); //0
    int sLayr = get_global_id(2); //0
    
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
#ifdef DEBOUT        
    if (x == 0) DEBPRINT
#endif
		
    bool validPostN = (nOffsets[sLayr+2] - nOffsets[sLayr+1]) > postN; //(nOffsets[sLayr+1] - nOffsets[sLayr]) > postN;
    int ofs = sOffsets[sLayr] + postN;
    int preNCount = sPreOffs[ofs + 1] - sPreOffs[ofs];
    bool validSyn = preNIndx < preNCount;
    int preN;

    int absPostNIndx = nOffsets[sLayr+1] + postN;
    
    absPostNIndx = absPostNIndx + postNeuronOffset[absPostNIndx];
    
    __local float preNCur[MAX_PRE_N + 1];
    float xx;
    
    if (validPostN && validSyn) {
        preN = nOffsets[sLayr] + sPreNs[ofs + preNIndx];
        xx = weights[ofs+preNIndx] * exp(0.1F*(spkd[preN] - *ctime)) * hasSpkd[preN];
        preNCur[preNIndx] = xx;        
    }
    if (validPostN && !validSyn) {
        preNCur[preNIndx] = 0.0F;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    float netCur = 0.0F;
    if (validPostN) {
        if (preNIndx == 0) {      
            //printf("PIndx %d\n", absPostNIndx);
            for (int i=0; i < preNCount; i++) {
                netCur += preNCur[i];
                //printf("%f @ %d (%d)\n", preNCur[i], absPostNIndx, i);
            }
            //printf("---\n");
            if (netCur > 0)
                //TODO: this command replaces the current, try atomic_add
                //atomic_add(inscurrent + absPostNIndx, netCur);
                inscurrent[absPostNIndx] += netCur;
            //printf("+++\n");
        }
        //printf("%d\n",absPostNIndx);
    } 
	
}

__kernel void 
synPost1_Pre0__test
             (_gi *sPreNs, _gi *sPreOffs, _gi *sOffsets, _gi *nOffsets, 
              _gs *postNeuronOffset,
              _gf *weights, _gf *inscurrent, 
              _gf *spkd, _gc *hasSpkd, 
              _gf *ctime)
{
    int postN = get_global_id(1); //1
    // int postN = get_global_id(1) + get_global_id(0)* get_global_size(1);
    int preNIndx  = get_local_id(0); //0
    int sLayr = get_global_id(2); //0
    
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
#ifdef DEBOUT        
    if (x == 0) DEBPRINT
#endif
    bool validPostN = (nOffsets[sLayr+2] - nOffsets[sLayr+1]) > postN; //(nOffsets[sLayr+1] - nOffsets[sLayr]) > postN;
    int ofs = sOffsets[sLayr] + postN;
    int preNCount = sPreOffs[ofs + 1] - sPreOffs[ofs];
    bool validSyn = preNIndx < preNCount;
    int preN;

    int absPostNIndx = nOffsets[sLayr+1] + postN;
    
    absPostNIndx = absPostNIndx + postNeuronOffset[absPostNIndx];
    
    __local float preNCur[MAX_PRE_N + 1];
    float xx;
    //if (postN == 0) printf("%d %d %d %d\n", sLayr, nOffsets[sLayr], nOffsets[sLayr+1], nOffsets[sLayr+2]);
    
    preNCur[preNIndx] = 0.0F;
    
    if (validPostN && validSyn) {
        preN = nOffsets[sLayr] + sPreNs[ofs + preNIndx];
        printf("%d,%d\n", absPostNIndx,preN);
        xx = weights[ofs+preNIndx] * exp(0.1F*(spkd[preN] - *ctime)) * hasSpkd[preN];
        printf("%d\n", xx);
        //printf("%d: %d\n",preN,hasSpkd[preN]);
        //printf("%d\n",absPostNIndx);
        //if (absPostNIndx == 2) printf("%d (%d) -> %d (%d), %d %f\n", preNIndx, preN, postN, absPostNIndx, sLayr, xx);
        //printf("preNCount %d\nS1 %d\nS2 %d\nofs %d\n", preNCount, sPreOffs[ofs + 1], sPreOffs[ofs], ofs);        
        //preNCur[preNIndx] = xx;        
    }
    
    /*
    barrier(CLK_LOCAL_MEM_FENCE);
    float netCur = 0.0F;
    if (validPostN) {
        if (preNIndx == 0) {      
            printf("PIndx %d\n", absPostNIndx);
            for (int i=0; i < preNCount; i++) {
                netCur += preNCur[i];
                //printf("%f @ %d (%d)\n", preNCur[i], absPostNIndx, i);
            }
            printf("---\n");
            inscurrent[absPostNIndx] += netCur;
            printf("+++\n");
        }
        //printf("%d\n",absPostNIndx);
    }   */   
}

__kernel void
increment_time(_gf *ctime, _gi *itr) {
    if (get_global_id(0) == 0) {
#ifdef DEBOUT
        DEBPRINT
#endif
        *ctime += DT;
        *itr   += 1;   
    }
}

__kernel void
log_var(_gf *pot, _gi *offsets, _gi *itr, 
    _gi *logInd,_gi *logSz, _gf *logPot) {
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
#ifdef DEBOUT
    if (x == 0) DEBPRINT2
#endif
    int layr = get_global_id(2);
    int nind = get_global_size(0)*get_global_id(1)+get_global_id(0);
    int absPos = offsets[layr] + nind;
    //int x = 0;
    if (absPos < offsets[layr+1]) {     
        //printf("\n%d",absPos);
        if (logInd[absPos] > 0) {
            x = logInd[absPos] + (*logSz)*(*itr) - 1;
            logPot[x] = pot[absPos];
            //printf("\n|%d,%d,%d,%d,%d|", absPos,logInd[absPos],*logSz,*itr,x);
        }
    }   
}

__kernel void
log_all(_gf *arr, _gi *itr, _gi *arrSz, _gf *store) {
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
#ifdef DEBOUT
    if (x == 0) DEBPRINT2
#endif
    int arr_i = get_global_size(0)*get_global_size(1)*get_global_id(2) + get_global_size(0)*get_global_id(1) + get_global_id(0);
    int store_i = (*itr)*(*arrSz) + arr_i;
    if (arr_i < *arrSz) {
        store[store_i] = arr[arr_i];
    }
}


/*
1. iterate_RS
2. synapse
3. triplet_stdp
4. detector_update
*/


// sPreNs: array of preneurons (indexed by sPreOffs)
// sPreOffs: array of offsets (indexed by sOffsets)
// sOffsets: array of offsets (indexed by layer number)
// L0_[0,1,5,6,8] -> L1_0; L0_[2,3,5] -> L1_1; L0_[1,4,7] -> L1_3
// L1_[1,2] -> L2_0; L2_[2,4] -> L2_1
// L2_[0,1] -> L3_0
// sPreNs = [0,1,5,6,8,2,3,5,1,4,7,1,2,2,4,0,1]
// sPreOffs = [0,5,8,11,13,15]
// sOffsets = [0,3,5]
// nOffsets = [0,9,14,16]


__kernel void
spikedLog(_gi *spkd, _gi *itr, _gi *offsets, _gi *logSpk, _gi *logSpkOffset, _gi *logItrOffset, _gf *ctime, _gf *potn) {
    unsigned int x = get_global_id(0) + get_global_id(1) + get_global_id(2);
#ifdef DEBOUT        
    if (x == 0) DEBPRINT
#endif
    int layr = get_global_id(2);        
    int lind = get_local_size(0)*get_local_id(1)+get_local_id(0);
    int lsize = offsets[layr+1] - offsets[layr];
    int padded_lsize = 32*ceil(lsize/32.0F); 
    
    __local bool outByte[MAX_N];

    if (lind < lsize) {
        int absPos = offsets[layr]+lind;        
        if (potn[absPos] > 0.0F) {
            //printf("Spike in!\n");
            outByte[lind] = 1;
        } else
            outByte[lind] = 0;        
    } else if (lind < padded_lsize) {
        outByte[lind] = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);   
    bool a = 0; 
    int bI = lind/32;
    int tmp = 0;
    
    if ((lind % 32 == 0) && (lind < padded_lsize)) {                    
        do {
            int sI = lind % 32;
            tmp = tmp | (outByte[lind]*(1 << sI));
            lind++;
            a = a | outByte[lind];
        } while (lind % 32 != 0);
        logSpk[bI + logSpkOffset[layr] + (*itr)*(*logItrOffset)] = tmp;
    }   
}

