# --------------------------------------------------------------
# spec library
# Set of processing algorithms and routines for analying high-resolution
# echelle/grating and cross-dispersed data
# G.L.Villanueva - NASA/GSFC - January/2021
# To run $>python -m spec module=crop show=show reset=reset
# modules: crop, clean, spectral, store, fluxcal, qscale, geometry, extract, retrieve
# reset:0 no reset
# reset:1 reset this module
# reset:2 fully reset
# show:0 no show
# show:1 show and no wait
# show:2 interactive
#
# Updated: S.Faggi - NASA/GSFC - April-July/2021
#(1)
# <DATA-ORIENTATION> has now 2 options: rotate/flip
# For NIRSPEC: orientation.find('rotate') -> np.rot90(..., k=3)
# For iSHELL: orientation.find('flip') -> np.flip(..., 0)
# The code aslo allows to both flip and rotate.
#(2)
# iSHELL data need to be divided by NDR, division is done after a1b1b2a2 beam identification
# flats/darks are taken at NDR =1
# For iSHELL, line 269: object_data/NDR
#(3)
# Q-scale: line code 1570,1576,1579, 1580, 1582, 1582 cmax is replaced with gmax.
#          line 1406 dslitwp definition has ot be without "-1"
#
#(4)
# Fringe correction has been updated. Also Removefringe function.
#
#(5)
# Retrievals updated with Liuzzi -  November 2021
#
#(6)
# Read data discarding hidden files: line 89, 93, 100, 128, 248 - December 2021
# files = [_ for _ in os.listdir(path) if not _.startswith('.')]
#
#(7)
# Spectral section "ref" variabble not used. Data/model needs to be implemented. - December 2021
# When nodding ON and not Radiance is now updated as follwoing line 960:
#
# datas[iset,0:rmid,:] = ma[iset,0:rmid,:]-mb[iset,0:rmid,:]  ( Mars-Sky) as before it was Sky-Mars
# datas[iset,rmid:szy,:] = mb[iset,rmid:szy,:]-ma[iset,rmid:szy,:] (same)
# --------------------------------------------------------------
import os, struct
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Functions to read parameters from configuration file
# Put the spec.py in your library folder (e.g., data/software/python)
# Set export PYTHONPATH="/Users/glvillan/data/software/python" in your .zshrc
# from the command line >python -m spec module=crop show=True
cfgvars=[]; cfgvals=[]; cfgfile='spec.cfg'
def cfgread(param,default='',reset=False):
    if len(cfgvars)==0 or reset:
        if not os.path.exists(cfgfile): return default
        fr = open(cfgfile,'r')
        cfglines = fr.readlines()
        fr.close()
        for ln in cfglines:
            ln=ln.strip()
            if len(ln)==0: continue
            if ln[0]!='<': continue
            sp = ln.find('>')
            cfgvars.append(ln[1:sp])
            vv = ln[sp+1:]
            if len(vv)==1: vv = vv.lower()
            cfgvals.append(vv)
        #End line loop
    #End
    if param not in cfgvars: return(default)
    return cfgvals[cfgvars.index(param)]
#End cfgread

# --------------------------------------------------------------
# Crop module
# --------------------------------------------------------------
def crop(show=False,reset=False):

    # Get variables
    from astropy.io import fits
    rawdir = cfgread('DATA-RAW','../raw')
    flatsdir = cfgread('DATA-FLATS','../flats')
    darksdir = cfgread('DATA-DARKS','../darks')
    npairs = cfgread('DATA-NPAIRS','all')
    sets = cfgread('DATA-SETS','all')
    orientation = cfgread('DATA-ORIENTATION',''); orientation=orientation.split(',')
    cropsec = cfgread('DATA-SECTION','500,500'); cropsec=cropsec.split(',')
    cropyrange = cfgread('DATA-YRANGE','auto'); cropyrange=cropyrange.split(',')
    cropxrange = cfgread('DATA-XRANGE','all'); cropxrange=cropxrange.split(',')
    nodding = cfgread('DATA-NODDING','on')
    nodtype = cfgread('DATA-NODTYPE','abba')

    # Calculate netflat frame --------
    files=[]
    if os.path.exists(flatsdir): files = [_ for _ in os.listdir(flatsdir) if not _.startswith('.')]
    if len(files)>0:
        data = fits.getdata('%s/%s' % (flatsdir,files[0]), ext=0)
        sz = data.shape
        netflat = np.zeros(sz)
        for file in files:
            data = fits.getdata('%s/%s' % (flatsdir,file), ext=0)
            netflat += data/len(files)
        #Endfor
        files=[]
        if os.path.exists(darksdir): files = [_ for _ in os.listdir(darksdir) if not _.startswith('.')]
        for file in files:
            data = fits.getdata('%s/%s' % (darksdir,file), ext=0)
            netflat -= data/len(files)
        #Endfor
        base = netflat
    else:
        files = [_ for _ in os.listdir(rawdir) if not _.startswith('.')]
        base = fits.getdata('%s/%s' % (rawdir,files[0]), ext=0)
        sz = data.shape
        netflat = np.zeros(sz)+1.0
    #Endelse

    # Rotate/flip as needed
    if len(orientation)==1 and orientation[0]!= '' :
        if orientation[0].find('rotate')>=0:
            base = np.rot90(base,k=3)
            netflat = np.rot90(netflat,k=3)
        if orientation[0].find('flip')>=0:
            base = np.flip(base,0)
            netflat = np.flip(netflat,0)
    if len(orientation)>1 :
        base = np.rot90(base,k=3)
        base = np.flip(base,0)
        netflat = np.rot90(netflat,k=3)
        netflat = np.flip(netflat,0)

    #Endif

    # Show image and allow extracting x,y positions for cropping
    if show:
        def onclick(event): print(int(event.xdata), int(event.ydata))
        fig, ax = plt.subplots(1,2,figsize=[10,5])
        ax[0].imshow(base,vmin=0,vmax=np.median(base)*2.0)
        ax[0].set_title('Flats-Darks')
        files = [_ for _ in os.listdir(rawdir) if not _.startswith('.')]
        rdata = fits.getdata('%s/%s' % (rawdir,files[0]), ext=0)
        if len(orientation)==1 and orientation[0]!= '' :
            if orientation[0].find('rotate')>=0: rdata = np.rot90(rdata,k=3)
            if orientation[0].find('flip')>=0:   rdata = np.flip(rdata,0)
        if len(orientation)>1 :
            rdata = np.rot90(rdata,k=3)
            rdata = np.flip(rdata,0)
        ax[1].imshow(rdata,vmin=0,vmax=np.median(rdata)*2.0)
        ax[1].set_title('Raw frame')
        plt.tight_layout()
        if show>1:
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1.0)
            plt.close()
        #Endelse
    #Endshow

    # Define crop region --------
    if cropxrange[0]=='all':
        x1=0; x2=sz[0]-1
    else:
        x1 = int(cropxrange[0]); x2 = int(cropxrange[1])
    #Endif
    szx = x2-x1+1
    xpix = np.arange(x1,x2+1)

    if len(cropsec)==2 or cropyrange[0]=='auto':
        # Identify order shape by edges
        if len(cropsec)>2:
            clo = [float(x) for x in cropsec]
            xp = int((x1+x2)/2)
            yp = int(np.polyval(clo, xp))
        else:
            xp = int(cropsec[0])
            yp = int(cropsec[1])
        #Endif
        y1 = np.zeros(sz[0]); y2 = np.zeros(sz[1])
        x = xp; y0 = yp; inc = 1;
        while x>=x1:
            if x>x2:
                x=xp-1
                y0=yp
                inc=-1
                continue
            #Endif to go the other direction
            vref = np.median(base[y0-2:y0+2,x])
            y = y0
            while y<sz[1]-3 and ((y-y2[x-inc])<10 or x==xp) and (((y2[x-inc]-y)>10 and x!=xp) or np.median(base[y-2:y+2,x])>0.2*vref): y=y+1
            y2[x] = y; y = y0
            while y>2 and ((y1[x-inc]-y)<10 or x==xp) and (((y-y1[x-inc])>10 and x!=xp) or np.median(base[y-2:y+2,x])>0.2*vref): y=y-1
            y1[x] = y;
            y = int((y1[x]+y2[x])*0.5)
            if abs(y-y0)<5 or y0==yp: y0=y
            x += inc
        #Endloop
        clo = np.polyfit(xpix, y1[x1:x2+1], 2)
        ylo = np.polyval(clo,xpix)
        chi = np.polyfit(xpix, y2[x1:x2+1], 2)
        yhi = np.polyval(chi,xpix)
        dymin = int(np.min(yhi-ylo))
        dymax = int(np.max(yhi-ylo))
        if show>1:
            print('Low edge: %e,%e,%e' % (clo[0],clo[1],clo[2]))
            print('High edge: %e,%e,%e' % (chi[0],chi[1],chi[2]))
            print('Maximum dy: %d' % dymax)
        #Endif
    #Endif
    if len(cropsec)==3:
        clo = np.asarray([float(x) for x in cropsec])
    elif len(cropsec)>3:
        pv = np.asarray([float(x) for x in cropsec])
        nn = int(len(pv)/2)
        pp = np.zeros([2,nn])
        for i in range(nn):
            pp[0,i]=pv[i*2]
            pp[1,i]=pv[i*2+1]
        #Endfor
        if nn==2: clo = np.concatenate([[0.0], np.polyfit(pp[0,:], pp[1,:], 1)])
        else: clo = np.polyfit(pp[0,:], pp[1,:], 2)
    #Endelse

    if len(cropyrange)==1 and cropyrange[0]!='auto':
        chi = 1.0*clo
        szy = int(cropyrange[0])
        if szy<0: clo[2] += szy
        else: chi[2] += szy
    elif len(cropyrange)==3:
        chi = np.asarray([float(x) for x in cropyrange])
    elif len(cropyrange)>3:
        pv = np.asarray([float(x) for x in cropyrange])
        nn = int(len(pv)/2)
        pp = np.zeros([2,nn])
        for i in range(nn):
            pp[0,i]=pv[i*2]
            pp[1,i]=pv[i*2+1]
        #Endfor
        if nn==2: chi = np.concatenate([[0.0], np.polyfit(pp[0,:], pp[1,:], 1)])
        else: chi = np.polyfit(pp[0,:], pp[1,:], 2)
    #Endelse

    # Define shifts for each type of frame
    ylo = np.polyval(clo,xpix)
    yhi = np.polyval(chi,xpix)
    cropyrange = int(np.max(yhi-ylo))
    szy = cropyrange
    ys = ((ylo+yhi-szy)/2.0).astype(int)
    nf = np.zeros([szy,szx])
    for x in range(szx): nf[:,x] = netflat[ys[x]:ys[x]+szy,x]
    nf = nf/np.median(nf)
    nfmask = np.zeros([szy,szx],dtype=bool)
    bad = ((nf<0.1) + (nf>10.0)).nonzero();
    nf[bad]=1.0;
    nfmask[bad]=True

    # Read data --------
    files=[]
    if os.path.exists(rawdir): files = sorted([_ for _ in os.listdir(rawdir) if not _.startswith('.')])
    if npairs=='all':
        npairs=int(len(files)/2)
    else:
        npairs=int(npairs)
    #Endif
    nsets = int(len(files)/(2*npairs))
    dt = np.zeros([szy,szx])
    a1 = np.zeros([nsets,szy,szx]);
    b1 = np.zeros([nsets,szy,szx]);
    b2 = np.zeros([nsets,szy,szx]);
    a2 = np.zeros([nsets,szy,szx]);
    fw = open('spec.sets.txt','w')

    for i in range(nsets*npairs*2):
        data = fits.getdata('%s/%s' % (rawdir,files[i]), ext=0)
        if len(orientation)==1 and orientation[0]!= '' :
            if orientation[0].find('rotate')>=0: data = np.rot90(data,k=3)
            if orientation[0].find('flip')>=0:   data = np.flip(data,0)
        if len(orientation)>1:
            data = np.rot90(data,k=3)
            data = np.flip(data,0)
        for x in range(szx): dt[:,x] = data[ys[x]:ys[x]+szy,x]/(nf[:,x]*npairs)
        iset = int(i/(npairs*2))
        if nodtype=='abba':
            if   i%4==0: a1[iset,:,:] += dt
            elif i%4==1: b1[iset,:,:] += dt
            elif i%4==2: b2[iset,:,:] += dt
            elif i%4==3: a2[iset,:,:] += dt
        else:
            if   i%4==0: a1[iset,:,:] += dt # for abab nodding
            elif i%4==1: b1[iset,:,:] += dt
            elif i%4==2: a2[iset,:,:] += dt
            elif i%4==3: b2[iset,:,:] += dt
        #Endelse

        # Process header information
        if i%(npairs*2)==npairs:
            header = fits.getheader('%s/%s' % (rawdir,files[i]), ext=0);
            inst=''; date='2020-06-01'; time='12:00'; airmass='1.0'; itime='60.0'; coadds='1'; ndr='1'
            if 'INSTRUME' in header: inst=header['INSTRUME'].upper()
            elif 'CURRINST' in header: inst=header['CURRINST'].upper()
            if inst.find('ISHELL')>=0:
                # NASA-IRTF/iSHELL
                if 'DATE_OBS' in header: date=header['DATE_OBS']
                if 'TIME_OBS' in header: time=header['TIME_OBS']
                if 'TCS_AM' in header: airmass=header['TCS_AM']
                if 'ITIME' in header: itime=float(header['ITIME'])
                if 'CO_ADDS' in header: coadds=header['CO_ADDS']
                if 'NDR' in header: ndr=header['NDR']
                a1[iset,:,:] /= ndr
                b1[iset,:,:] /= ndr
                b2[iset,:,:] /= ndr
                a2[iset,:,:] /= ndr
            elif inst.find('NIRSPEC')>=0:
                # Keck/NIRSPEC
                if 'DATE-OBS' in header: date=header['DATE-OBS']
                if 'UTSTART' in header: time=header['UTSTART']
                elif 'UTC' in header: time=header['UTC']
                if 'AIRMASS' in header: airmass=header['AIRMASS']
                if 'ITIME' in header: itime=float(header['ITIME'])/1e3
                if 'COADDS' in header: coadds=header['COADDS']
                if 'NUMREADS' in header: ndr=header['NUMREADS']
            #Endelse
            fw.write('%s %.5s,%s,%d,%s,%s,%s\n' % (date,time,airmass,npairs*2,itime,coadds,ndr))
        #Endif
    #End for sets-loop
    fw.close()
    # Organize the data into A and B frame
    if npairs>1:
        ma = (a1+a2)/2.0
        mb = (b1+b2)/2.0
        da = (a2-a1)
        db = (b2-b1)
    else:
        ma = np.zeros([nsets,szy,szx]);
        mb = np.zeros([nsets,szy,szx]);
        da = np.zeros([nsets,szy,szx]);
        db = np.zeros([nsets,szy,szx]);
        for i in range(nsets):
            if i%2==0:
                ma[i,:,:] = a1[i,:,:]
                mb[i,:,:] = b1[i,:,:]
                da[i,:,:] = a2[i+1,:,:]-a1[i,:,:]
                db[i,:,:] = b2[i+1,:,:]-b1[i,:,:]
            else:
                ma[i,:,:] = a2[i,:,:]
                mb[i,:,:] = b2[i,:,:]
                da[i,:,:] = a2[i,:,:]-a1[i-1,:,:]
                db[i,:,:] = b2[i,:,:]-b1[i-1,:,:]
            #Endif
        #Endif
    #Endif

    # Save the data
    fw = open('spec.crop.dat','wb')
    np.asarray([nsets,szy,szx]).tofile(fw)
    ma.tofile(fw)
    mb.tofile(fw)
    da.tofile(fw)
    db.tofile(fw)
    nfmask.tofile(fw)
    ylo.tofile(fw)
    yhi.tofile(fw)
    fw.close()
#End crop module

# --------------------------------------------------------------
# Clean module
# --------------------------------------------------------------
def clean(show=False,reset=False):

    # Read crop results
    if reset==2 or not os.path.exists('spec.crop.dat'): crop(show=show,reset=reset)
    fr = open('spec.crop.dat','rb')
    nsets = np.fromfile(fr, dtype=int, count=1)[0]
    szy = np.fromfile(fr, dtype=int, count=1)[0]
    szx = np.fromfile(fr, dtype=int, count=1)[0]
    ma = np.fromfile(fr, dtype=float, count=nsets*szx*szy); ma = np.reshape(ma, [nsets,szy,szx])
    mb = np.fromfile(fr, dtype=float, count=nsets*szx*szy); mb = np.reshape(mb, [nsets,szy,szx])
    da = np.fromfile(fr, dtype=float, count=nsets*szx*szy); da = np.reshape(da, [nsets,szy,szx])
    db = np.fromfile(fr, dtype=float, count=nsets*szx*szy); db = np.reshape(db, [nsets,szy,szx])
    nfmask = np.fromfile(fr, dtype=bool, count=szx*szy); nfmask = np.reshape(nfmask, [szy,szx])
    ylo = np.fromfile(fr, dtype=float, count=szx)
    yhi = np.fromfile(fr, dtype=float, count=szx)
    fr.close()

    # Get variables
    limits = cfgread('CLEAN-THRESHOLD','4.0')
    orders = cfgread('CLEAN-ORDER','4')
    nodding = cfgread('DATA-NODDING','on')

    # Prepare variables
    limits = limits.split(','); v = float(limits[0])
    if len(limits)==1: limits=[v,v,v,v]
    else: limits=[float(v) for v in limits]
    orders = orders.split(','); v = int(orders[0])
    if len(orders)==1: orders=[v,v,v,v]
    else: orders=[int(v) for v in orders]
    dymin = int(np.min(yhi-ylo))
    dymax = int(np.max(yhi-ylo))
    ypix = np.arange(0,szy) - int(szy/2)

    # Iterate through sets
    for iset in range(nsets):
        # Identify bad pixels
        mask = 1*nfmask
        for k in range(4):
            #Define beams and regions to clean (sequence is dB,dA,B,A)
            y1 = 0; y2 = szy
            if   k==0: dt=db[iset,:,:]
            elif k==1: dt=da[iset,:,:]
            elif k==2:
                dt = mb[iset,:,:]
                if nodding=='on':
                    y1 = 0
                    y2 = int(szy/2)
                #Endif
            elif nodding=='on':
                dt = ma[iset,:,:]
                y1 = int(szy/2)
                y2 = szy
            else: continue

            # Mask the array
            xvals = np.arange(y1,y2)
            for x in range(szx):
                prof = dt[y1:y2,x]
                msk = 1-mask[y1:y2,x]
                nbad=np.sum(1-msk); ibad=1
                while ibad>0 and nbad<(y2-y1):
                    fit = np.polyval(np.polyfit(xvals, prof, orders[k], w=msk), xvals)
                    sigma = np.std(msk*(prof-fit))
                    bad = (abs(msk*(prof-fit)) > limits[k]*sigma).nonzero()[0]
                    ibad = len(bad)
                    nbad += ibad
                    msk[bad] = 0
                #Bad pixel loop
                mask[y1:y2,x] = 1-msk
            #Endfor x
        #Endfor beams

        # Clean the array
        bad = (mask==True).nonzero()
        for i in range(len(bad[0])):
            ngood=0; r=2; xp=bad[1][i]; yp=bad[0][i]
            while ngood<20:
                xg1=xp-r; xg2=xp+r+1;
                yg1=yp-r; yg2=yp+r+1;
                if xg1<0: xg1=0
                if xg2>szx: xg2=szx
                if yg1<0: yg1=0
                if yg2>szy: yg2=szy
                good = (mask[yg1:yg2,xg1:xg2]==False).nonzero()
                ngood = len(good[0])
                r=r+1
            #Endloop
            vals = ma[iset,yg1:yg2,xg1:xg2];
            val = np.median(vals[good])
            ma[iset,yp,xp] = val
            vals = mb[iset,yg1:yg2,xg1:xg2];
            val = np.median(vals[good])
            mb[iset,yp,xp] = val
        #Endfor

        # Peform anamorphic spatial correction
        ys = ((ylo+yhi-szy)/2.0).astype(int)
        for i in range(szx):
            ypia = ypix*(yhi[i]-ylo[i])/dymax + (((ylo[i]+yhi[i]-szy)/2.0)-ys[i])
            ma[iset,:,i] = np.interp(ypia, ypix, ma[iset,:,i])
            mb[iset,:,i] = np.interp(ypia, ypix, mb[iset,:,i])
        #Endfor

        if show:
            fig,ax = plt.subplots(3,figsize=[12,6])
            fig.suptitle('Set %d' % iset)
            ax[0].imshow(mask); ax[0].set_title('Mask (%d bad pixels)' % len(bad[0]))
            ax[1].imshow(ma[iset,:,:]); ax[1].set_title('A clean')
            ax[2].imshow(mb[iset,:,:]); ax[2].set_title('B clean')
            plt.tight_layout()
            plt.savefig('spec.clean.%02d.png' % iset)
            if show>1:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(1.0)
                plt.close()
            #Endelse
        #Endif
    #Endfor sets


    # Save the data
    fw = open('spec.clean.dat','wb')
    np.asarray([nsets,szy,szx]).tofile(fw)
    ma.tofile(fw)
    mb.tofile(fw)
    fw.close()
#End clean module

# --------------------------------------------------------------
# Spectral alignment and calibration module
# --------------------------------------------------------------
# Compute telluric transmittances
def gentrans(vrange=[2920,2930], show=0, reset=0, atmfile='spec.model', iset=-1, config=0, vstar=0.0):
    # Get telluric parameters
    psgserver = cfgread('SPECTRAL-SERVER','https://psg.gsfc.nasa.gov')
    psgkey = cfgread('SPECTRAL-PSGKEY','')
    if len(psgkey)>0: psgserver = '-d key=%s %s' % (psgkey, psgserver)
    atmloc = cfgread('ATMOSPHERE-LOCATION','19.82066407,204.53193360')
    atmdate = cfgread('ATMOSPHERE-DATE','spec.sets.txt')
    atmam = cfgread('ATMOSPHERE-AIRMASS','spec.sets.txt')
    atmscl = cfgread('ATMOSPHERE-ABUNDANCES','spec.model.scl')
    atmres = cfgread('ATMOSPHERE-RESOLUTION','70000')
    matm = cfgread('ATMOSPHERE-MULTIPLE','n')
    if matm=='n': iset=-1

    # Get telluric profile
    if not os.path.exists('%s.atm' % atmfile):
        # Define observing data
        if os.path.exists(atmdate):
            fr = open(atmdate,'r'); lines=fr.readlines(); fr.close()
            if iset<0: k=int(len(lines)/2)
            else: k=iset
            atmdate = lines[k].split(',')[0]
        #Endif
        atmdate = atmdate.replace('-','/')
        fw = open('%s.cfg' % atmfile,'w')
        fw.write('<OBJECT>Earth\n')
        fw.write('<OBJECT-DATE>%s\n' % atmdate)
        fw.write('<OBJECT-OBS-LATITUDE>%s\n' % atmloc.split(',')[0])
        fw.write('<OBJECT-OBS-LONGITUDE>%s\n' % atmloc.split(',')[1])
        fw.write('<GEOMETRY>Lookingup\n')
        fw.write('<GEOMETRY-OBS-ALTITUDE>0.0\n')
        fw.write('<GEOMETRY-ALTITUDE-UNIT>km\n')
        fw.write('<ATMOSPHERE-NGAS>8\n')
        fw.write('<ATMOSPHERE-GAS>H2O,CO2,O3,N2O,CO,CH4,O2,N2\n')
        fw.write('<ATMOSPHERE-TYPE>HIT[1],HIT[2],HIT[3],HIT[4],HIT[5],HIT[6],HIT[7],HIT[22]\n')
        fw.write('<ATMOSPHERE-UNIT>scl,scl,scl,scl,scl,scl,scl,scl\n')
        fw.close()
        os.system('curl -s -d type=cfg -d wephm=y -d watm=y --data-urlencode file@%s.cfg %s/api.php > %s.atm' % (atmfile,psgserver,atmfile))
    #Endif

    # Compute radiances / transmittances
    if iset<0: atmout = '%s.all' % atmfile
    else: atmout = '%s.%02d.all' % (atmfile,iset)
    if not os.path.exists(atmout) or reset or config:
        # Define airmass
        if os.path.exists(atmam):
            fr = open(atmam,'r'); lines=fr.readlines(); fr.close()
            if iset<0:
                airmass=0.0
                for line in lines: airmass += float(line.split(',')[1])/len(lines)
            else:
                airmass = float(lines[iset].split(',')[1])
            #Endelse
        else: airmass=float(atmam)

        # Define molecular scalers
        if os.path.exists(atmscl):
            fr = open(atmscl,'r'); lines=fr.readlines(); fr.close()
            if iset<0: k=int(len(lines)/2)
            else: k=iset
            if len(lines)<=k: atmscl='1,1,1,1,1,1,1,1'
            else: atmscl = lines[k][:-1]
        elif len(atmscl.split(','))<=1:
            atmscl = '1,1,1,1,1,1,1,1'
        #Endif
        fr = open('%s.atm' % atmfile,'r'); lines=fr.readlines(); fr.close()
        fw = open('%s.cfg' % atmfile,'w');
        fw.writelines(lines)
        fw.write('<ATMOSPHERE-ABUN>%s\n' % atmscl)
        fw.write('<OBJECT-STAR-VELOCITY>%.4f\n' % vstar)
        fw.write('<GENERATOR-CONT-STELLAR>Y\n')
        fw.write('<GEOMETRY-USER-PARAM>%.4f\n' % ((180.0/np.pi)*np.arccos(1.0/airmass)))
        fw.write('<GENERATOR-RANGE1>%.4f\n' % vrange[0])
        fw.write('<GENERATOR-RANGE2>%.4f\n' % vrange[1])
        fw.write('<GENERATOR-RANGEUNIT>cm\n')
        fw.write('<GENERATOR-RESOLUTION>%s\n' % atmres)
        fw.write('<GENERATOR-RESOLUTIONUNIT>RP\n')
        fw.write('<GENERATOR-RESOLUTIONKERNEL>Y\n')
        fw.close()
        if config: return
        else: os.system('curl -s -d type=all --data-urlencode file@%s.cfg %s/api.php > %s' % (atmfile,psgserver,atmout))
    #Endif

    # Read telluric radiances / transmittances
    fr = open(atmout,'r'); lines=fr.readlines(); fr.close()
    mtel = np.zeros([4, int(len(lines)/3)])
    inrad=0; intrn=0; instr=0; maxstr=0.0
    for line in lines:
        if line[:8]=='results_': inrad=0; intrn=0; instr=0; npts=0
        if line[0]=='#':
            continue
        elif line[:-1]=='results_rad.txt':
            inrad=1; continue
        elif line[:-1]=='results_trn.txt':
            intrn=1; continue
        elif line[:-1]=='results_str.txt':
            instr=1; continue
        if inrad:
            ss=line.split()
            mtel[0,npts] = float(ss[0])
            mtel[1,npts] = float(ss[1])
            npts=npts+1
        elif intrn:
            ss=line.split()
            mtel[2,npts] = float(ss[1])
            npts=npts+1
        elif instr:
            ss=line.split()
            mtel[3,npts] = float(ss[1])
            if mtel[3,npts]>maxstr: maxstr=mtel[3,npts]
            npts=npts+1
        #Endif
    #Endfor
    mtel = np.flip(mtel[:,0:npts],axis=1)
    if maxstr>0: mtel[3,:] = mtel[3,:]/maxstr - 1.0
    return mtel
#End gentrans module

# Retrieve transmittance
def retrievetrans(freq, spec, noise, show=0, atmfile='spec.model', iset=-1, vstar=0.0):

    # Initialize parameters
    psgserver = cfgread('SPECTRAL-SERVER','https://psg.gsfc.nasa.gov')
    psgkey = cfgread('SPECTRAL-PSGKEY','')
    if len(psgkey)>0: psgserver = '-d key=%s %s' % (psgkey, psgserver)
    rmols = cfgread('ATMOSPHERE-RETRIEVE','n').split(',')
    atmam = cfgread('ATMOSPHERE-AIRMASS','spec.sets.txt')
    atmscl = cfgread('ATMOSPHERE-ABUNDANCES','spec.model.scl')
    matm = cfgread('ATMOSPHERE-MULTIPLE','n')
    fitfreq = cfgread('EXTRACT-FITFREQ','n'); fitfreq.split(',')[0]
    ngain = int(cfgread('EXTRACT-FITGAIN','2'))
    if ngain=='n': ngain=-1
    else: ngain = int(ngain)
    nbase = cfgread('EXTRACT-REMOVEOFFSET','-1')
    if nbase=='n': nbase=-1
    else: nbase = int(nbase)
    nfringe = cfgread('EXTRACT-REMOVEFRINGE','0')
    if nfringe=='n': nfringe=0
    else: nfringe = int(nfringe)
    fitstr = cfgread('EXTRACT-FITSTELLAR','n')
    if matm=='n': iset=-1
    fmin = np.min(freq)-2
    fmax = np.max(freq)+2

    # Define molecules to retrieve
    rmol = []; imol = []; xmol=[]
    mols = ['H2O','CO2','O3','N2O','CO','CH4','O2','N2']
    for mol in rmols:
        for x in range(len(mols)):
            if mol.lower()==mols[x].lower(): rmol.append(mols[x]); imol.append(x)
    #Endfor
    if len(rmol)==0: return

    # Define molecular scalers and number of sets
    if os.path.exists(atmam): fr = open(atmam,'r'); namsets=len(fr.readlines()); fr.close()
    else: namsets=1
    if matm=='n': nsets=1; kset=0
    elif iset<0: nsets=namsets; kset=int(nsets/2)
    else: nsets=namsets; kset=iset
    if os.path.exists(atmscl):
        fr = open(atmscl,'r'); mlines=fr.readlines(); fr.close()
        for i in range(len(mlines)): mlines[i] = mlines[i][:-1]
        for i in range(nsets-len(mlines)): mlines.append('1,1,1,1,1,1,1,1')
        atmscl = mlines[kset]
    elif len(atmscl.split(','))<len(mols):
        atmscl = '1,1,1,1,1,1,1,1'; mlines=[]
        for i in range(nsets): mlines.append(atmscl)
    #Endif
    atmxs = [float(x) for x in atmscl.split(',')]
    for x in imol: xmol.append(atmxs[x])

    # Write retrieval config
    gentrans([fmin,fmax], show=show, atmfile=atmfile, iset=iset, config=1, vstar=vstar)
    fw = open('%s.cfg' % atmfile,'a');
    fw.write('<GEOMETRY>LookingObject\n');
    fw.write('<GEOMETRY-STELLAR-TYPE>G\n');
    fw.write('<GEOMETRY-STELLAR-TEMPERATURE>5778\n');
    fw.write('<GEOMETRY-STELLAR-MAGNITUDE>-10\n');
    fw.write('<OBJECT-OBS-VELOCITY>%.4f\n' % vstar)
    fw.write('<GENERATOR-RADUNITS>rif\n');
    fw.write('<RETRIEVAL-FITGAIN-PHOTOMETRIC>N\n');
    fw.write('<RETRIEVAL-REMOVEOFFSET>%d\n' % nbase);
    fw.write('<RETRIEVAL-FITGAIN>%d\n' % ngain);
    fw.write('<RETRIEVAL-FITSTELLAR>%s\n' % fitstr.upper());
    fw.write('<RETRIEVAL-FITFREQ>%s\n' % fitfreq.upper());
    fw.write('<RETRIEVAL-REMOVEFRINGE>%d\n');
    fw.write('<RETRIEVAL-FITTELLURIC>N\n');
    fw.write('<RETRIEVAL-FITRESOLUTION>N\n');
    fw.write('<RETRIEVAL-NVARS>%d\n' % len(rmol));
    svar=''; sval=''; smin=''; smax=''; suni=''
    for i in range(len(rmol)):
        if i==0: sfmt=''
        else: sfmt=','
        svar = '%s%sATMOSPHERE-%s' % (svar,sfmt,rmol[i])
        sval = '%s%s%.3f' % (sval,sfmt,xmol[i])
        smin = '%s%s0.0' % (smin,sfmt)
        smax = '%s%s10.0' % (smax,sfmt)
        suni = '%s%sscl' % (suni,sfmt)
    #Endfor
    fw.write('<RETRIEVAL-VARIABLES>%s\n' % svar);
    fw.write('<RETRIEVAL-VALUES>%s\n' % sval);
    fw.write('<RETRIEVAL-MIN>%s\n' % smin);
    fw.write('<RETRIEVAL-MAX>%s\n' % smax);
    fw.write('<RETRIEVAL-UNITS>%s\n' % suni);
    fw.write('<DATA>\n');
    for i in range(len(freq)): fw.write('%f %e %e\n' % (freq[i],spec[i],noise[i]));
    fw.write('</DATA>\n');
    fw.close()
    os.system('curl -s -d type=ret --data-urlencode file@%s.cfg %s/api.php > %s.ret' % (atmfile,psgserver,atmfile))

    # Process results
    fr = open('%s.ret' % atmfile ,'r'); lines=fr.readlines(); fr.close()
    svals = '<RETRIEVAL-VALUES>1,1,1,1,1,1,1,1,1'
    for line in lines:
        if line.find('<RETRIEVAL-VALUES>')>=0: svals=line; break
    svals=svals.split('>')[1]; svals=svals.split(','); sval=''
    for i in range(len(rmol)): atmxs[imol[i]] = float(svals[i])
    for i in range(len(mols)):
        if i==0: sfmt=''
        else: sfmt=','
        sval = '%s%s%.4f' % (sval,sfmt,atmxs[i])
    #Endfor
    mlines[kset] = sval
    fw = open('%s.scl' % atmfile ,'w')
    for i in range(nsets): fw.write('%s\n' % mlines[i])
    fw.close()

    # Generate new solution
    mtel = gentrans([fmin,fmax], show=show, atmfile=atmfile, iset=iset, reset=1, vstar=vstar)
    return mtel
#End retrievetrans


# Module to compute frequency vector from coefficients
def genfreq(szx, cf, x1=0, x2=0):
    if x2==0: x2=szx
    xmid = int(szx/2)
    xpix = np.arange(0,szx)
    lf = np.polyval([cf[3],cf[1],cf[0]], xpix[x1:xmid]-xmid)
    rf = np.polyval([cf[4],cf[2],cf[0]], xpix[xmid:x2]-xmid)
    fr = np.concatenate((lf,rf))
    return fr
#End genfreq module

# Frequency fitting algorithm
ffit_cf=0; ffit_scl=1
def ffit(spec, mfreq, mspec, coeff0, vs=[400,100,100,5,0.5], vl=[1e-2,1e-4,1e-6], type='dual', show=0):

    import matplotlib.gridspec as gridspec
    from matplotlib.widgets import Button
    npts = len(spec)
    xmid = int(npts/2)
    xpix = np.arange(0,npts)
    cf = np.asarray(coeff0)*1.0; iter=0
    spec = spec-np.min(spec)
    spec = spec/np.max(spec)
    mspec= mspec-np.min(mspec)
    mspec= mspec/np.max(mspec)

    # Perform the center search, and also first dispersion (when 2nd is not asked)
    while iter<3:
        # Select case of search
        k1=1; k2=2
        lk1=vl[1]; lk2=vl[2]
        nk1=int(vs[1]); nk2=int(vs[2])
        if iter==0: # Center search
            x1 = xmid - int(npts*vs[4]/2.0)
            x2 = xmid + int(npts*vs[4]/2.0)+1
            if x1<0: x1=0
            if x2>npts: x2=npts
            k1=0; k2=1
            lk1=vl[0]; lk2=vl[1]
            nk1=int(vs[0]); nk2=0
            if vs[2]<1 and type=='single': nk2=int(vs[1])
        elif iter==1 and type=='single': # 1st/2nd dispersion search (left=right)
            x1 = 0
            x2 = npts
        elif iter==1 and (type=='dual' or type=='left'): # 1st/2nd dispersion search (left)
            x1 = 0
            x2 = xmid
        elif iter==2 and (type=='dual' or type=='right'): # 1st/2nd dispersion search (right)
            x1 = xmid
            x2 = npts
        else: iter=iter+1; continue

        # Search for solution
        corrs=np.zeros([3,nk1+1,nk2+1]); dv1=0; dv2=0;
        for i in range(nk1+1):
            if nk1>0: dv1 = 2.0*(lk1/nk1)*(i-int(nk1/2))
            for j in range(nk2+1):
                if nk2>0: dv2 = 2.0*(lk2/nk2)*(j-int(nk2/2))
                cfi=1.0*cf
                if k1==0: cfi[0] += dv1
                if k2==1: cfi[1:3] = cfi[1:3] + dv2
                if k1==1: cfi[1:3] += dv1
                if k2==2: cfi[3:5] += dv2
                fs = genfreq(npts, cfi, x1, x2)
                ms = np.interp(fs, mfreq, mspec)
                corrs[0,i,j] = np.corrcoef(ms, spec[x1:x2])[1,0]
                corrs[1,i,j] = dv1
                corrs[2,i,j] = dv2
            #Endfor
        #Endfor

        # Assign solution
        imax = np.unravel_index(np.argmax(corrs[0,:,:]), corrs[0,:,:].shape)
        if iter==0:
            cf[0]   += corrs[1,imax[0],imax[1]]
            cf[1:3] += corrs[2,imax[0],imax[1]]
            if vs[2]<1 and (type=='single' or vs[1]<1): iter=3
            else: iter=iter+1
            corr0=corrs
        elif iter==1 and type=='single':
            cf[1:3] += corrs[1,imax[0],imax[1]]
            cf[3:5] += corrs[2,imax[0],imax[1]]
            iter=3
            corr1=corrs
        elif iter==1:
            cf[1]   += corrs[1,imax[0],imax[1]]
            cf[3]   += corrs[2,imax[0],imax[1]]
            iter=iter+1
            corr1=corrs
        else:
            cf[2]   += corrs[1,imax[0],imax[1]]
            cf[4]   += corrs[2,imax[0],imax[1]]
            iter=iter+1
            corr2=corrs
        #Endelse
    #Endwhile
    if not show: return cf

    # Plot the results
    if vs[1]>1 and vs[2]>1 and type=='dual':
        fig = plt.figure(figsize=[11,7])
        gs = fig.add_gridspec(2, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(corr0[1,:,0],corr0[0,:,0])
        ax1.set_title('Center')
        ax1.set_xlabel('Frequency displacement [cm-1]')
        ax2 = fig.add_subplot(gs[0,-2])
        ax2.imshow(corr1[0,:,:])
        ax2.set_title('Left')
        ax2.set_xlabel('2nd dispersion delta')
        ax2.set_ylabel('1st dispersion delta')
        ax3 = fig.add_subplot(gs[0,-1])
        ax3.imshow(corr2[0,:,:])
        ax3.set_title('Right')
        ax3.set_xlabel('2nd dispersion delta')
        ax3.set_ylabel('1st dispersion delta')
        pspec = fig.add_subplot(gs[-1,:])
        py1=0.33; py2=0.38; dpy=0.04
    else:
        fig = plt.figure(figsize=[10,4])
        gs = fig.add_gridspec(1, 1)
        pspec = fig.add_subplot(gs[-1,:])
        py1=0.70; py2=0.78; dpy=0.06
    #Endelse

    # Define events
    def ffit_click(event):
        global ffit_scl, ffit_fs
        if event==0: btn=''
        else: btn = event.inaxes.get_label()
        if btn=='bscl':
            ffit_scl = ffit_scl*2
            if ffit_scl>64: ffit_scl=0.25
            if ffit_scl>=1: bscl.label.set_text('x%d' % ffit_scl)
            else: bscl.label.set_text('x%.2f' % ffit_scl)
        elif btn=='bsav':
            global ffit_cf
            ffit_cf=1.0*cf
            plt.close()
            return
        elif btn=='bc0p' and vs[0]>0: cf[0] += ffit_scl*2.0*vl[0]/vs[0]
        elif btn=='bc0m' and vs[0]>0: cf[0] -= ffit_scl*2.0*vl[0]/vs[0]
        elif btn=='bl1p' and vs[1]>0: cf[1] += ffit_scl*2.0*vl[1]/vs[1]
        elif btn=='bl1m' and vs[1]>0: cf[1] -= ffit_scl*2.0*vl[1]/vs[1]
        elif btn=='br1p' and vs[1]>0: cf[2] += ffit_scl*2.0*vl[1]/vs[1]
        elif btn=='br1m' and vs[1]>0: cf[2] -= ffit_scl*2.0*vl[1]/vs[1]
        elif btn=='bl2p' and vs[2]>0: cf[3] += ffit_scl*2.0*vl[2]/vs[2]
        elif btn=='bl2m' and vs[2]>0: cf[3] -= ffit_scl*2.0*vl[2]/vs[2]
        elif btn=='br2p' and vs[2]>0: cf[4] += ffit_scl*2.0*vl[2]/vs[2]
        elif btn=='br2m' and vs[2]>0: cf[4] -= ffit_scl*2.0*vl[2]/vs[2]
        fs = genfreq(npts, cf)
        ms = np.interp(fs, mfreq, mspec)
        pspec.cla()
        pspec.set_title('%.5f %e %e %e %e - Correlation: %.7f' % (cf[0],cf[1],cf[2],cf[3],cf[4],np.corrcoef(ms,spec)[1,0]), fontsize=10)
        pspec.plot(fs,spec)
        pspec.plot(fs,ms)
        rmax = np.max(abs(spec-ms))
        pspec.plot(fs,(1.0/rmax)*(spec-ms)-1.0,linewidth=0.8)
        dp=int(vs[4]*npts/2)
        if dp>xmid: dp=xmid
        if dp+xmid>=npts: dp = npts-xmid-1
        pspec.plot([fs[xmid-dp],fs[xmid-dp]],[-2,2],linestyle='dotted',color='black')
        pspec.plot([fs[xmid+dp],fs[xmid+dp]],[-2,2],linestyle='dotted',color='black')
        pspec.set_xlim([fs[0],fs[npts-1]])
        pspec.set_ylim([-2.0,1.1])
        plt.draw()
    #End ffit_click event

    # Define events
    global ffit_cf; ffit_cf=1.0*cf
    ffit_click(0)
    plt.tight_layout()
    if show>1:
        bl2p = Button(plt.axes([0.70,py2,0.02,dpy],label='bl2p'),'+'); bl2p.on_clicked(ffit_click)
        bl2m = Button(plt.axes([0.70,py1,0.02,dpy],label='bl2m'),'-'); bl2m.on_clicked(ffit_click)
        bl1p = Button(plt.axes([0.73,py2,0.02,dpy],label='bl1p'),'+'); bl1p.on_clicked(ffit_click)
        bl1m = Button(plt.axes([0.73,py1,0.02,dpy],label='bl1m'),'-'); bl1m.on_clicked(ffit_click)
        bc0p = Button(plt.axes([0.77,py2,0.02,dpy],label='bc0p'),'+'); bc0p.on_clicked(ffit_click)
        bc0m = Button(plt.axes([0.77,py1,0.02,dpy],label='bc0m'),'-'); bc0m.on_clicked(ffit_click)
        br1p = Button(plt.axes([0.81,py2,0.02,dpy],label='br1p'),'+'); br1p.on_clicked(ffit_click)
        br1m = Button(plt.axes([0.81,py1,0.02,dpy],label='br1m'),'-'); br1m.on_clicked(ffit_click)
        br2p = Button(plt.axes([0.84,py2,0.02,dpy],label='br2p'),'+'); br2p.on_clicked(ffit_click)
        br2m = Button(plt.axes([0.84,py1,0.02,dpy],label='br2m'),'-'); br2m.on_clicked(ffit_click)
        bsav = Button(plt.axes([0.88,py2,0.06,dpy],label='bsav'),'OK',color='0.5'); bsav.on_clicked(ffit_click)
        bscl = Button(plt.axes([0.88,py1,0.06,dpy],label='bscl'),'x1',color='0.5'); bscl.on_clicked(ffit_click)
        plt.show()
        fs = genfreq(npts, ffit_cf)
    else:
        plt.show(block=False)
        plt.pause(1.0)
        plt.close()
    #Endelse
    return ffit_cf
#End ffit module

# Spectral alignment module
def spectral(show=False,reset=False):

    # Read clean results
    if reset==2 or not os.path.exists('spec.clean.dat'): clean(show=show,reset=reset)
    fr = open('spec.clean.dat','rb')
    nsets = np.fromfile(fr, dtype=int, count=1)[0]
    szy = np.fromfile(fr, dtype=int, count=1)[0]
    szx = np.fromfile(fr, dtype=int, count=1)[0]
    ma = np.fromfile(fr, dtype=float, count=nsets*szx*szy); ma = np.reshape(ma, [nsets,szy,szx])
    mb = np.fromfile(fr, dtype=float, count=nsets*szx*szy); mb = np.reshape(mb, [nsets,szy,szx])
    fr.close()

    # Get variables
    nodding = cfgread('DATA-NODDING','on')
    order = int(cfgread('SPECTRAL-ORDER','2'))
    rows = cfgread('SPECTRAL-ROWS','all')
    vrange = cfgread('SPECTRAL-FREQUENCY','2920,2950'); vrange=[float(x) for x in vrange.split(',')]
    ref = cfgread('SPECTRAL-REFERENCE','model')
    reftype = cfgread('SPECTRAL-MODEL','radiance')
    vtype = cfgread('SPECTRAL-TYPE','dual')
    vsteps = cfgread('SPECTRAL-STEPS','400,100,100,5,0.5'); vsteps=[float(x) for x in vsteps.split(',')]
    vlimits = cfgread('SPECTRAL-LIMITS','1e-2,1e-4,1e-6'); vlimits=[float(x) for x in vlimits.split(',')]

    # Define reference frames
    datas = np.zeros([nsets,szy,szx])
    rmid = int(szy/2)
    for iset in range(nsets):
        if reftype=='radiance':
            if nodding=='on':
                datas[iset,rmid:szy,:] = ma[iset,rmid:szy,:]
                datas[iset,0:rmid,:] = mb[iset,0:rmid,:]
            else:
                datas = 1.0*mb
            #Endelse
        else:
            if nodding=='on':
                datas[iset,0:rmid,:] = ma[iset,0:rmid,:]-mb[iset,0:rmid,:]
                datas[iset,rmid:szy,:] = mb[iset,rmid:szy,:]-ma[iset,rmid:szy,:]
            else:
                datas = 1.0*(ma-mb)
            #Endelse
        #Endelse
    #Endelse
    data = np.sum(datas,axis=0)

    # Define rows to process
    if rows=='all':
        if reftype=='radiance': irows = range(5,szy-5)
        else:
            prof = np.sum(data,axis=1)
            irows = (prof>np.max(prof)*0.5).nonzero()[0]
        #Endelse
    else: irows = [int(x) for x in rows.split(',')]
    data = data[irows,:]

    # Get telluric model
    if ref=='model':
        mtel = gentrans([min(vrange)-10.0, max(vrange)+10.0], show=show, reset=reset)
        if reftype!='radiance': mtel[1,:]=mtel[2,:]
    else:
        f0 = (vrange[1]+vrange[0])/2.0
        df = (vrange[1]-vrange[0])/szx
        coeff = [f0, df, df, 0.0, 0.0]
        ff = genfreq(szx, coeff)
        mtel = np.zeros([2,szx])
        mtel[0,:] = genfreq(szx, coeff)
        mtel[1,:] = data[0,:]*0.5 + data[1,:]*0.5
    #Endelse

    # Calculate a solution
    vfile = 'spec.spectral.txt'
    if not os.path.exists(vfile) or reset:
        # Obtain a solution close to the data
        f0 = (vrange[1]+vrange[0])/2.0
        df = (vrange[1]-vrange[0])/szx
        coeff = [f0, df, df, 0.0, 0.0]
        if reftype!='radiance':
            ff = genfreq(szx, coeff)
            mt = np.interp(ff, mtel[0,:], mtel[1,:])
            fm = np.polyval(np.polyfit(range(szx),mt,1),range(szx))
            dm = np.polyval(np.polyfit(range(szx),data[0,:],1),range(szx))
            data[0,:] *= fm/dm
        #Endelse
        coeff = ffit(data[0,:], mtel[0,:], mtel[1,:], coeff, vs=[400,0,0,0,1.0], vl=[5.0,0.0,0.0], type=vtype)
        coeff = ffit(data[0,:], mtel[0,:], mtel[1,:], coeff, vs=vsteps, vl=vlimits, type=vtype, show=show)
        vstps = np.asarray(vsteps);  vstps[0:3] = vstps[0:3]/vsteps[3]
        vlims = np.asarray(vlimits); vlims[0:3] = vlims[0:3]/vsteps[3]
        cmid = 1.0*coeff

        # Derive curvature
        fw = open(vfile,'w')
        for i in range(len(irows)):
            if reftype!='radiance':
                ff = genfreq(szx, coeff)
                mt = np.interp(ff, mtel[0,:], mtel[1,:])
                fm = np.polyval(np.polyfit(range(szx),mt,1),range(szx))
                dm = np.polyval(np.polyfit(range(szx),data[i,:],1),range(szx))
                data[i,:] *= fm/dm
            #Endelse
            coeff = ffit(data[i,:], mtel[0,:], mtel[1,:], coeff, vs=vstps, vl=vlims, type=vtype)
            fw.write('%3d %.5f %.5e %.5e %.5e %.5e\n' % (irows[i], coeff[0],coeff[1],coeff[2],coeff[3],coeff[4]))
        #Endfor
        fw.close()
    #Endif

    # Process solution
    va = np.zeros([nsets,szy,szx]);
    vb = np.zeros([nsets,szy,szx]);
    vd = np.zeros([nsets,szy,szx]);
    cfs = np.genfromtxt(vfile)
    cps = np.empty([szy,6])
    cps[:,0] = np.arange(szy)
    cfits = np.empty([order+1,5])
    for i in range(5):
        norder=order
        if i>=1: norder=order-1
        cfits[0:norder+1,i] = np.polyfit(cfs[:,0], cfs[:,1+i], norder)
        cps[:,1+i] = np.polyval(cfits[0:norder+1,i], cps[:,0])
    #Endfor

    f0 = genfreq(szx, cps[rmid,1:6])
    for iset in range(nsets):
        for i in range(szy):
            fi = genfreq(szx, cps[i,1:6])
            if f0[0]>f0[1]:
                fi = np.flip(fi)
                va[iset,i,:] = np.interp(f0, fi, np.flip(ma[iset,i,:]))
                vb[iset,i,:] = np.interp(f0, fi, np.flip(mb[iset,i,:]))
                vd[iset,i,:] = np.interp(f0, fi, np.flip(datas[iset,i,:]))
            else:
                va[iset,i,:] = np.interp(f0, fi, ma[iset,i,:])
                vb[iset,i,:] = np.interp(f0, fi, mb[iset,i,:])
                vd[iset,i,:] = np.interp(f0, fi, datas[iset,i,:])
        #Endfor rows
    #Endfor sets

    # Save frequency solutions
    vsol = 'spec.frequency.txt'
    if not os.path.exists(vsol) or reset:
        fw = open(vsol,'w')
        for iset in range(nsets):
            mms = np.sum(vd[iset,irows,:],axis=0)
            coeff = ffit(mms, mtel[0,:], mtel[1,:], cps[rmid,1:6], vs=vsteps, vl=vlimits, type=vtype)
            fw.write('%3d %.5f %.5e %.5e %.5e %.5e\n' % (iset, coeff[0],coeff[1],coeff[2],coeff[3],coeff[4]))
        #Endfor sets
        fw.close()
    #Endif save solutions

    # Show results
    if show:
        data = np.sum(vd[:,irows,:],axis=0)
        diff = np.zeros([len(irows),szx])
        ref = np.sum(data,axis=0)
        ref = ref/np.median(ref)
        for i in range(len(irows)):
            rdiff = data[i,:]
            rdiff = rdiff/np.median(rdiff)
            diff[i,:] = rdiff - ref
        #Endfor rows
        diff = diff - np.median(diff)
        rms = np.std(diff)
        fig = plt.figure(figsize=[12,6])
        gs  = fig.add_gridspec(2, 5)
        aim = fig.add_subplot(gs[0, :])
        aim.imshow(diff,vmin=-2*rms, vmax=4*rms)
        ax0 = fig.add_subplot(gs[1, 0])
        ax0.plot(cfs[:,0],cfs[:,1],linestyle='none',marker='.')
        ax0.plot(cps[:,0],cps[:,1],color='black')
        ax1 = fig.add_subplot(gs[1, 1])
        ax1.plot(cfs[:,0],cfs[:,2],linestyle='none',marker='.',color='red')
        ax1.plot(cps[:,0],cps[:,2],color='black')
        ax2 = fig.add_subplot(gs[1, 2])
        ax2.plot(cfs[:,0],cfs[:,3],linestyle='none',marker='.',color='red')
        ax2.plot(cps[:,0],cps[:,3],color='black')
        ax3 = fig.add_subplot(gs[1, 3])
        ax3.plot(cfs[:,0],cfs[:,4],linestyle='none',marker='.',color='green')
        ax3.plot(cps[:,0],cps[:,4],color='black')
        ax4 = fig.add_subplot(gs[1, 4])
        ax4.plot(cfs[:,0],cfs[:,5],linestyle='none',marker='.',color='green')
        ax4.plot(cps[:,0],cps[:,5],color='black')
        plt.tight_layout()
        plt.savefig('spec.spectral.png')
        if show>1:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1.0)
            plt.close()
        #Endelse
    #Endif show

    # Save the data
    fw = open('spec.spectral.dat','wb')
    np.asarray([nsets,szy,szx]).tofile(fw)
    va.tofile(fw)
    vb.tofile(fw)
    fw.close()
#End valign module

def gauss(x, *p):
    A, mu, sigma, base = p
    return A*np.exp(-(x-mu)**2/(2.0*sigma**2)) + base
#End gauss model definition

# Store, align and combine section ---------------------------------
def store(show=False,reset=False):

    from scipy.optimize import curve_fit
    if reset==2 or not os.path.exists('spec.spectral.dat'): spectral(show=show,reset=reset)
    fr = open('spec.spectral.dat','rb')
    nsets = np.fromfile(fr, dtype=int, count=1)[0]
    szy = np.fromfile(fr, dtype=int, count=1)[0]
    szx = np.fromfile(fr, dtype=int, count=1)[0]
    ma = np.fromfile(fr, dtype=float, count=nsets*szx*szy); ma = np.reshape(ma, [nsets,szy,szx])
    mb = np.fromfile(fr, dtype=float, count=nsets*szx*szy); mb = np.reshape(mb, [nsets,szy,szx])
    fr.close()

    # Get variables
    nodding = cfgread('DATA-NODDING','on')
    align = cfgread('STORE-ALIGN','y')
    mpeak = cfgread('STORE-PEAKMETHOD','gauss')
    detector = cfgread('DATA-DETECTOR','1.8,12.0,0.1'); dtc=[float(x) for x in detector.split(',')]

    # Identify shifts needed on the frame
    afile = 'spec.align.txt'
    shifts = np.zeros([nsets,3])
    ypix = np.arange(0,szy)
    rmid = int(szy/2)

    if align=='y':
        if not os.path.exists(afile) or reset:
            fw = open(afile,'w')
            for iset in range(nsets):
                pa = np.sum(ma[iset,:,:],axis=1)
                pb = np.sum(mb[iset,:,:],axis=1)
                if nodding=='on':
                    if mpeak=='centroid':
                        pd = pa - pb
                        pd = pd - np.min(pd[2:rmid])
                        ca = int(np.round(np.sum(pd[2:rmid]*ypix[2:rmid])/np.sum(pd[2:rmid]) - (szy/4.0)))
                        pd = pb - pa
                        pd = pd - np.min(pd[rmid:szy-2])
                        cb = int(np.sum(pd[rmid:szy-2]*ypix[rmid:szy-2])/np.sum(pd[rmid:szy-2]) - (szy*3.0/4.0))
                    elif mpeak=='maximum':
                        pd = pa - pb
                        ca = np.argmax(pd[2:rmid])+2 - int(szy/4.0)
                        cb = np.argmax(-pd[rmid:szy-2])+rmid - int(szy*3.0/4.0)
                    elif mpeak=='gauss':
                        pd = pa - pb
                        pm = np.argmax(pd[2:rmid])+2
                        cf,vm = curve_fit(gauss, ypix[2:rmid], pd[2:rmid], p0=[pd[pm],pm,2.0,0.0])
                        ca = int(np.round(cf[1] - (szy/4.0)))
                        pd = pb - pa
                        pm = np.argmax(pd[rmid:szy-2])+rmid
                        cf,vm = curve_fit(gauss, ypix[rmid:szy-2], pd[rmid:szy-2], p0=[pd[pm],pm,2.0,0.0])
                        cb = int(np.round(cf[1] - (szy*3.0/4.0)))
                    #Endelse
                else:
                    cb=0
                    if mpeak=='centroid':
                        pd = pa - pb
                        pd = pd - np.min(pd[2:szy-2])
                        ca = int(np.round(np.sum(pd[2:szy-2]*ypix[2:szy-2])/np.sum(pd[2:szy-2]) - (szy/2.0)))
                    elif mpeak=='maximum':
                        pd = pa - pb
                        ca = np.argmax(pd[2:szy-2]) - int(szy/2.0)
                    elif mpeak=='gauss':
                        pd = pa - pb
                        pm = np.argmax(pd[2:szy-2])+2
                        cf,vm = curve_fit(gauss, ypix[2:szy-2],pd[2:szy-2], p0=[pd[pm],pm,2.0,0.0])
                        ca = int(np.round(cf[1] - szy/2.0))
                    #Endelse
                #Endif nodding
                cmax = int(szy/2)
                if ca>cmax: ca=cmax
                elif ca<-cmax: ca=-cmax
                if cb>cmax: cb=cmax
                elif cb<-cmax: cb=-cmax
                fw.write('%3d %d %d\n' % (iset, ca, cb))
                shifts[iset,:] = [iset,ca,cb]
            #Endfor sets
            fw.close()
        else:
            shifts = np.reshape(np.genfromtxt(afile),[nsets,3])
        #Endelse
    #Endif

    # Shift and combine the frames
    beams = np.zeros([nsets,szy*2,szx])
    noise = np.zeros([nsets,szy*2,szx])
    info = np.reshape(np.genfromtxt('spec.sets.txt',delimiter=',',usecols=[2,3,4,5]),[nsets,4])
    for iset in range(nsets):
        nread = dtc[1]*np.sqrt(info[iset,0]*info[iset,2]/info[iset,3])   # Read noise [e-^2]
        nbeam = dtc[0]*info[iset,0]*(ma[iset,:,:] + mb[iset,:,:])/2.0    # Signal [e-]
        dn = np.abs(nbeam + nread**2)/(2.0*info[iset,0])                 # Noise frame [e-^2] - Factor of 2 empirically derived
        md = ma[iset,:,:] - mb[iset,:,:]
        profiles = np.zeros([3,szy*2])
        if nodding=='on':
            md /= 2.0; dn /= 2.0
            y1 = int(szy*3.0/4.0 - shifts[iset,1])
            y2 = y1 + szy
            beams[iset,y1:y2,:] = md[0:szy,:]
            noise[iset,y1:y2,:] = dn[0:szy,:]
            profiles[0,y1:y2] = np.sum(md[0:szy,:],axis=1)
            profiles[1,y1:y2] = np.sum(md[0:szy,:],axis=1)
            y1 = int(szy/4.0 - shifts[iset,2])
            y2 = y1 + szy
            beams[iset,y1:y2,:] -= md[0:szy,:]
            noise[iset,y1:y2,:] += dn[0:szy,:]
            noise[iset,:,:] = np.sqrt(noise[iset,:,:])/dtc[0]
            profiles[0,y1:y2] -=  np.sum(md[0:szy,:],axis=1)
            profiles[2,y1:y2]  = -np.sum(md[0:szy,:],axis=1)
        else:
            y1 = int(szy/2.0 - shifts[iset,1])
            y2 = y1 + szy
            beams[iset,y1:y2,:] = md[0:szy,:]
            noise[iset,y1:y2,:] = dn[0:szy,:]
            noise[iset,:,:] = np.sqrt(noise[iset,:,:])/dtc[0]
            profiles[0,y1:y2] = np.sum(md[0:szy,:],axis=1)
            profiles[1,y1:y2] = np.sum(md[0:szy,:],axis=1)
        #Endif nodding

        if show:
            fig,ax = plt.subplots(3,figsize=[10,7])
            fig.suptitle('Set %d' % iset)
            ax[0].plot(profiles[0,:],label='A-B combined')
            ax[0].plot(profiles[1,:],label='A-beam')
            ax[0].plot(profiles[2,:],label='B-beam')
            ax[0].set_xlim([0,szy*2])
            ax[0].legend()
            ax[0].set_title('Profiles')
            ax[1].imshow(beams[iset,:,:]); ax[1].set_title('Beam frame')
            ax[2].imshow(noise[iset,:,:]); ax[2].set_title('Noise frame')
            plt.tight_layout()
            plt.savefig('spec.store.%02d.png' % iset)
            if show>1:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(1.0)
                plt.close()
            #Endelse
        #Endif
    #Endfor sets

    # Save the data
    fw = open('spec.store.dat','wb')
    np.asarray([nsets,szy*2,szx]).tofile(fw)
    beams.tofile(fw)
    noise.tofile(fw)
    fw.close()
#End store module


# Flux calibration module ---------------------------------
def fluxcal(show=False,reset=False):

    from scipy.optimize import curve_fit
    if reset==2 or not os.path.exists('spec.store.dat'): store(show=show,reset=reset)
    fr = open('spec.store.dat','rb')
    nsets = np.fromfile(fr, dtype=int, count=1)[0]
    szy = np.fromfile(fr, dtype=int, count=1)[0]
    szx = np.fromfile(fr, dtype=int, count=1)[0]
    beams = np.fromfile(fr, dtype=float, count=nsets*szx*szy); beams = np.reshape(beams, [nsets,szy,szx])
    fr.close()
    rmid = int(szy/2)
    ypix = np.arange(0,szy)

    # Get variables
    smag = float(cfgread('DATA-FLUXCAL','5.0'))
    nodding = cfgread('DATA-NODDING','on')
    dtele = float(cfgread('DATA-DIAMTELE','10.0'))
    detector = cfgread('DATA-DETECTOR','1.8,12.0,0.1'); dtc=[float(x) for x in detector.split(',')]
    slitw = float(cfgread('DATA-SLITWIDTH','0.5'))
    slitl = float(cfgread('DATA-SLITLENGTH','20.0'))
    slitp = float(cfgread('DATA-SLITSCALE','0.0'))
    if slitp==0: slitp = slitl/(szy/2.0)

    # Read sets and frequency information, and telluric model
    info = np.reshape(np.genfromtxt('spec.sets.txt',delimiter=',',usecols=[3,4]),[nsets,2])
    cfs = np.reshape(np.genfromtxt('spec.frequency.txt'),[nsets,6])
    fmid = cfs[int(nsets/2),1]
    mdf = abs(cfs[int(nsets/2),2]+cfs[int(nsets/2),3])/2.0
    freq = genfreq(szx, cfs[int(nsets/2),1:6])
    mtel = gentrans()
    trn = np.interp(freq, mtel[0,:], mtel[2,:])
    mtrn = np.average(trn)

    # Photometry table
    # names V        I     J     H     K     L     L'    M                       N                       Q
    lms = [0.5556, 0.90, 1.25, 1.65, 2.20, 3.45, 3.80, 4.80, 7.80, 8.70, 9.80, 10.1, 10.3, 11.6, 12.5, 20.0] # Band wavelength [um]
    jys = [3540,   2250, 1600, 1020, 657,  290,  252,  163,  65.3, 53.0, 42.3, 39.8, 38.5, 30.5, 26.4, 10.4] # Band intensity for Vega [Jy]
    filter = int(np.round(np.interp(1e4/fmid,lms,np.arange(0,len(lms)))))
    cs = 2.99792458e10          # Speed of light [cm/s]
    hp = 6.6260693E-34          # Speed of light [W/s2]
    sflux  = jys[filter]        # Band intensity for Vega [Jy]
    sflux *= 1e-26              # Band intensity for Vega [W/m2/Hz] http://en.wikipedia.org/wiki/Jansky
    sflux *= cs                 # Band intensity for Vega [W/m2/cm-1]
    sflux *= 10.0**(-0.4*smag)  # Band intensity for object [W/m2/cm-1]
    cflux  = sflux*np.pi*((dtele/2)**2)*mdf # Stellar radiant flux [W]
    cflux *= (1.0/fmid)/(cs*hp*dtc[0]) # Radiant flux [ADU/s]

    # Iterate through sets
    npix = 20; nscl = 2
    fcal = np.zeros([nsets,3,4]); sk=['T','A','B']
    if show and nodding=='on':
        fig,ax=plt.subplots(1,3,figsize=[10,5],sharey=True)
        ax[0].set_title('Combined beams')
        ax[1].set_title('A beam')
        ax[2].set_title('B beam')
        plt.tight_layout()
    elif show:
        fig,ax=plt.subplots(1,1,figsize=[6,5])
        ax.set_title('Stellar beam')
        plt.tight_layout()
    #Endelse

    fw = open('spec.fluxcal.txt','w')
    fw.write('# -----------------------------------------\n')
    fw.write('# Gamma and efficiency table\n')
    fw.write('# -----------------------------------------\n')
    fw.write('# Mid-frequency [cm-1]: %f\n' % fmid)
    fw.write('# Delta-frequency [cm-1]: %f\n' % mdf)
    fw.write('# Mean transmittance: %f\n' % mtrn)
    fw.write('# Expected stellar flux [W/m2/cm-1]: %e\n' % sflux)
    fw.write('# Expected stellar counts [ADU/s]: %e\n' % cflux)
    fw.write('# -----------------------------------------\n')
    fw.write('#  Seeing Losses    Gamma     Efficiency\n')
    fw.write('# -----------------------------------------\n')

    for iset in range(nsets):
        prof = np.sum(beams[iset,:,:],axis=1)
        for k in range(3):
            # Obtain photometric values of peak
            if k==0: yc=rmid
            elif k==1 and nodding=='on': yc = int(szy/4.0)
            elif k==2 and nodding=='on': yc = int(szy*3.0/4.0)
            else: continue
            xs = ypix[yc-npix:yc+npix]-yc
            if k==0: ys = prof[yc-npix:yc+npix]
            else: ys = -2.0*prof[yc-npix:yc+npix]
            cf,vm = curve_fit(gauss, xs,ys, p0=[np.max(ys),0.0,2.0,0.0])

            # Calculate slit-losses
            img = np.zeros([npix*nscl*2+1,npix*nscl*2+1])
            tflux=0.0; tinslit=0.0;
            for i in range(0,npix*nscl*2+1):
                for j in range(0,npix*nscl*2+1):
                    xp = (i-npix*nscl)/nscl
                    yp = (j-npix*nscl)/nscl
                    rp = np.sqrt(xp*xp + yp*yp)
                    fp = np.exp(-rp**2/(2.0*cf[2]**2))
                    tflux += fp
                    if abs(xp)*slitp<slitw: tinslit += fp
                #Endfor j
            #Endfor i
            slitloss = tinslit/tflux
            seeing = cf[2]*slitp*2.355

            # Calculate Gammas and efficiencies
            cps = cf[0]*cf[2]*np.sqrt(2.0*np.pi)/(info[iset,0]*info[iset,1]*szx)
            gamma = sflux*mtrn*slitloss/cps
            efficiency = cps/(cflux*mtrn*slitloss)
            fcal[iset,k,:] = [seeing, slitloss, gamma, efficiency]
            fw.write('%d %s %.3f %.4f %e %.5f\n' % (iset, sk[k], seeing, slitloss, gamma, efficiency))

            if show and nodding=='on': ax[k].plot(xs-cf[1],ys,label='Set %d' % iset)
            elif show: ax.plot(xs-cf[1],ys,label='Set %d' % iset)
        #Endfor
    #Endfor
    fw.write('# -----------------------------------------\n')
    if nodding=='on': ind=[1,2]
    else: ind=[0]
    fw.write('# Median seeing [arcsec]: %.3f\n' % np.median(fcal[:,ind,0]))
    fw.write('# Median gamma [W/m2/cm-1/ADU/sec]: %e\n' % np.median(fcal[:,ind,2]))
    fw.write('# Median efficiency: %.5f\n' % np.median(fcal[:,ind,3]))
    fw.write('# -----------------------------------------\n')
    fw.close()

    # Show and save plot
    if show:
        plt.savefig('spec.fluxcal.png')
        if show>1:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(2.0)
            plt.close()
        #Endelse
    #Endif
#End fluxcal

# Clicking function
def extract_click(event):
    global extract_mask
    if event.xdata is None: return
    extract_mask.append(event.xdata)
    plt.plot([event.xdata,event.xdata],[-1e6,1e6],linestyle='dotted',linewidth=0.7,color='black')
    plt.draw()
    print(event.xdata)
#End click event

# Q-curve cometary emission profiles analysis ---------------------------------
def qscale(show=False,reset=False):

    import matplotlib.gridspec as gridspec
    from matplotlib.widgets import Button
    if reset==2 or not os.path.exists('spec.store.dat'): store(show=show,reset=reset)
    fr = open('spec.store.dat','rb')
    nsets = np.fromfile(fr, dtype=int, count=1)[0]
    szy = np.fromfile(fr, dtype=int, count=1)[0]
    szx = np.fromfile(fr, dtype=int, count=1)[0]
    beams = np.fromfile(fr, dtype=float, count=nsets*szx*szy); beams = np.reshape(beams, [nsets,szy,szx])
    noise = np.fromfile(fr, dtype=float, count=nsets*szx*szy); noise = np.reshape(noise, [nsets,szy,szx])
    fr.close()
    rmid = int(szy/2)
    xpix = np.arange(0,szx)

    # Get variables
    sets = cfgread('EXTRACT-SETS','all')
    rows = cfgread('EXTRACT-ROWS','all').split(',')
    mask = cfgread('EXTRACT-MASK','').split(',')
    qmask = cfgread('QSCALE-MASK','').split(',')
    shift = int(cfgread('QSCALE-SHIFT','0'))
    offset = float(cfgread('QSCALE-OFFSET','0'))
    regions = cfgread('QSCALE-REGIONS','1,5,1'); regions=[int(x) for x in regions.split(',')]
    align = cfgread('STORE-ALIGN','y')
    nodding = cfgread('DATA-NODDING','on')
    slitw = float(cfgread('DATA-SLITWIDTH','0.5'))
    slitl = float(cfgread('DATA-SLITLENGTH','20.0'))
    slitp = float(cfgread('DATA-SLITSCALE','0.0'))
    seeing = float(cfgread('DATA-SEEING','0.8'))
    if slitp==0: slitp = slitl/(szy/2.0)
    dslitwp = int(np.round((slitw/slitp))/2.0)
    slitwp = 2*dslitwp+1

    # Stack all sets
    cfs = np.reshape(np.genfromtxt('spec.frequency.txt'),[nsets,6])
    if sets=='all': sets=np.arange(0,nsets)
    else: sets=[int(x) for x in sets.split(',')]
    beams = np.sum(beams[sets,:,:],axis=0)/len(sets)
    noise = np.sqrt(np.sum(noise[sets,:,:]**2,axis=0))/len(sets)
    rset=int(len(sets)/2)
    fq = genfreq(szx, cfs[rset,1:6])
    fmin = np.min(fq); fmax = np.max(fq)
    mtel = gentrans([fmin-3,fmax+3],iset=-1)
    trn = np.interp(fq, mtel[0,:], mtel[2,:]) + 1e-4

    # Define rows
    if align=='y': shifts = np.reshape(np.genfromtxt('spec.align.txt'),[nsets,3])
    else: shifts = np.zeros([nsets,3])
    if nodding=='on':
        rmin = int(np.round(szy*(3.0/8.0) - np.min(shifts[:,1]) + 2))
        rmax = int(np.round(szy*(5.0/8.0) - np.max(shifts[:,2]) - 2))
    else:
        rmin = int(np.round(szy*(1.0/4.0) - np.min(shifts[:,1]) + 2))
        rmax = int(np.round(szy*(3.0/4.0) - np.max(shifts[:,1]) - 2))
    #Endelse

    # Process extract mask
    if mask[0]=='ask': mask=[]
    wmask = np.zeros(szx)+1.0
    for i in range(0,int(len(mask)),2): wmask[int(mask[i]):int(mask[i+1])+1] = 0.0

    # Process the rows
    beams = beams[rmin:rmax+1,:]
    noise = noise[rmin:rmax+1,:]
    rmid = rmid - rmin
    szy = rmax-rmin+1
    data = np.zeros([3,szy,szx])
    for irow in range(szy):
        md = beams[irow,:]
        nd = noise[irow,:]
        fit = np.polyval(np.polyfit(xpix, md/trn, 3, w=trn*wmask), xpix)
        data[0,irow,:] = md
        data[1,irow,:] = nd
        data[2,irow,:] = fit*trn
    #Endfor rows

    # Process molecular mask
    wqmask = np.zeros(szx)
    if qmask[0]!='lines' and qmask[0]!='ask':
        for i in range(0,int(len(qmask)),2): wqmask[int(qmask[i]):int(qmask[i+1])+1] = 1.0
    #Endif
    if show or qmask[0]=='lines' or qmask[0]=='ask':
        md = np.sum(data[0,rmid-5:rmid+5,:],axis=0)
        nd = np.sqrt(np.sum(data[1,rmid-5:rmid+5,:]**2,axis=0))
        mm = np.sum(data[2,rmid-5:rmid+5,:],axis=0)
        dd = md-mm
        if qmask[0]=='lines':
            if len(qmask)>1: sigma=float(qmask[1])
            else: sigma=5.0
            ind = (dd>(sigma*nd)).nonzero()[0]
            xlast = ind[0]; str=''
            for i in range(0,len(ind)):
                if i==len(ind)-1: str='%s%d,%d,' % (str,xlast,ind[i])
                elif (ind[i+1]-ind[i])>1: str='%s%d,%d,' % (str,xlast,ind[i]); xlast=ind[i+1]
            #Endfor
            if show>1: print('Lines: %s' % str[:-1])
            wqmask[ind] = 1.0
        #Endif
        if show or qmask[0]=='ask':
            fig, ax = plt.subplots(1,1,figsize=[10,5])
            plt.plot(fq,dd,linewidth=0.7)
            plt.plot(fq, nd,color='green',label='1.0 sigma')
            plt.plot(fq,-nd,color='green')
            if qmask[0]=='lines': plt.plot(fq,sigma*nd,label='%.1f sigma' % sigma)
            plt.xlim([np.min(fq),np.max(fq)])
            plt.ylim([-np.max(nd)*5,np.max(dd)+np.max(nd)*5])
            plt.xlabel('Frequency [cm-1]')
            plt.tight_layout()
            ind = (wqmask>0).nonzero()
            plt.plot(fq[ind],dd[ind],'.',linestyle='',color='red')
            plt.legend()
            if qmask[0]=='ask':
                fig.canvas.mpl_connect('button_press_event', extract_click)
                plt.show()
                str=''; extract_mask.sort()
                if fq[0]>fq[1]: extract_mask.reverse()
                for ip in extract_mask:
                    if fq[0]>fq[1]: pix = np.interp(ip,np.flip(fq),np.flip(xpix))
                    else: pix = np.interp(ip,fq,xpix)
                    print(ip,pix)
                    if pix<0 or pix>=szx: continue
                    if len(str)==0: str='%d' % pix
                    else: str='%s,%d' % (str,pix)
                #Endfor
                print('Lines: %s' % str)
                exit()
            elif show>1:
                plt.savefig('spec.qscalef.png')
                plt.show()
            else:
                plt.savefig('spec.qscalef.png')
                plt.show(block=False)
                plt.pause(1.0)
                plt.close()
            #Endelse
        #Endif
    #Endif

    # Compute weighted profiles
    cnt = np.zeros(szy); cwt=0.0
    gas = np.zeros(szy); gwt=0.0
    for x in range(szx):
        if wmask[x]>0:
            vl = data[0,:,x]
            wt = (vl/data[1,:,x])**2
            cnt += wt*vl
            cwt += wt
        #Endif
        if wqmask[x]>0:
            vl = data[0,:,x]-data[2,:,x]
            wt = (vl/data[1,:,x])**2
            gas += wt*vl
            gwt += wt
        #Endif
    #Endfor
    cnt = cnt/cwt
    gas = gas/gwt
    cmax = np.argmax(cnt)
    gmax = np.argmax(gas)
    if show>1: print('Peaks offset: %d' % (gmax-cmax))

    # Compute synthetic profiles
    npix = szy*2+1
    img = np.zeros([npix,npix])
    smask = np.zeros([npix,npix])
    for i in range(0,npix):
        for j in range(0,npix):
            xp = (i-szy)*slitp
            yp = (j-szy)*slitp
            rp = np.sqrt(xp*xp + yp*yp)
            if rp==0: fp=1.0
            else: fp = slitp/rp
            #fp *= np.exp(-rp/4.0)
            img[j,i] = fp
            if abs(xp)<slitw/2.0: smask[j,i]=1.0
    nker = int(2.0*seeing/slitp)
    kpix = 2*nker+1
    xker = np.arange(kpix)-nker
    ker = np.zeros([kpix,kpix])
    for i in range(0,kpix):
        for j in range(0,kpix):
            xp = (i-nker)*slitp
            yp = (j-nker)*slitp
            rp = np.sqrt(xp*xp + yp*yp)
            ker[j,i] = np.exp(-rp**2/(2.0*(seeing/2.355)**2))
    ktot = np.sum(ker)
    cnv = 1.0*img
    for i in range(nker,npix-nker):
        for j in range(nker,npix-nker):
            cnv[i,j] = np.sum(img[i-nker:i+nker+1,j-nker:j+nker+1]*ker)/ktot
    pf = np.sum(img*smask,axis=1)
    pt = np.sum(cnv*smask,axis=1)
    xs = np.arange(npix)-szy

    # Normalize profiles based on terminal region
    y1 =  regions[0]*slitwp - dslitwp
    y2 =  regions[1]*slitwp + dslitwp+1
    y3 = -regions[1]*slitwp - dslitwp
    y4 = -regions[0]*slitwp + dslitwp+1
    tt = np.sum(gas[gmax+shift+y1:gmax+shift+y2]+gas[gmax+shift+y3:gmax+shift+y4])/slitwp + offset; gas/=tt
    tt = np.sum(cnt[cmax+y1:cmax+y2]+cnt[cmax+y3:cmax+y4])/slitwp + offset; cnt/=tt
    tt = np.sum(pf[szy+y1:szy+y2]+pf[szy+y3:szy+y4])/slitwp + offset; pf/=tt
    tt = np.sum(pt[szy+y1:szy+y2]+pt[szy+y3:szy+y4])/slitwp + offset; pt/=tt

    # Perform the Q-curve analysis
    nspix = int(np.min([gmax-shift,szy-(gmax-shift)])/slitwp)
    qprof = np.zeros([5,nspix])
    for i in range(nspix):
        y1 = gmax+shift + i*slitwp - dslitwp
        y2 = gmax+shift + i*slitwp + dslitwp+1
        qprof[0,i] = np.sum(gas[y1:y2])/slitwp + offset # Right side
        y1 = gmax+shift - i*slitwp - dslitwp
        y2 = gmax+shift - i*slitwp + dslitwp+1
        qprof[1,i] = np.sum(gas[y1:y2])/slitwp + offset # Left side
        qprof[2,i] = 0.5*(qprof[0,i]+qprof[1,i])        # Symmetrized values
        y1 = szy - i*slitwp - dslitwp
        y2 = szy - i*slitwp + dslitwp+1
        qprof[3,i] = np.sum(pf[y1:y2])/slitwp # Synthetic
        qprof[4,i] = np.sum(pt[y1:y2])/slitwp # Synthetic convolved by seeing
    #Endfor
    qcnt = qprof[2,0]; qtot = qprof[3,0]; qcnv = qprof[4,0]
    for i in range(1,regions[2]+1): qcnt += 2.0*qprof[2,i]; qtot += 2.0*qprof[3,i]; qcnv += 2.0*qprof[4,i]
    qscl = qtot/qcnt
    qmod = qtot/qcnv

    # Show results
    if show:
        fig = plt.figure(figsize=[14,5])
        gs = fig.add_gridspec(2, 5)
        im1 = fig.add_subplot(gs[0, 0]); im1.imshow(np.log10(img),vmin=-2,vmax=0); im1.set_axis_off(); im1.set_title('Cometary image')
        im2 = fig.add_subplot(gs[1, 0]); im2.imshow(np.log10(cnv),vmin=-2,vmax=0); im2.set_axis_off(); im2.set_title('Seeing convolved')
        ppl = fig.add_subplot(gs[:,1:3])

        pref = np.max(pt)
        ypix = np.arange(szy)-cmax-shift
        ppl.step(ypix/slitwp,gas+offset,label='Gas')
        ppl.plot(ypix/slitwp,cnt,linestyle=':',label='Continuum')
        ppl.plot(xker/slitwp,ker[nker,:]*pref,label='Seeing')
        ppl.plot(xs/slitwp,pt,label='Model')
        ppl.plot(xs/slitwp,pf,label='Model TOA', linestyle='--')
        ppl.legend()
        ppl.set_xlim([(0-rmid)/slitwp,(szy-rmid)/slitwp]); ppl.set_xlabel('Slitwide pixels along the slit (x%d pixels)' % slitwp)
        ax2=ppl.twiny(); ax2.set_xlim([0-rmid,szy-rmid]); ax2.set_xlabel('Pixels along the slit (%.3f arcsec/pixel)' % slitp)
        ppl.set_ylim([1e-2*pref,3.0*pref])
        ppl.set_yscale('log')

        qpl = fig.add_subplot(gs[:,3:5])
        qpl.errorbar(range(nspix),qprof[2,:]/qprof[3,:],xerr=-0.5,label='Symmetrized',fmt='o')
        qpl.plot(qprof[0,:]/qprof[3,:],'o',label='Right')
        qpl.plot(qprof[1,:]/qprof[3,:],'o',label='Left')
        qpl.plot(qprof[4,:]/qprof[3,:],'o',label='Model')
        qpl.plot(xs[szy:]/slitwp,pt[szy:]/pf[szy:])
        qpl.set_title('Q-curve analysis')
        qpl.set_xlim([-0.5,nspix]); qpl.set_xlabel('Slitwide pixels along the slit (x%d pixels)' % slitwp)
        qpl.set_ylim([0,2])
        qpl.text(2,0.2,'Qscale Q(0-%d)/Q(%d-%d): %.4f' % (regions[2],regions[0],regions[1],qscl))
        qpl.text(2,0.1,'Qscale expected: %.4f' % qmod)
        qpl.legend()

        plt.tight_layout()
        plt.savefig('spec.qscale.png')
        if show>1:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(2.0)
            plt.close()
        #Endelse
    #Endif
#End qscale

# Geometry module ---------------------------------
geo_offset=0; geo_shift=0; geo_scl=0
def geometry(show=False,reset=False):

    if show==0: return
    from matplotlib import image
    from matplotlib.widgets import Button
    global geo_offset, geo_shift, geo_scl
    if reset==2 or not os.path.exists('spec.store.dat'): store(show=show,reset=reset)
    fr = open('spec.store.dat','rb')
    nsets = np.fromfile(fr, dtype=int, count=1)[0]
    szy = np.fromfile(fr, dtype=int, count=1)[0]
    szx = np.fromfile(fr, dtype=int, count=1)[0]
    beams = np.fromfile(fr, dtype=float, count=nsets*szx*szy); beams = np.reshape(beams, [nsets,szy,szx])
    fr.close()
    rmid = int(szy/2)

    # Get variables
    psgserver = cfgread('SPECTRAL-SERVER','https://psg.gsfc.nasa.gov')
    psgkey = cfgread('SPECTRAL-PSGKEY','')
    if len(psgkey)>0: psgserver = '-d key=%s %s' % (psgkey, psgserver)
    align = cfgread('STORE-ALIGN','y')
    object = cfgread('DATA-OBJECT','Mars')
    nodding = cfgread('DATA-NODDING','on')
    diameter = float(cfgread('DATA-OBJECT-DIAMETER','6779')) # [km]
    slitw = float(cfgread('DATA-SLITWIDTH','0.5'))
    slitl = float(cfgread('DATA-SLITLENGTH','20.0'))
    slitp = float(cfgread('DATA-SLITSCALE','0.0'))
    seeing = float(cfgread('DATA-SEEING','0.8'))
    if slitp==0: slitp = slitl/(szy/2.0)
    if nodding=='on': slitl/=2.0

    # Compute ephemeris
    if reset or not os.path.exists('spec.ephm.txt'):
        fr = open('spec.sets.txt','r'); dates=fr.readlines(); fr.close()
        dates = [x.split(',')[0] for x in dates]
        fe = open('spec.ephm.txt','w')
        for i in range(nsets):
            fw = open('spec.ephm.cfg','w')
            fw.write('<OBJECT-NAME>%s\n' % object)
            fw.write('<OBJECT-DATE>%s\n' % dates[i].replace('-','/'))
            fw.close()
            os.system('curl -s -d type=cfg -d wephm=y --data-urlencode file@spec.ephm.cfg %s/api.php > spec.ephm.out' % psgserver)
            fr = open('spec.ephm.out','r'); lines=fr.readlines(); fr.close()
            os.system('rm spec.ephm.cfg spec.ephm.out')
            for line in lines:
                if line.find('GEOMETRY-OBS-ALTITUDE')>0: delta=float(line.split('>')[1])  # 0
                if line.find('OBJECT-STAR-DISTANCE')>0: rh=float(line.split('>')[1])      # 1
                if line.find('OBJECT-OBS-VELOCITY')>0: vdelta=float(line.split('>')[1])   # 2
                if line.find('OBJECT-STAR-VELOCITY')>0: vrh=float(line.split('>')[1])     # 3
                if line.find('OBJECT-OBS-LONGITUDE')>0: olon=float(line.split('>')[1])    # 4
                if line.find('OBJECT-OBS-LATITUDE')>0: olat=float(line.split('>')[1])     # 5
                if line.find('OBJECT-SOLAR-LONGITUDE')>0: slon=float(line.split('>')[1])  # 6
                if line.find('OBJECT-SOLAR-LATITUDE')>0: slat=float(line.split('>')[1])   # 7
                if line.find('OBJECT-SEASON')>0: season=float(line.split('>')[1])         # 8
            #Endfor
            fe.write('%f %f %f %f %f %f %f %f %f\n' % (delta,rh,vdelta,vrh,olon,olat,slon,slat,season))
        #Endfor
        fe.close()
    #Endif

    # Prepare ancilliary information
    if os.path.exists('spec.geometry.txt'): slitpos = np.reshape(np.genfromtxt('spec.geometry.txt'),[nsets,3])
    else: slitpos = np.zeros([nsets,3])
    if align=='y': shifts = np.reshape(np.genfromtxt('spec.align.txt'),[nsets,3])
    else: shifts = np.zeros([nsets,3])
    ephm = np.reshape(np.genfromtxt('spec.ephm.txt'),[nsets,9])
    if not os.path.exists('spec.albedo.png'): os.system('curl -s https://psg.gsfc.nasa.gov/data/objects/maps/%s%s.png > spec.albedo.png' % (object[0].upper(), object[1:]))
    if os.path.getsize('spec.albedo.png')<1e4: os.system('curl -s https://psg.gsfc.nasa.gov/data/objects/maps/Object.png > spec.albedo.png')
    albedo = image.imread('spec.albedo.png')

    # Create seeing kernel
    nker = int(2.0*seeing/slitp)
    kpix = 2*nker+1
    ker = np.zeros(kpix)
    for i in range(0,kpix):
        xp = (i-nker)*slitp
        ker[i] = np.exp(-xp**2/(2.0*(seeing/2.355)**2))
    #Endfor
    ker = ker/np.sum(ker)

    # Iterate through sets
    for iset in range(nsets):
        # Calculate observed profile
        if nodding=='on':
            rmin = int(np.round(szy*(3.0/8.0) - shifts[iset,1] + 2))
            rmax = int(np.round(szy*(5.0/8.0) - shifts[iset,2] - 2))
        else:
            rmin = int(np.round(szy*(1.0/4.0) - shifts[iset,1] + 2))
            rmax = int(np.round(szy*(3.0/4.0) - shifts[iset,1] - 2))
        #Endelse
        yprof = np.sum(beams[iset,rmin:rmax,:],axis=1)
        yprof /= np.average(yprof)
        xprof = (np.arange(rmin,rmax) - rmid)*slitp

        # Create image of the object
        geo_offset = slitpos[iset,0]; geo_shift = slitpos[iset,1]; geo_scl = 1.0
        oblon = ephm[iset,4]; oblat = ephm[iset,5]; sllon = ephm[iset,6]; sllat = ephm[iset,7]; npix = 200
        img = np.zeros([npix,npix,4])
        for x in range(npix):
            for y in range(npix):
                px = (x-(npix/2.0))/(npix/2.0)*1.2
                py = (y-(npix/2.0))/(npix/2.0)*1.2
                ro = np.sqrt(px*px + py*py)
                if ro>1: continue
                if ro==0.0: ro=1e-9
                c = np.arcsin(ro)
                k = 180.0/np.pi

                # Calculate physical longitude
                lon = oblon + 180.0 + k*np.arctan2(px*np.sin(c), (ro*np.cos(-oblat/k)*np.cos(c) - py*np.sin(-oblat/k)*np.sin(c)))
                lat = k*np.arcsin(np.cos(c)*np.sin(-oblat/k) + py*np.sin(c)*np.cos(-oblat/k)/ro)
                if lat>=90: lat=89
                if lat>=90: exit()

                # Calculate illumination
                localtime = (lon-180.0-sllon)/k
                coschi = np.sin(lat/k)*np.sin(-sllat/k) + np.cos(lat/k)*np.cos(-sllat/k)*np.cos(localtime)
                if coschi<0.0: coschi = 0.0

                # Assign value from albedo map
                if lon<0.0: lon=lon+360.0
                if lon>=360.0: lon=lon-360.0
                ix = int(lon*albedo.shape[1]/360.0)
                iy = int((lat+90)*albedo.shape[0]/180.0)
                img[y,x,0:3] = albedo[iy,ix,0:3]*coschi
                img[y,x,3] = 1
            #End y-loop
        #End x-loop
        adiam = 1.2*1.3788e-3*diameter/ephm[iset,0] # Diameter of planet and border [arcsec]
        scl = npix/adiam                    # Scaler for image [pix/arcsec]
        rbox = np.zeros([4,2])
        fig,ax=plt.subplots(1,2,figsize=[10,5])
        slit = plt.Polygon(rbox, fill=False, color="lime")
        ax[0].add_patch(slit)
        ax[0].imshow(img)
        ax[0].set_axis_off()

        def geometry_click(event):
            global geo_offset, geo_shift, geo_scl
            if event==0: btn=''
            else: btn = event.inaxes.get_label()
            if btn=='bsc':
                geo_scl = geo_scl*2
                if geo_scl>64: geo_scl=1
                bsc.label.set_text('x%d' % geo_scl)
            elif btn=='bxp': geo_offset += slitp*geo_scl
            elif btn=='bxm': geo_offset -= slitp*geo_scl
            elif btn=='byp': geo_shift  += slitp*geo_scl
            elif btn=='bym': geo_shift  -= slitp*geo_scl
            elif btn=='bok':
                geo_scl = 0
                plt.close('all')
            #Endif
            sbox = np.zeros([4,2])
            sbox[0,0] = (-slitw/2.0 + geo_offset)*scl
            sbox[0,1] = (-slitl/2.0 - geo_shift)*scl
            sbox[2,0] = sbox[0,0] + slitw*scl
            sbox[2,1] = sbox[0,1] + slitl*scl
            sbox[1,0] = sbox[2,0]; sbox[1,1] = sbox[0,1]
            sbox[3,0] = sbox[0,0]; sbox[3,1] = sbox[2,1]
            rc = np.cos(slitpos[iset,2]*np.pi/180.0); rs = np.sin(slitpos[iset,2]*np.pi/180.0)
            for i in range(4): rbox[i,0] = sbox[i,0]*rc - sbox[i,1]*rs + npix/2; rbox[i,1]= sbox[i,0]*rs + sbox[i,1]*rc + npix/2
            slit.xy = rbox

            ymod = np.zeros(szy)
            xmod = (np.arange(szy)-rmid)*slitp
            for i in range(szy):
                xp = -xmod[i]
                yp = geo_offset
                rc = np.cos(slitpos[iset,2]*np.pi/180.0); rs = np.sin(slitpos[iset,2]*np.pi/180.0)
                xr = int((xp*rc - yp*rs)*scl + npix/2)
                yr = int((xp*rs + yp*rc)*scl + npix/2)
                if xr>=0 and xr<npix and yr>=0 and yr<npix: ymod[i] = np.sum(img[xr,yr,0:3])/3.0
            #Endfor
            ycnv = 0.0*ymod
            for i in range(nker,szy-nker):
                for j in range(0,kpix):
                    ycnv[i] += ymod[i+j-nker]*ker[j]
                #Endfor
            #Endfor
            tt = np.interp(xprof+geo_shift, xmod, ycnv)
            ycnv /= np.average(tt)

            ax[1].cla()
            ax[1].plot(xprof + geo_shift, yprof)
            ax[1].plot(xmod, ycnv)
            ax[1].set_xlabel('Arcseconds from center')
            ax[1].set_ylabel('Normalized intensity')
            ax[1].set_title('Set %d - Offsets: %.3f,%.3f arcsecs' % (iset, geo_offset, geo_shift))
            plt.draw()
        #Click image

        if nsets==0: fpng = 'spec.geometry.png'
        else: fpng = 'spec.geometry.%02d.png' % iset
        geometry_click(0)
        if show>1:
            plt.tight_layout()
            bxp = Button(plt.axes([0.12,0.12,0.04,0.05],label='bxp'),r'$\rightarrow$');  bxp.on_clicked(geometry_click)
            bxm = Button(plt.axes([0.02,0.12,0.04,0.05],label='bxm'),r'$\leftarrow$');  bxm.on_clicked(geometry_click)
            bok = Button(plt.axes([0.07,0.12,0.04,0.05],label='bok'),'OK',color='green'); bok.on_clicked(geometry_click)
            byp = Button(plt.axes([0.07,0.19,0.04,0.05],label='byp'),r'$\uparrow$');  byp.on_clicked(geometry_click)
            bym = Button(plt.axes([0.07,0.05,0.04,0.05],label='bym'),r'$\downarrow$');  bym.on_clicked(geometry_click)
            bsc = Button(plt.axes([0.12,0.05,0.04,0.05],label='bsc'),'x1',color='1.0'); bsc.on_clicked(geometry_click)
            plt.savefig(fpng)
            plt.show()
            if geo_scl==0: slitpos[iset,0] = geo_offset; slitpos[iset,1] = geo_shift;
        else:
            plt.tight_layout()
            plt.savefig(fpng)
            plt.show(block=False)
            plt.pause(1.0)
            plt.close()
        #Endelse
    #Endfor
    fw = open('spec.geometry.txt','w')
    for iset in range(nsets): fw.write('%.3f %.3f %.1f\n' % (slitpos[iset,0],slitpos[iset,1],slitpos[iset,2]))
    fw.close()
#End geometry module

# Spectral extract module ---------------------------------
global nbase, ngain, fitstr, basetrn, basestr, baseline, basegain, basefringe, basefringefq
def removefringe(x, *p):
    cf = p
    global basetrn, basegain, basefringe, basefringefq
    basefringe = basetrn*cf[0]*np.sin(x*basefringefq + cf[1])
    return basefringe
#End removefringe

def basefit(x, *p):
    cf = p
    global basetrn, basestr, nbase, ngain, fitstr, baseline, basegain
    if nbase>=0: baseline=np.polyval(cf[0:nbase+1],x)
    else: baseline=0.0
    if ngain>=0: basegain=np.polyval(cf[nbase+1:nbase+ngain+2],x)
    else: basegain=1.0
    if fitstr=='y': baseline += cf[nbase+ngain+2]*basestr*basetrn*basegain
    return baseline + basegain*basetrn
#End removebaseline function

extract_mask=[]
def extract(show=False,reset=False):

    from scipy.optimize import curve_fit
    from scipy.signal import lombscargle
    global basetrn, basestr, nbase, ngain, fitstr, baseline, basegain, basefringe, basefringefq
    if reset==2 or not os.path.exists('spec.store.dat'): store(show=show,reset=reset)
    fr = open('spec.store.dat','rb')
    nsets = np.fromfile(fr, dtype=int, count=1)[0]
    szy = np.fromfile(fr, dtype=int, count=1)[0]
    szx = np.fromfile(fr, dtype=int, count=1)[0]
    beams = np.fromfile(fr, dtype=float, count=nsets*szx*szy); beams = np.reshape(beams, [nsets,szy,szx])
    noise = np.fromfile(fr, dtype=float, count=nsets*szx*szy); noise = np.reshape(noise, [nsets,szy,szx])
    fr.close()
    rmid = int(szy/2)
    xpix = np.arange(0,szx)

    # Get variables
    sets = cfgread('EXTRACT-SETS','all')
    stack = cfgread('EXTRACT-STACK','y')
    rows = cfgread('EXTRACT-ROWS','all').split(',')
    mask = cfgread('EXTRACT-MASK','').split(',')
    fsys = cfgread('EXTRACT-SYSTEMATICS','')
    resiscl = float(cfgread('EXTRACT-SCALER','1.0'))
    minsnr = float(cfgread('EXTRACT-MINSNR','0.0'))
    ngain = int(cfgread('EXTRACT-FITGAIN','2'))
    if ngain=='n': ngain=-1
    else: ngain = int(ngain)
    nbase = cfgread('EXTRACT-REMOVEOFFSET','-1')
    if nbase=='n': nbase=-1
    else: nbase = int(nbase)
    nfringe = cfgread('EXTRACT-REMOVEFRINGE','0')
    if nfringe=='n': nfringe=0
    else: nfringe = int(nfringe)
    fitstr = cfgread('EXTRACT-FITSTELLAR','n')
    fitfreq = cfgread('EXTRACT-FITFREQ','n')
    if fitfreq.find('show')>0: fitfreqshow=2
    else: fitfreqshow=0
    if fitfreq.find('save')>0: fitfreqsave=True
    else: fitfreqsave=False
    fitfreq = fitfreq.split(',')[0]
    vtype = cfgread('EXTRACT-FREQTYPE','')
    if vtype=='': vtype = cfgread('SPECTRAL-TYPE','dual')
    vsteps = cfgread('EXTRACT-FREQSTEPS','')
    if vsteps=='': vsteps = cfgread('SPECTRAL-STEPS','400,100,100,5,0.5');
    vlimits = cfgread('EXTRACT-FREQLIMITS','')
    if vlimits=='': vlimits = cfgread('SPECTRAL-LIMITS','1e-2,1e-4,1e-6');
    vsteps=[float(x) for x in vsteps.split(',')]
    vlimits=[float(x) for x in vlimits.split(',')]
    fnorm = cfgread('EXTRACT-NORMALIZE','n')
    align = cfgread('STORE-ALIGN','y')
    mret = cfgread('ATMOSPHERE-RETRIEVE','n')
    matm = cfgread('ATMOSPHERE-MULTIPLE','n')
    nodding = cfgread('DATA-NODDING','on')
    slitw = float(cfgread('DATA-SLITWIDTH','0.5'))
    slitl = float(cfgread('DATA-SLITLENGTH','20.0'))
    slitp = float(cfgread('DATA-SLITSCALE','0.0'))
    if slitp==0: slitp = slitl/(szy/2.0)

    # Read parmaters and define scaling factors
    setinfo = np.reshape(np.genfromtxt('spec.sets.txt',delimiter=',',usecols=[1,3,4]),[nsets,3])
    if os.path.exists('spec.ephm.txt'): ephm = np.reshape(np.genfromtxt('spec.ephm.txt'),[nsets,9])
    else: ephm = np.zeros([nsets,9])
    if os.path.exists('spec.geometry.txt'): slitpos = np.reshape(np.genfromtxt('spec.geometry.txt'),[nsets,3])
    else: slitpos = np.zeros([nsets,3])
    cfs = np.reshape(np.genfromtxt('spec.frequency.txt'),[nsets,6])
    if os.path.exists(fsys): psys = np.genfromtxt(fsys)
    else: psys=[]

    # Process the selected sets
    if sets=='all': sets=np.arange(0,nsets)
    else: sets=[int(x) for x in sets.split(',')]
    if stack=='y':
        beams = np.reshape(np.sum(beams[sets,:,:],axis=0)/len(sets),[1,szy,szx])
        noise = np.reshape(np.sqrt(np.sum(noise[sets,:,:]**2,axis=0))/len(sets),[1,szy,szx])
        sets = [-1];
    #Endif
    msets=len(sets)

    # Establish the rows to analyze
    if rows[0]=='all':
        if align=='y': shifts = np.reshape(np.genfromtxt('spec.align.txt'),[nsets,3])
        else: shifts = np.zeros([nsets,3])
        if nodding=='on':
            rmin = int(np.round(szy*(3.0/8.0) - np.min(shifts[:,1]) + 2)) - rmid
            rmax = int(np.round(szy*(5.0/8.0) - np.max(shifts[:,2]) - 2)) - rmid
        else:
            rmin = int(np.round(szy*(1.0/4.0) - np.min(shifts[:,1]) + 2)) - rmid
            rmax = int(np.round(szy*(3.0/4.0) - np.max(shifts[:,1]) - 2)) - rmid
        #Endelse
        if len(rows)==1: drows = rmax-rmin+1
        else: drows = int(rows[1])
        print('All rows are from %d to %d\n' % (rmin,rmax))
    else:
        rmin = int(rows[0])
        if len(rows)==1: rmax=rmin
        else: rmax = int(rows[1])
        if len(rows)<=2: drows = rmax-rmin+1
        else: drows = int(rows[2])
    #Endelse
    rows=[]; i=rmin
    while (i+drows)<=(rmax+drows): rows.append(i); i=i+drows
    nrows=len(rows)

    # Define the mask
    wmask = np.zeros(szx)+1.0
    if len(mask)>1 and mask[0]!='ask':
        for i in range(0,int(len(mask)),2): wmask[int(mask[i]):int(mask[i+1])+1] = 0.0
    #Endif
    gaslines = np.zeros([10,200]); gasname =[]; gasn = np.zeros(10,dtype=np.int)
    fq = genfreq(szx, cfs[int(nsets/2),1:6])
    for i in range(10):
        str = cfgread('EXTRACT-GAS%d' % (i+1),'')
        if len(str)<=0: gasname.append(''); continue
        str = str.split(',')
        if len(str)<3: continue
        gasname.append(str[0])
        wl = int((int(str[1])-1)/2)
        gasn[i] = len(str)-2
        for j in range(len(str)-2):
            gaslines[i,j] = float(str[j+2])
            fl = gaslines[i,j]*(1.0 - ephm[int(nsets/2),2]/3e5);
            if fq[0]<fq[1]: il = int(np.round(np.interp(fl, fq, xpix)))
            else: il = int(np.round(np.interp(fl, np.flip(fq), np.flip(xpix))))
            if il-wl>=0 and il+wl<szx: wmask[il-wl:il+wl+1]=0.0
        #Endfor
    #Endfor

    # Iterate across all sets
    data = np.zeros([msets+1,nrows+1,5,szx]);
    info = np.zeros([msets+1,nrows+1,21])
    for i in range(msets):
        kset = sets[i]
        if kset<0: rset=int(nsets/2); mset=0
        else:      rset=kset; mset=kset
        fcoeff = cfs[rset,1:6]
        fq = genfreq(szx, fcoeff)
        fmin = np.min(fq); fmax = np.max(fq)
        vstar = ephm[rset,2] + ephm[rset,3] - 1.0
        if i==0: fsys = 1.0*fq; msys = np.zeros([nrows+msets+1,szx]); wsys = np.zeros([nrows+msets+1])

        # Get telluric model
        if len(mret)>1 and (matm!='n' or i==0) and mret[0]=='y':
            irow = int(nrows/2)
            r1 = rows[irow] + rmid
            r2 = rows[irow] + drows + rmid
            nd = np.sqrt(np.sum(noise[mset,r1:r2,:]**2,axis=0))
            md = np.sum(beams[mset,r1:r2,:],axis=0)
            mx = np.max(md*wmask)
            md = md/mx
            nd = nd/(mx*(wmask+1e-4))
            if len(psys)>2: md -= np.interp(fq, psys[:,0], psys[:,1])
            mtel = retrievetrans(fq,md,nd,iset=kset,show=show,vstar=vstar)
        else:
            mtel = gentrans([fmin-3,fmax+3],iset=kset,vstar=vstar,reset=reset)
        #Endelse
        if len(psys)>2: mtel[2,:] += np.interp(mtel[0,:], psys[:,0], psys[:,1])

        basetrn = np.interp(fq, mtel[0,:], mtel[2,:])
        basestr = np.interp(fq, mtel[0,:], mtel[3,:])

        # Process the rows
        for irow in range(nrows):
            r1 = rows[irow] + rmid
            r2 = rows[irow] + drows + rmid
            info[i,irow,2] = r1-rmid
            info[i,irow,3] = r2-rmid-1
            info[i,irow,4:7] = slitpos[rset,:]
            info[i,irow,7] = slitw
            info[i,irow,8] = slitp
            info[i,irow,9:12] = setinfo[rset,:]
            info[i,irow,12:21] = ephm[rset,:]
            md = np.sum(beams[mset,r1:r2,:],axis=0)
            nd = np.sqrt(np.sum(noise[mset,r1:r2,:]**2,axis=0)); pf=[]
            if nbase>0: pf.extend(np.repeat(0.0,nbase))
            if nbase>=0: pf.append(0.0)
            if ngain>0: pf.extend(np.repeat(0.0,ngain))
            if ngain>=0: pf.append(np.max(md))
            if fitstr=='y': pf.append(1.0)
            cf,vm = curve_fit(basefit, xpix, md, sigma=nd/(wmask+1e-4), p0=pf)
            md -= baseline
            info[i,irow,1] = np.mean(basegain)
            if fnorm=='y': md/=basegain; nd/=basegain; basegain=1.0
            resi = md/basegain - basetrn

            # Remove fringes
            for k in range(nfringe):
                freqs = np.arange(3000)*0.01 + 2.0
                lnp = lombscargle(fq, resi*wmask, freqs)
                imax = np.argmax(lnp)

                if imax<10: continue
                fa = np.sqrt(lnp[imax]*4.0/szx)
                if fa< (np.mean(nd)/np.mean(basegain)): continue
                basefringefq = np.sum(freqs[imax-3:imax+4]*lnp[imax-3:imax+4])/np.sum(lnp[imax-3:imax+4])
                curve_fit(removefringe, fq, resi*wmask, sigma=nd/(basegain*(wmask+1e-4)), p0=[fa,0.0])
                md -= basefringe*basegain
                resi = md/basegain - basetrn
            #Endfor


            # Perform frequency corrections
            if fitfreq=='y':
                fcoeff = ffit(md/basegain, mtel[0,:], mtel[2,:], fcoeff, vs=vsteps, vl=vlimits, type=vtype, show=fitfreqshow)
                fq = genfreq(szx, fcoeff)
                basetrn = np.interp(fq, mtel[0,:], mtel[2,:])
                basestr = np.interp(fq, mtel[0,:], mtel[3,:])
                if irow==int(nrows/2): cfs[rset,1:6] = fcoeff
            #Endif

            resi = md/basegain - basetrn
            snr = 1.0/np.std(resi*wmask)
            if snr<minsnr: snr=0.0; md = basegain*basetrn; resi = 0.0*resi
            sresi = (snr*snr)*np.interp(fsys, fq, resi)
            msys[irow,:] += sresi; wsys[irow] += snr*snr
            msys[nrows+i,:] += sresi; wsys[nrows+i] += snr*snr
            msys[nrows+msets,:] += sresi; wsys[nrows+msets] += snr*snr
            info[i,irow,0] = snr
            data[i,irow,0,:] = fq
            data[i,irow,1,:] = md
            data[i,irow,2,:] = nd
            data[i,irow,3,:] = basegain*basetrn
            data[i,irow,4,:] = basetrn
        #Endfor rows
    #Endfor sets

    # Save systematic residual file
    for i in range(nrows+msets+1):
        if wsys[i]>0: msys[i,:] /= wsys[i]
        isnr = np.std(msys[i,:]*wmask)
        if isnr>0: snr=1.0/isnr
        else: snr=0.0
        if i<nrows: info[msets,i,0] = snr
        else: info[i-nrows,nrows,0] = snr
    #Endfors
    fw = open('spec.extract.sys','w')
    for x in range(szx): fw.write(' %.5f %e\n' % (fsys[x],msys[nrows+msets,x]*wmask[x]))
    fw.close()

    # Save updated frequency solutions
    if fitfreqsave:
        fw = open('spec.frequency.txt','w')
        for iset in range(nsets): fw.write('%3d %.5f %.5e %.5e %.5e %.5e\n' % (iset, cfs[iset,1],cfs[iset,2],cfs[iset,3],cfs[iset,4],cfs[iset,5]))
        fw.close()
    #Endif

    # Show results
    if show or mask[0]=='ask':
        # Iterate through rows
        for i in range(msets):
            kset = sets[i]
            if kset<0: rset=int(nsets/2); efile = 'spec.extract.png';
            elif msets==1: efile = 'spec.extract.png'; rset=sets[i]
            else: efile = 'spec.extract.%02d.png' % sets[i]
            dt = data[i,0:nrows,:,:]
            vmax = np.max(dt[:,1,:])
            if fnorm=='y': rmax=0.5; rmin=-0.5
            else: rmax = np.max(dt[:,1,:]-dt[:,3,:])*resiscl; rmin = np.min(dt[:,1,:]-dt[:,3,:])*resiscl
            nmax = np.max(dt[:,2,:])
            if len(rows)<6: figh=5 + 0.5*len(rows)
            else: figh=8
            fig, ax = plt.subplots(1,1,figsize=[11,figh])
            plt.xlabel('Frequency [cm-1]'); rtop=0.0
            plt.plot(np.average(data[i,0:nrows,0,:],axis=0),np.average(data[i,0:nrows,3,:],axis=0),color='red',label='Telluric')
            if fitstr=='y' and fnorm=='y': rtop+=0.3; plt.plot(mtel[0,:],mtel[3,:]*mtel[2,:]-rmax*rtop,linewidth=0.7,color='blue',label='Solar')
            if len(psys)>2 and fnorm=='y': rtop+=0.3; plt.plot(psys[:,0],psys[:,1]-rmax*rtop,linewidth=0.7,color='green',label='Systematics')
            if nrows>1:
                rtop+=0.6;
                plt.plot(fsys,resiscl*msys[nrows+i,:]-rmax*rtop,color='black');
                plt.text(fmin+(fmax-fmin)*0.03,-rmax*(rtop-0.3),'Total SNR:%d' % info[i,nrows,0],color='blue')
                rtop+=0.4;
            #Endif

            for irow in range(len(rows)):
                if info[i,irow,0]<=0: continue
                plt.plot(dt[irow,0,:], dt[irow,1,:],color='black', linewidth=0.7)
                plt.plot(dt[irow,0,:], resiscl*(dt[irow,1,:]-data[i,irow,3,:])-rmax*(irow+1+rtop),color='black', linewidth=0.7)
                plt.text(fmin+(fmax-fmin)*0.03,-rmax*(irow+0.7+rtop),'%d %d SNR:%d' % (rows[irow],rows[irow]+drows-1,info[i,irow,0]),color='blue')
                if nrows==1:
                    plt.plot(dt[irow,0,:], resiscl*dt[irow,2,:]-rmax*(irow+1+rtop),color='green')
                    plt.plot(dt[irow,0,:],-resiscl*dt[irow,2,:]-rmax*(irow+1+rtop),color='green')
                #Endif
            #Endfor irow

            # Add gas labels
            tcolor=['orange','purple','olive','green','red','blue','brown','pink','gray','cyan']
            for i in range(10):
                if gasn[i]<=0: continue
                for j in range(gasn[i]):
                    fl = gaslines[i,j]*(1.0 - ephm[rset,2]/3e5); gn=''
                    if j==0: gn = gasname[i]
                    plt.plot([fl,fl],[-rmax*(nrows+rtop)+rmin-nmax,vmax*1.1],linestyle=':',color='tab:%s' % tcolor[i],label=gn)
                #Endfor
            #Endfor

            plt.plot(dt[0,0,:], -wmask*rmin*0.3 -rmax*(nrows+rtop)+rmin-nmax,color='blue', linewidth=0.7)
            plt.ylim([-rmax*(nrows+rtop)+rmin-nmax,vmax*1.1])
            plt.xlim([np.min(dt[irow,0,:]),np.max(dt[irow,0,:])])
            plt.legend()
            plt.tight_layout()
            plt.savefig(efile, dpi=500)
            if mask[0]=='ask':
                fig.canvas.mpl_connect('button_press_event', extract_click)
                plt.show()
                str=''; extract_mask.sort()
                if fq[0]>fq[1]: extract_mask.reverse()
                for ip in extract_mask:
                    if fq[0]>fq[1]: pix = np.interp(ip,np.flip(fq),np.flip(xpix))
                    else: pix = np.interp(ip,fq,xpix)
                    print(ip,pix)
                    if pix<0 or pix>=szx: continue
                    if len(str)==0: str='%d' % pix
                    else: str='%s,%d' % (str,pix)
                #Endfor
                print(str)
                exit()
            elif show>1:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(1.0)
                plt.close()
            #Endelse
        #Endfor iset
    #Endif show

    # Save results
    # Info array -----------------------
    # 0: SNR
    # 1: Continuum [ADU]
    # 2: row 1 [from center slit]
    # 3: row 2 [from center slit]
    # 4: Slit x-pos [arcsec]
    # 5: Slit y-pos [arcsec]
    # 6: Slit rotation [degrees from NS on planet]
    # 7: Slitwidth [arcsec]
    # 8: Slitscale [arcsec/pixel]
    # 9: Airmass
    #10: Integration time [sec]
    #11: Coadds
    #12: Delta [AU]
    #13: Heliocentric distance [AU]
    #14: Delta velocity [km/s]
    #15: Helio velocity [km/s]
    #16: Observer's sub-longitude [E degrees]
    #17: Observer's sub-latitude [degrees]
    #18: Solar sub-longitude [E degrees]
    #19: Solar sub-latitude [degrees]
    #20: Season [degrees]
    fw = open('spec.extract.dat','wb')
    np.asarray([msets,nrows,szx,21]).tofile(fw)
    data[:msets,:nrows,:,:].tofile(fw)
    info[:msets,:nrows,:].tofile(fw)
    fw.close()
#End extract module

# Retrieval module ---------------------------------
def retrieve(show=False,reset=False):
    if reset==2 or not os.path.exists('spec.extract.dat'): store(show=show,reset=reset)
    fr = open('spec.extract.dat','rb')
    nsets = np.fromfile(fr, dtype=int, count=1)[0]
    nrows = np.fromfile(fr, dtype=int, count=1)[0]
    npts = np.fromfile(fr, dtype=int, count=1)[0]
    ninfo = np.fromfile(fr, dtype=int, count=1)[0]
    data = np.fromfile(fr, dtype=float, count=nsets*nrows*5*npts); data = np.reshape(data, [nsets,nrows,5,npts])
    info = np.fromfile(fr, dtype=float, count=nsets*nrows*ninfo); info = np.reshape(info, [nsets,nrows,ninfo])
    fr.close()
    info_sets = np.genfromtxt('spec.sets.txt', delimiter=',',dtype='<U20')
    obs_date = [x.replace('-','/') for x in info_sets[:,0]]
    # Read parameters
    fbase = cfgread('RETRIEVAL-CONFIG','spec.retrieve.base')
    pfile = cfgread('RETRIEVAL-PARAMS','spec.retrieve.params')
    wephm = cfgread('RETRIEVAL-UPDATEEPHM','n')
    wgeo = cfgread('RETRIEVAL-UPDATEGEOMETRY','n')
    watm = cfgread('RETRIEVAL-UPDATEATMOSPHERE','n')
    wcont = cfgread('RETRIEVAL-INCLUDECONTINUUM','n')
    fscl = float(cfgread('RETRIEVAL-FLUXSCALER','1.0'))
    fnorm = cfgread('EXTRACT-NORMALIZE','n')
    psgserver = cfgread('SPECTRAL-SERVER','https://psg.gsfc.nasa.gov')
    psgkey = cfgread('SPECTRAL-PSGKEY','')
    if len(psgkey)>0: psgserver = '-d key=%s %s' % (psgkey, psgserver)
    modres = cfgread('RETRIEVAL-RESOLUTION','')
    seeing = float(cfgread('DATA-SEEING','0.7'))
    if len(modres)==0: modres = cfgread('ATMOSPHERE-RESOLUTION','70000')
    vars = cfgread('RETRIEVAL-VARIABLES','')
    if len(vars)==0: return
    else: svars=vars.split(','); nvar = len(svars)
    vals = cfgread('RETRIEVAL-VALUES','')
    mins = cfgread('RETRIEVAL-MIN','')
    maxs = cfgread('RETRIEVAL-MAX','')
    scls = cfgread('RETRIEVAL-UNITS','')
    taus = cfgread('RETRIEVAL-LIFETIMES','')
    dtele = float(cfgread('DATA-DIAMTELE','10.0'))
    neff = float(cfgread('DATA-EFFICIENCY','0.1'))
    gamma = float(cfgread('DATA-GAMMA','0.0'))
    dtc = cfgread('DATA-DETECTOR','1.8,12.0,0.1'); dtc=[float(x) for x in dtc.split(',')]
    nbase = cfgread('RETRIEVAL-REMOVEOFFSET','')
    if len(nbase)==0: nbase = cfgread('EXTRACT-REMOVEOFFSET','n')
    if nbase=='n': nbase=-1
    else: nbase = int(nbase)
    ngain = cfgread('RETRIEVAL-FITGAIN','')
    if len(ngain)==0: ngain = cfgread('EXTRACT-FITGAIN','2')
    if ngain=='n': ngain=-1
    else: ngain = int(ngain)
    nfringe = cfgread('RETRIEVAL-REMOVEFRINGE','')
    if len(nfringe)==0: nfringe = cfgread('EXTRACT-REMOVEFRINGE','n')
    if nfringe=='n': nfringe=0
    else: nfringe = int(nfringe)
    modtel = cfgread('RETRIEVAL-TELLURIC','')
    fitstr = cfgread('RETRIEVAL-FITSTELLAR','')
    if len(fitstr)==0: fitstr = cfgread('EXTRACT-FITSTELLAR','n')
    fitfreq = cfgread('RETRIEVAL-FITFREQ','')
    if len(fitfreq)==0: fitfreq = cfgread('EXTRACT-FITFREQ','n')
    fitatm = cfgread('RETRIEVAL-FITTELLURIC','n')
    if wcont!='y': fitatm='n'

    # Define config if missing
    if not os.path.exists(fbase):
        object = cfgread('DATA-OBJECT','Mars')
        fw = open('spec.retrieve.cfg','w')
        fw.write('<OBJECT-NAME>%s\n' % object)
        fw.write('<OBJECT-DATE>%s\n' % obs_date[0])
        fw.close()
        os.system('curl -s -d type=cfg -d wephm=y -d watm=y -d wgeo=y --data-urlencode file@spec.retrieve.cfg %s/api.php > %s' % (psgserver,fbase))
        sn=''; st=''; ngas=0
        for var in svars:
            ss = var.split('-')
            if ss[0].upper()!='ATMOSPHERE' and ss[0].upper()!='COMA': continue
            ss = ss[1].split('[')
            if ss[0].upper()=='TEMPERATURE': continue
            if len(ss)==1: type = 'GSFC[%s]' % ss[0]
            elif ss[1][0].isnumeric(): type = 'HIT[%s' % ss[1]
            else: type = 'GSFC[%s' % ss[1]
            if len(sn)==0: sn = ss[0]
            else: sn = '%s,%s' % (sn, ss[0])
            if len(st)==0: st = type
            else: st = '%s,%s' % (st, type)
            ngas = ngas+1
        #Endfor
        fw = open(fbase,'a')
        fw.write('<ATMOSPHERE-NGAS>%d\n' % ngas)
        fw.write('<ATMOSPHERE-GAS>%s\n' % sn)
        fw.write('<ATMOSPHERE-TYPE>%s\n' % st)
        fw.close()
    #Endif

    fr = open(fbase,'r'); lbase = fr.readlines(); fr.close()
    asn=''; asg=''; ast=''; asa=''; asu=''
    for line in lbase:
        if '<ATMOSPHERE-NGAS>' in line: asn = line
        if '<ATMOSPHERE-GAS>'  in line: asg = line
        if '<ATMOSPHERE-TYPE>' in line: ast = line
        if '<ATMOSPHERE-ABUN>' in line: asa = line
        if '<ATMOSPHERE-UNIT>' in line: asu = line
    #Endfor

    # Read additional retrieval parameters
    if os.path.exists(pfile): fr = open(pfile,'r'); lparams = fr.readlines(); fr.close()
    else: lparams=[]

    # Compute gamma if efficiency provided
    if gamma==0:
        fmid = np.mean(data[0,0,0,:])
        mdf = np.mean(data[0,0,0,:-1] - data[0,0,0,1:])
        cs = 2.99792458e10          # Speed of light [cm/s]
        hp = 6.6260693E-34          # Speed of light [W/s2]
        cflux = np.pi*((dtele/2)**2)*mdf
        cflux *= (1.0/fmid)/(cs*hp*dtc[0])
        gamma = 1.0/(neff*cflux)
    #Endif
    # Iterate across sets
    data2 = np.zeros([nsets,nrows,7,npts]); data2[:,:,:5,:] = data
    info2 = np.zeros([nsets,nrows,ninfo+nvar*2]); info2[:,:,:ninfo] = info
    for iset in range(nsets):
        for irow in range(nrows):
            if info[iset,irow,0]==0: continue

            # Prepare the retrieval configuration file
            fw = open('spec.retrieve.cfg','w')
            for line in lbase:
                if '<OBJECT-DATE>' in line: line = '<OBJECT-DATE>'+obs_date[iset]+'\n'
                fw.write(line)
            fw.write('<RETRIEVAL-NVARS>%d\n' % nvar)
            fw.write('<RETRIEVAL-VARIABLES>%s\n' % vars)
            fw.write('<RETRIEVAL-VALUES>%s\n' % vals)
            fw.write('<RETRIEVAL-MIN>%s\n' % mins)
            fw.write('<RETRIEVAL-MAX>%s\n' % maxs)
            fw.write('<RETRIEVAL-UNITS>%s\n' % scls)
            fw.write('<RETRIEVAL-FLUXSCALER>%.4e\n' % fscl)
            fw.write('<RETRIEVAL-FITTELLURIC>%s\n' % fitatm.upper())
            fw.write('<RETRIEVAL-FITGAIN>%d\n' % ngain)
            if fnorm=='n': fw.write('<RETRIEVAL-FITGAIN-PHOTOMETRIC>Y\n'); fw.write('<GENERATOR-RADUNITS>rif\n')
            else: fw.write('<RETRIEVAL-FITGAIN-PHOTOMETRIC>N\n')
            fw.write('<RETRIEVAL-REMOVEOFFSET>%d\n' % nbase)
            fw.write('<RETRIEVAL-REMOVEFRINGE>%d\n' % nfringe)
            fw.write('<RETRIEVAL-FITSTELLAR>%s\n' % fitstr.upper())
            fw.write('<RETRIEVAL-FITFREQ>%s\n' % fitfreq.upper())
            fw.write('<RETRIEVAL-FITFREQ>%s\n' % fitfreq.upper())
            fw.write('<GENERATOR-CONT-MODEL>%s\n' % wcont.upper())
            fw.write('<GENERATOR-RANGE1>%.2f\n' % np.min(data[0,0,0,:]))
            fw.write('<GENERATOR-RANGE2>%.2f\n' % np.max(data[0,0,0,:]))
            fw.write('<GENERATOR-RANGEUNIT>cm\n')
            fw.write('<GENERATOR-RESOLUTION>%s\n' % modres)
            fw.write('<GENERATOR-RESOLUTIONUNIT>RP\n')
            fw.write('<GENERATOR-RESOLUTIONKERNEL>Y\n')
            if len(modtel)>0:
                fw.write('<GENERATOR-TRANS-APPLY>Y\n')
                fw.write('<GENERATOR-TRANS-SHOW>Y\n')
                fw.write('<GENERATOR-TRANS>%s\n' % modtel)
            #Endif
            fw.write('<GENERATOR-BEAM>%.3f,%.3f,%.3f,R\n' % (info[iset,irow,7], (info[iset,irow,3]-info[iset,irow,2]+1)*info[iset,irow,8], info[iset,irow,6]))
            fw.write('<GENERATOR-BEAM-UNIT>arcsec\n')
            tint = info[iset,irow,10]*info[iset,irow,11]
            if fnorm=='y':
                tscl=1.0
                fw.write('<GENERATOR-RADUNITS>rif\n')
            else:
                tscl = gamma/tint
                fw.write('<GENERATOR-RADUNITS>Wm2cm\n')
            #Endif
            if wephm=='y':
                fw.write('<GEOMETRY-OBS-ALTITUDE>%.5f\n' % info[iset,irow,12])
                fw.write('<OBJECT-STAR-DISTANCE>%.5f\n' % info[iset,irow,13])
                fw.write('<OBJECT-OBS-VELOCITY>%.5f\n' % info[iset,irow,14])
                fw.write('<OBJECT-STAR-VELOCITY>%.5f\n' % info[iset,irow,15])
                fw.write('<OBJECT-OBS-LONGITUDE>%.5f\n' % info[iset,irow,16])
                fw.write('<OBJECT-OBS-LATITUDE>%.5f\n' % info[iset,irow,17])
                fw.write('<OBJECT-SOLAR-LONGITUDE>%.5f\n' % info[iset,irow,18])
                fw.write('<OBJECT-SOLAR-LATITUDE>%.5f\n' % info[iset,irow,19])
                fw.write('<OBJECT-SEASON>%.5f\n' % info[iset,irow,20])
            #Endif
            if wgeo=='y':
                px = info[iset,irow,4]
                py = info[iset,irow,5] + (info[iset,irow,2]+info[iset,irow,3])*info[iset,irow,8]*0.5
                rc = np.cos(info[iset,irow,6]*np.pi/180.0); rs = np.sin(info[iset,irow,6]*np.pi/180.0)
                rx = px*rc - py*rs
                ry = px*rs + py*rc
                fw.write('<GEOMETRY-OFFSET-NS>%.5f\n' % ry)
                fw.write('<GEOMETRY-OFFSET-EW>%.5f\n' % rx)
                fw.write('<GEOMETRY-OFFSET-UNIT>arcsec\n')
            #Endif
            if len(taus)>0: fw.write('<ATMOSPHERE-TAU>%s\n' % taus)
            fw.close()

            # Update atmosphere
            if watm == 'y':
                os.system('curl -s -d type=cfg -d watm=y -d wgeo=n --data-urlencode file@spec.retrieve.cfg %s/api.php > spec.retrieve.atm' % (psgserver))
                fr = open('spec.retrieve.atm','r'); lcfg = fr.readlines(); fr.close()
                fw = open('spec.retrieve.cfg','w')
                for line in lcfg: fw.write(line)
                if len(asn)>0: fw.write(asn)
                if len(asg)>0: fw.write(asg)
                if len(ast)>0: fw.write(ast)
                if len(asa)>0: fw.write(asa)
                if len(asu)>0: fw.write(asu)
                fw.close()
                os.system('rm spec.retrieve.atm')
            #Endif

            # Add data and perform retrieval
            fw = open('spec.retrieve.cfg','a')
            for line in lparams: fw.write(line)
            fw.write('<DATA>\n')
            fq = data[iset,irow,0,:]
            nd = data[iset,irow,2,:]*tscl
            tr = data[iset,irow,4,:]
            if wcont=='y': dt = data[iset,irow,1,:]*tscl
            else: dt = (data[iset,irow,1,:]-data[iset,irow,3,:])*tscl
            for i in range(npts): fw.write('%.5f %e/%.5f %e\n' % (fq[i], dt[i], tr[i], nd[i]))
            fw.write('</DATA>\n')
            fw.close()
            os.system('curl -s -d wjac=y -d watm=n -d type=ret --data-urlencode file@spec.retrieve.cfg %s/api.php > spec.retrieve.ret' % (psgserver))

            # Read results
            fr = open('spec.retrieve.ret','r'); lines=fr.readlines(); fr.close()
            mret = np.zeros([3+nvar, npts])
            inrad=0; injac=0; i=0; cols=''
            for line in lines:
                if line[:8]=='results_': inrad=0; injac=0
                elif line[:22]=='<RETRIEVAL-FLUXLABELS>': cols=line[22:-1].split(',')
                elif line[:18]=='<RETRIEVAL-VALUES>': rvals=line[18:-1].split(',')
                elif line[:18]=='<RETRIEVAL-SIGMAS>': rsigs=line[18:-1].split(',')
                if line[:-1]=='# Jacobians':
                    injac=1; i=0; continue
                elif line[0]=='#' or line[0]=='<':
                    continue
                elif line[:-1]=='results_dat.txt':
                    inrad=1; i=0; continue
                if inrad:
                    ss=line.split(); gain=1.0; base=0.0
                    model = float(ss[3])
                    if len(cols)>3:
                        if cols[3]=='Gain' and len(ss)>3: gain=float(ss[4])
                        if cols[3]=='Base' and len(ss)>3: base=float(ss[4])
                    if len(cols)>4:
                        if cols[4]=='Base' and len(ss)>4: base=float(ss[5])
                    mret[0,i] = model
                    mret[1,i] = gain
                    mret[2,i] = base
                    i=i+1
                elif injac:
                    ss=line.split()
                    for j in range(nvar): mret[3+j,i] = float(ss[j+1])
                    i=i+1
                #Endif
            #Endfor
            md = np.nan_to_num((mret[0,:]-mret[2,:])/mret[1,:], posinf=0,neginf=0)
            dt = np.nan_to_num((dt-mret[2,:])/mret[1,:], posinf=0,neginf=0)
            data2[iset,irow,5,:] = md
            data2[iset,irow,6,:] = dt
            info2[iset,irow,ninfo:ninfo+nvar] = np.nan_to_num(rvals, posinf=0,neginf=0)
            info2[iset,irow,ninfo+nvar:ninfo+nvar*2] = np.nan_to_num(rsigs, posinf=0,neginf=0)

            # Plot the results
            if show:
                fig,ax=plt.subplots(2,1,figsize=[10,8],sharex=True)
                rmax =  np.max([dt,md])*1.1
                rmid = -np.max(dt-md)*1.1 - np.max(nd) + np.min([dt,md])
                rmin = rmid + np.min(dt-md)*1.1 - np.max(nd)
                ax[0].set_title('Set %d, Rows: %d %d' % (iset,info[iset,irow,2],info[iset,irow,3]))
                ax[0].set_ylim([rmin,rmax])
                ax[0].set_xlim([np.min(fq),np.max(fq)])
                ax[1].set_xlabel('Frequency [cm-1]')
                if fnorm=='y': ax[0].set_ylabel('Normalized intensity')
                else: ax[0].set_ylabel('Radiance [W/m2/cm-1]')
                ax[0].plot(fq,md,color='red',label='Model')
                ax[0].plot(fq,dt,linewidth=0.7,color='black',label='Data')
                ax[0].plot(fq,rmid+nd,color='green',label='1-sigma')
                ax[0].plot(fq,rmid-nd,color='green')
                ax[0].plot(fq,dt-md+rmid,linewidth=0.7,color='black',label='Residual')
                ax[0].legend()

                jmax=[]
                for j in range(nvar): jmax.append(np.max(abs(mret[j+3,:])))
                jmmax = np.max(jmax)
                for j in range(nvar):
                    if jmax[j]==0.0 or jmmax==0.0: scl=0.0
                    elif jmax[j]/jmmax<0.3: scl=0.3/jmax[j]
                    else: scl=1.0/jmmax
                    ax[1].plot(fq, mret[j+3,:]*scl - 0.2*j,label='%s %s %s' % (svars[j],rvals[j],rsigs[j]))
                #Endfor
                ax[1].legend(fontsize=10)
                ax[1].set_ylabel('Jacobians')
                plt.tight_layout()
                if nsets>1 and nrows>1: fimg = 'spec.retrieve%d-%d.png' % (iset,irow)
                elif nsets>1: fimg = 'spec.retrieve%d.png' % iset
                elif nrows>1: fimg = 'spec.retrieve%d.png' % irow
                else: fimg = 'spec.retrieve.png'
                plt.savefig(fimg)
                if show>1:
                    plt.show()
                else:
                    plt.show(block=False)
                    plt.pause(1.0)
                    plt.close()
                #Endelse
            #Endif
        #Endfor
    #Endfor

    # Save results
    fw = open('spec.retrieve.dat','wb')
    np.asarray([nsets,nrows,npts,ninfo+nvar*2]).tofile(fw)
    data2.tofile(fw)
    info2.tofile(fw)
    fw.close()

#End retrieve

# --------------------------------------------------------------
# Main module: verify if it's being run as script
# --------------------------------------------------------------
# To run python -m spec module=crop show=True
if __name__ == "__main__":
    import sys
    module = 'store'; args='';
    for arg in sys.argv[1:]:
        if arg[0:8]=='cfgfile=':
            cfgfile=arg[8:]
        elif arg[0:7]=='module=':
            module=arg[7:]
        else:
            if len(args)==0: args=arg
            else: args='%s,%s' % (args,arg)
        #Endelse
    #Endloop
    eval('%s(%s)' % (module,args))
#End Main
