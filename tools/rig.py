import sys, os, json, struct, io, numpy as np
from PIL import Image
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from load import load, acc, img
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
SRC=os.path.join(ROOT,'source','tripo_model.glb')
j,BIN=load(SRC)
p=j['meshes'][0]['primitives'][0]
V=acc(j,BIN,p['attributes']['POSITION']).astype(np.float64)
F=acc(j,BIN,p['indices']).reshape(-1,3)
# per-vertex colour and metal/roughness sampled from the source textures
def _sample():
    pr=j['meshes'][0]['primitives'][0]; UV_=acc(j,BIN,pr['attributes']['TEXCOORD_0'])
    bc_=np.asarray(img(j,BIN,0).convert('RGB')).astype(float)/255; rm_=np.asarray(img(j,BIN,1).convert('RGB')).astype(float)/255
    H_,W_,_=bc_.shape; px=np.clip((UV_[:,0]%1)*W_,0,W_-1).astype(int); py=np.clip((UV_[:,1]%1)*H_,0,H_-1).astype(int)
    return bc_[py,px], rm_[py,px]
C,R=_sample()
n=len(V)
CX=-0.035

# ---------- landmarks ----------
orange=(C[:,0]>0.6)&(C[:,1]<0.5)&(C[:,2]<0.3)
dark=C.sum(1)<0.35
metal=R[:,2]>0.5
O=V[orange]; oc=O.mean(0); _,_,vt=np.linalg.svd(O-oc); nrm=vt[2]
if nrm[1]<0: nrm=-nrm          # point up/forward toward the head
collar_c=oc.copy(); collar_c[0]=CX
sd=(V-oc)@nrm                  # signed distance to collar plane
print('collar centre',oc.round(3),'normal',nrm.round(3))

eyesD=V[dark&(V[:,1]>0.68)&(V[:,2]<0.43)]
eyeL=eyesD[eyesD[:,0]>CX].mean(0); eyeR=eyesD[eyesD[:,0]<CX].mean(0)
noseC=V[dark&(V[:,2]>0.43)].mean(0)
paws={}
low=V[:,1]<0.10
for name,zs,xs in [('FL',1,1),('FR',1,-1),('HL',-1,1),('HR',-1,-1)]:
    m=low&((V[:,2]+0.08)*zs>0)&((V[:,0]-CX)*xs>0)
    pc=V[m].mean(0); paws[name]=pc
print('paws',{k:v.round(3) for k,v in paws.items()})

# collar sits a hair further out from the neck
rad=(V-oc)-np.outer((V-oc)@nrm,nrm); rl=np.linalg.norm(rad,axis=1,keepdims=True)
# smooth the neck under the collar: the old collar band is relaxed into a plain neck surface (no steps at its edges)
from scipy.sparse import coo_matrix as _coo, diags as _diags
_key=np.round(V,5); _,_inv=np.unique(_key,axis=0,return_inverse=True); _inv=_inv.ravel(); _m=_inv.max()+1
_E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); _E=_inv[_E]; _E=np.unique(np.sort(_E,1),axis=0); _E=_E[_E[:,0]!=_E[:,1]]
_A=_coo((np.ones(len(_E)*2),(np.r_[_E[:,0],_E[:,1]],np.r_[_E[:,1],_E[:,0]])),shape=(_m,_m)).tocsr()
_deg=np.asarray(_A.sum(1)).ravel(); _deg[_deg==0]=1; _P=_diags(1/_deg)@_A
_Vw=np.zeros((_m,3)); _Vw[_inv]=V
_band=np.zeros(_m,bool); _band[_inv[orange]]=True
for _ in range(2): _band=_band|(np.asarray(_A@_band.astype(float)).ravel()>0)      # two rings wider than the orange band
V=_Vw[_inv].copy()
neckBand=_band[_inv]
UVmod=acc(j,BIN,p['attributes']['TEXCOORD_0']).astype(np.float32).copy()
# texels covered by the neck band: painted plain skin with a flat normal, feathered at the edge
from PIL import ImageDraw as _IDraw, ImageFilter as _IF
_bandF=F[neckBand[F].all(1)]
_mimg=Image.new('L',(2048,2048),0); _dr=_IDraw.Draw(_mimg)
for _t in _bandF:
    _dr.polygon([(float(UVmod[k,0]%1*2048),float(UVmod[k,1]%1*2048)) for k in _t],fill=255)
_mimg=_mimg.filter(_IF.MaxFilter(5)).filter(_IF.GaussianBlur(5))
bandMask=np.asarray(_mimg).astype(np.float32)/255
# ---------- bones (world positions, rest rotation = identity) ----------
B={}   # name -> (parent, pos)
def bone(name,parent,pos): B[name]=(parent,np.array(pos,float))
bone('Root',None,[0,0,0])
bone('Hips','Root',[CX,0.40,-0.24])
bone('Chest','Hips',[CX,0.42,0.06])
bone('Head','Chest',[CX,collar_c[1]+0.02,collar_c[2]-0.03])
earBase={'L':[0.14,0.80,0.15],'R':[-0.17,0.80,0.15]}
bone('Ear_L','Head',earBase['L']); bone('Ear_R','Head',earBase['R'])
bone('Eye_L','Head',eyeL); bone('Eye_R','Head',eyeR)
bone('Mouth','Head',[noseC[0],noseC[1]-0.085,noseC[2]-0.03])
for k in ['FL','FR']:
    pc=paws[k]; bone('Leg_'+k,'Chest',[pc[0],0.36,pc[2]-0.03]); bone('Paw_'+k,'Leg_'+k,[pc[0],0.11,pc[2]-0.03])
for k in ['HL','HR']:
    pc=paws[k]; bone('Leg_'+k,'Hips',[pc[0],0.36,pc[2]+0.02]); bone('Paw_'+k,'Leg_'+k,[pc[0],0.11,pc[2]+0.02])
bone('Tail','Hips',[CX,0.47,-0.40])
names=list(B.keys()); idx={k:i for i,k in enumerate(names)}

# ---------- hard labels ----------
lab=np.full(n,idx['Chest'])
lab[V[:,2]<-0.10]=idx['Hips']
head=sd>0.005
lab[head]=idx['Head']
ear=head&((V[:,1]>0.84)|((V[:,1]>0.70)&(np.abs(V[:,0]-CX)>0.17)&(V[:,2]<0.26)))
lab[ear&(V[:,0]>CX)]=idx['Ear_L']; lab[ear&(V[:,0]<CX)]=idx['Ear_R']
tail=(V[:,2]<-0.395)&(V[:,1]>0.42)
lab[tail]=idx['Tail']
for k,pc in paws.items():
    r=np.hypot(V[:,0]-pc[0],V[:,2]-pc[2])
    legm=(V[:,1]<0.33)&(r<0.105)&~head
    lab[legm]=idx['Leg_'+k]
    lab[legm&(V[:,1]<0.12)]=idx['Paw_'+k]

# ---------- adjacency on welded positions ----------
key=np.round(V,5); _,inv=np.unique(key,axis=0,return_inverse=True); inv=inv.ravel(); m=inv.max()+1
E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); E=inv[E]; E=np.unique(np.sort(E,1),axis=0); E=E[E[:,0]!=E[:,1]]
from scipy.sparse import coo_matrix, diags
A=coo_matrix((np.ones(len(E)*2),(np.r_[E[:,0],E[:,1]],np.r_[E[:,1],E[:,0]])),shape=(m,m)).tocsr()
deg=np.asarray(A.sum(1)).ravel(); deg[deg==0]=1; P=diags(1/deg)@A
W=np.zeros((m,len(names)))
# welded label = label of first vertex
first=np.zeros(m,int); first[inv[::-1]]=np.arange(n)[::-1]
W[np.arange(m),lab[first]]=1
def smooth(W,cols,iters):
    Wc=W[:,cols].copy()
    for _ in range(iters): Wc=0.5*Wc+0.5*(P@Wc)
    W2=W.copy(); W2[:,cols]=Wc; return W2
body=[idx[k] for k in ['Hips','Chest','Head','Tail']+['Leg_'+q for q in paws]]
W=smooth(W,body,22)
W=smooth(W,[idx[k] for k in ['Head','Ear_L','Ear_R']],10)
W=smooth(W,[idx['Leg_'+q] for q in paws]+[idx['Paw_'+q] for q in paws],6)
# eyes: analytic falloff around the eye centre, carved out of the head weight
Wv=W[inv]
for k,c in [('Eye_L',eyeL),('Eye_R',eyeR)]:
    d=np.linalg.norm(V-c,axis=1); w=np.clip((0.042-d)/0.016,0,1)*(sd>0)
    Wv[:,idx['Head']]-=np.minimum(w,Wv[:,idx['Head']]); Wv[:,idx[k]]+=w
# metal earrings ride rigidly with the nearest of head / left ear
ring=metal&(V[:,1]>0.62)
for i in np.where(ring)[0]:
    tgt=idx['Ear_L'] if V[i,1]>0.84 else idx['Head']
    Wv[i]=0; Wv[i,tgt]=1
# collar + tag copy the weights of the skin right under them, so skin never pokes through
from scipy.spatial import cKDTree
ct=orange|(metal&(V[:,1]<0.5))
skin=np.where(~ct&~metal)[0]
tree=cKDTree(V[skin])
dd,nn=tree.query(V[ct],k=6)
w=1/np.maximum(dd,1e-4); w/=w.sum(1,keepdims=True)
Wv[ct]=np.einsum('nk,nkb->nb',w,Wv[skin][nn])
# the tag hangs as one rigid piece from the front of the collar
tagm=metal&(V[:,1]<0.5)
tc=V[tagm].mean(0)
near=orange&(np.linalg.norm(V-tc,axis=1)<0.09)
Wv[tagm]=Wv[near].mean(0)
Wv=np.clip(Wv,0,None)
_rr=np.linalg.norm((V-oc)-np.outer((V-oc)@nrm,nrm),axis=1)
_neck=(np.abs(sd)<0.11)&(_rr<0.17)&~metal&(V[:,1]<0.72)
_t=np.clip((sd[_neck]+0.08)/0.16,0,1); _t=_t*_t*(3-2*_t)
Wv[_neck]=0; Wv[_neck,idx['Head']]=_t; Wv[_neck,idx['Chest']]=1-_t
print('neck blend verts',int(_neck.sum()))
# above the neck everything belongs to the head (no leftover chest pull that folds the skin behind the ears); below, no head pull
_body=[idx[k] for k in ['Chest','Hips','Leg_FL','Leg_FR','Paw_FL','Paw_FR']]
_up=(sd>0.07)&~_neck
_mv=Wv[_up][:,_body].sum(1); Wv[np.ix_(np.where(_up)[0],_body)]=0; Wv[_up,idx['Head']]+=_mv
_dn=(sd<-0.07)&~_neck&(_rr<0.25)
_mv2=Wv[_dn][:,idx['Head']]; Wv[_dn,idx['Head']]=0; Wv[_dn,idx['Chest']]+=_mv2
top=np.argsort(-Wv,1)[:,:4]; tw=np.take_along_axis(Wv,top,1); tw[tw<0.01]=0; tw/=tw.sum(1,keepdims=True)
J=top.astype(np.uint8); Wt=tw.astype(np.float32)


# ---------- textures ----------
def jpeg(im,size,q=88):
    im=im.convert('RGB').resize((size,size),Image.LANCZOS); bio=io.BytesIO(); im.save(bio,'JPEG',quality=q,optimize=True,progressive=False); return bio.getvalue()
bc=img(j,BIN,0).convert('RGB').resize((2048,2048),Image.LANCZOS)
a=np.asarray(bc).astype(np.float32)/255
mx=a.max(2); mn=a.min(2); sat=(mx-mn)/np.maximum(mx,1e-4)
r,g,b=a[...,0],a[...,1],a[...,2]
pinkish=(r>=g)&(b>=g-0.02)&(mx>0.55)
k=np.clip((0.22-sat)/0.10,0,1)*pinkish*0.5      # halve the pink tint on the body; ears and freckles keep theirs
lum=(0.3*r+0.59*g+0.11*b)[...,None]
white=np.clip(lum*1.06+0.03,0,1)*np.array([1.0,0.995,0.985])
a=a*(1-k[...,None])+white*k[...,None]
lum2=(0.3*a[...,0]+0.59*a[...,1]+0.11*a[...,2])
darkm=np.clip((0.42-lum2)/0.14,0,1)[...,None]      # nose, eyes, mouth line -> deeper neutral black
a=a*(1-darkm)+(lum2[...,None]*0.28)*darkm
# the skin under the collar: paint the old orange band in skin colour (the real collar is a separate piece now)
from scipy.ndimage import binary_dilation
om=((a[...,0]-a[...,1])>0.14)&((a[...,0]-a[...,2])>0.22)&(a[...,0]>0.35)
om=binary_dilation(om,iterations=6)
mx2=a.max(2); sat2=(mx2-a.min(2))/np.maximum(mx2,1e-4)
skinpx=a[(mx2>0.75)&(sat2<0.12)&~om]
a[om]=np.median(skinpx,0)*0.97
_bm=bandMask[...,None]; a=a*(1-_bm)+np.median(skinpx,0)*0.97*_bm
bcj=jpeg(Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8)),2048,88)
rm_im=img(j,BIN,1).convert('RGB').resize((2048,2048),Image.LANCZOS)
rma=np.asarray(rm_im).astype(np.float32)/255
ab=a  # final base colour
orangeT=((ab[...,0]>0.6)&(ab[...,1]<0.5)&(ab[...,2]<0.3)).astype(np.float32)
metalT=(rma[...,2]>0.5).astype(np.float32)
skin=np.clip(1-darkm[...,0]-orangeT-metalT,0,1)
rma[...,1]=np.maximum(rma[...,1],0.78)*skin + 0.22*darkm[...,0] + 0.38*orangeT*(1-darkm[...,0]) + rma[...,1]*metalT*(1-darkm[...,0])   # matte skin, glossy nose/eyes, satin collar
rmj=jpeg(Image.fromarray((rma*255).astype(np.uint8)).resize((1024,1024),Image.LANCZOS),1024,90)
_nm=np.asarray(img(j,BIN,2).convert('RGB').resize((2048,2048),Image.LANCZOS)).astype(np.float32)/255
_nm=_nm*(1-bandMask[...,None])+np.array([0.5,0.5,1.0],np.float32)*bandMask[...,None]
nmj=jpeg(Image.fromarray((_nm*255).astype(np.uint8)),2048,90)

# ---------- write GLB ----------
out=bytearray(); views=[]; accs=[]
def add_view(data,target=None):
    while len(out)%4: out.append(0)
    o=len(out); out.extend(data); v={'buffer':0,'byteOffset':o,'byteLength':len(data)}
    if target: v['target']=target
    views.append(v); return len(views)-1
def add_acc(arr,ctype,typ,target=34962,minmax=False):
    vi=add_view(arr.tobytes(),target); a={'bufferView':vi,'componentType':ctype,'count':len(arr),'type':typ}
    if minmax: a['min']=arr.min(0).tolist(); a['max']=arr.max(0).tolist()
    accs.append(a); return len(accs)-1
# ---------- stitch the neck: bridge the head's and the body's open rims with a strip of skin ----------
from collections import defaultdict as _dd
keepF=~(metal[F].sum(1)>=np.where(V[F].mean(1)[:,1]>0.65,1,2)) & ~(orange[F].all(1))
Fk=F[keepF]
Nrm=acc(j,BIN,p['attributes']['NORMAL']).astype(np.float32)
Vo=V.astype(np.float32); No=Nrm.copy(); UVo=UVmod.copy(); Jo=J.copy(); Wo=Wt.copy()
def _loops(Fs):
    E=np.concatenate([Fs[:,[0,1]],Fs[:,[1,2]],Fs[:,[2,0]]]); Ei=inv[E]; srt=np.sort(Ei,1)
    u,idx,c=np.unique(srt,axis=0,return_index=True,return_counts=True); bd=Ei[idx[c==1]]
    adj=_dd(list)
    for a_,b_ in bd: adj[a_].append(b_); adj[b_].append(a_)
    seen=set(); out=[]
    for st in adj:
        if st in seen: continue
        comp=[];stack=[st]
        while stack:
            x=stack.pop()
            if x in seen: continue
            seen.add(x); comp.append(x); stack+=adj[x]
        out.append(comp)
    return out
_first=np.zeros(m,int); _first[inv[::-1]]=np.arange(n)[::-1]
cands=[]
for L in _loops(Fk):
    pts=V[_first[L]]; c_=pts.mean(0)
    if len(L)>=20 and np.linalg.norm(c_-oc)<0.08: cands.append((c_[1],L))
cands.sort(key=lambda t:t[0])
print('neck rims',[(round(float(y),3),len(L)) for y,L in cands])
newF=[]
if len(cands)>=2:
    lowL,upL=cands[0][1],cands[-1][1]
    ax=nrm/np.linalg.norm(nrm); u_=np.cross(ax,[1,0,0]); u_/=np.linalg.norm(u_); w_=np.cross(ax,u_)
    def walk(Lset):
        Ls=set(Lset)
        E=np.concatenate([Fk[:,[0,1]],Fk[:,[1,2]],Fk[:,[2,0]]]); Wd=inv[E]; srt=np.sort(Wd,1)
        u,ix,c=np.unique(srt,axis=0,return_index=True,return_counts=True)
        nxt={}
        for e in ix[c==1]:
            if Wd[e,0] in Ls: nxt[Wd[e,0]]=Wd[e,1]
        st=next(iter(nxt)); seq=[st]; x=nxt[st]
        while x!=st and x in nxt and len(seq)<=len(Ls): seq.append(x); x=nxt[x]
        ids=_first[np.array(seq)]; d=V[ids]-oc; ang=np.unwrap(np.arctan2(d@w_,d@u_))
        if ang[-1]-ang[0]<0: ids=ids[::-1]; d=d[::-1]
        a0=np.arctan2(d@w_,d@u_); s0=int(np.argmin(np.abs(a0))); ids=np.roll(ids,-s0)
        seg=np.linalg.norm(np.diff(V[np.r_[ids,ids[:1]]],axis=0),axis=1); t=np.r_[0,np.cumsum(seg)[:-1]]/seg.sum()
        return ids,t
    # tuck both rims and their neighbours a little inside the collar so nothing flares out past it
    rimW=np.zeros(m); rimW[lowL]=1; rimW[upL]=1
    _Ek=np.concatenate([Fk[:,[0,1]],Fk[:,[1,2]],Fk[:,[2,0]]]); _Ek=inv[_Ek]
    _Ak=coo_matrix((np.ones(len(_Ek)*2),(np.r_[_Ek[:,0],_Ek[:,1]],np.r_[_Ek[:,1],_Ek[:,0]])),shape=(m,m)).tocsr()
    w1=np.minimum(1,(_Ak@(rimW>0).astype(float))>0)*0.6; w2=np.minimum(1,(_Ak@(w1>0).astype(float))>0)*0.3
    shrink=np.maximum(rimW,np.maximum(w1,w2))
    ax0=ax
    radv=(V-oc)-np.outer((V-oc)@ax0,ax0); radv/=np.maximum(np.linalg.norm(radv,axis=1,keepdims=True),1e-6)
    V=V-radv*(0.014*shrink[inv])[:,None]
    Vo=V.astype(np.float32).copy()
    LA,aa=walk(lowL); LB,bb=walk(upL)
    print('rim walk lengths',len(LA),len(LB))
    uvFill=UVmod[np.where(orange)[0][0]]
    def dup(ids):
        base=len(Vo); return base+np.arange(len(ids))
    dA=len(Vo)+np.arange(len(LA)); dB=len(Vo)+len(LA)+np.arange(len(LB))
    allI=np.r_[LA,LB]
    _rn=(V[allI]-oc)-np.outer((V[allI]-oc)@ax0,ax0); _rn/=np.linalg.norm(_rn,axis=1,keepdims=True)
    Vo=np.vstack([Vo,V[allI].astype(np.float32)]); No=np.vstack([No,_rn.astype(np.float32)]); UVo=np.vstack([UVo,np.tile(uvFill,(len(allI),1))])
    Jo=np.vstack([Jo,J[allI]]); Wo=np.vstack([Wo,Wt[allI]])
    i=k=0; nA,nB=len(LA),len(LB)
    while i<nA or k<nB:
        na=aa[(i+1)%nA]+(1.0 if i+1>=nA else 0); nb=bb[(k+1)%nB]+(1.0 if k+1>=nB else 0)
        if k>=nB or (i<nA and na<=nb):
            tri=[dA[i%nA],dA[(i+1)%nA],dB[k%nB]]; i+=1
        else:
            tri=[dA[i%nA],dB[(k+1)%nB],dB[k%nB]]; k+=1
        P0,P1,P2=Vo[tri[0]],Vo[tri[1]],Vo[tri[2]]; fn=np.cross(P1-P0,P2-P0); outv=(P0+P1+P2)/3-oc; outv-=ax*(outv@ax)
        if fn@outv<0: tri=[tri[0],tri[2],tri[1]]
        newF.append(tri)
Fout=np.vstack([Fk,np.array(newF,dtype=Fk.dtype).reshape(-1,3)]) if newF else Fk
print('stitch faces',len(newF))
invo=np.r_[inv,inv[allI]] if newF else inv.copy()
def _bloops(Fs):
    E=np.concatenate([Fs[:,[0,1]],Fs[:,[1,2]],Fs[:,[2,0]]]); W=invo[E]; srt=np.sort(W,1)
    u,idx_,c=np.unique(srt,axis=0,return_index=True,return_counts=True); sel=idx_[c==1]
    nxt={}; orig={}
    for e in sel: nxt[W[e,0]]=W[e,1]; orig[W[e,0]]=E[e,0]
    seen=set(); loops=[]
    for st in list(nxt):
        if st in seen: continue
        L=[]; x=st
        while x in nxt and x not in seen: seen.add(x); L.append(orig[x]); x=nxt[x]
        loops.append(L)
    return loops
fillF=[]; extraV=[]; 
for L in _bloops(Fout)+[None]:
    if L is None: break
    if not (3<=len(L)<=40): continue
    L=np.array(L)
    _cc=Vo[L].mean(0)
    if np.min(np.linalg.norm(V[metal|orange]-_cc,axis=1))>0.07: continue   # only holes left by removed jewellery / old collar (eye and nose sockets stay open)
    src=L[np.argmax(C[np.minimum(L,len(C)-1)].sum(1))]
    base=len(Vo); k_=len(L)
    ctr=Vo[L].mean(0); nc=No[L].mean(0); nc/=max(np.linalg.norm(nc),1e-6)
    Vo=np.vstack([Vo,Vo[L],ctr[None]]); No=np.vstack([No,No[L],nc[None]])
    UVo=np.vstack([UVo,np.tile(UVo[src],(k_+1,1))]); Jo=np.vstack([Jo,Jo[L],Jo[src][None]]); Wo=np.vstack([Wo,Wo[L],Wo[src][None]])
    cI=base+k_
    for q in range(k_):
        a_=base+q; b_=base+(q+1)%k_
        fillF.append([b_,a_,cI]); fillF.append([a_,b_,cI])
if fillF: Fout=np.vstack([Fout,np.array(fillF,dtype=Fout.dtype)])
# second pass: any remaining small opening near the old collar / jewellery, filled from undirected rims (double-sided)
invo=np.r_[invo,np.arange(invo.max()+1,invo.max()+1+len(Vo)-len(invo))]
_E2=np.concatenate([Fout[:,[0,1]],Fout[:,[1,2]],Fout[:,[2,0]]]); _W2=np.sort(invo[_E2],1)
_u2,_ix2,_c2=np.unique(_W2,axis=0,return_index=True,return_counts=True)
_adj=_dd(list); _rep={}
for e in _ix2[_c2==1]:
    a_,b_=invo[_E2[e,0]],invo[_E2[e,1]]; _adj[a_].append(b_); _adj[b_].append(a_); _rep[a_]=_E2[e,0]; _rep[b_]=_E2[e,1]
_seen=set(); fill2=[]
for st in list(_adj):
    if st in _seen: continue
    seq=[st]; _seen.add(st); prev=None; x=st
    while True:
        nx=[y for y in _adj[x] if y!=prev and y not in _seen]
        if not nx: break
        prev,x=x,nx[0]; seq.append(x); _seen.add(x)
    if not (3<=len(seq)<=40): continue
    L=np.array([_rep[q] for q in seq]); cc=Vo[L].mean(0)
    if np.min(np.linalg.norm(V[metal|orange]-cc,axis=1))>0.1: continue
    src=L[0]; base=len(Vo); k_=len(L)
    nc=No[L].mean(0); nc/=max(np.linalg.norm(nc),1e-6)
    Vo=np.vstack([Vo,Vo[L],cc[None]]); No=np.vstack([No,No[L],nc[None]])
    UVo=np.vstack([UVo,np.tile(UVo[src],(k_+1,1))]); Jo=np.vstack([Jo,Jo[L],Jo[src][None]]); Wo=np.vstack([Wo,Wo[L],Wo[src][None]])
    for q in range(k_):
        a_=base+q; b_=base+(q+1)%k_; fill2+=[[b_,a_,base+k_],[a_,b_,base+k_]]
if fill2: Fout=np.vstack([Fout,np.array(fill2,dtype=Fout.dtype)])
print('second pass fill faces',len(fill2))
print('hole fill faces',len(fillF))
# ---------- skin under the collar sinks inside the band, so it can never show through it while he moves ----------
# (there was also a leftover lump of skin exactly where the tag ring hangs)
def nodeM(n):
    M=np.eye(4)
    if 'rotation' in n:
        x,y,z,w=n['rotation']; M[:3,:3]=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    if 'translation' in n: M[:3,3]=n['translation']
    return M
_PJ,_PB=load(os.path.join(ROOT,'source','parts_split.glb'))
_cn=[n for n in _PJ['nodes'] if n.get('name')=='tripo_part_7'][0]; _M=nodeM(_cn)
CP=acc(_PJ,_PB,_PJ['meshes'][_cn['mesh']]['primitives'][0]['attributes']['POSITION']).astype(np.float64)@_M[:3,:3].T+_M[:3,3]
_c=CP.mean(0); _ax=np.linalg.svd(CP-_c)[2][2]
_e1=np.array([1.,0,0])-_ax*_ax[0]; _e1/=np.linalg.norm(_e1); _e2=np.cross(_ax,_e1)
def _polar(P):
    d=P-_c; h=d@_ax; d=d-np.outer(h,_ax); return h,np.linalg.norm(d,axis=1),np.arctan2(d@_e2,d@_e1)
ch,cr,ca=_polar(CP); hlo,hhi=ch.min(),ch.max(); hmid=(hlo+hhi)/2; hhalf=(hhi-hlo)/2
vh,vr,va=_polar(Vo)
cand=np.where((vh>hlo-0.004)&(vh<hhi+0.004)&(vr<cr.max()+0.05))[0]
moved=0
for i in cand:
    da=np.abs((ca-va[i]+np.pi)%(2*np.pi)-np.pi)
    near=(da<0.16)
    if not near.any(): continue
    loc=near&(np.abs(ch-vh[i])<0.014)
    outer=cr[loc].max() if loc.any() else cr[near].max(); inner=cr[near].min()
    t=np.clip(1-abs(vh[i]-hmid)/hhalf,0,1); t=t*t*(3-2*t)              # 1 in the middle of the band, 0 at its edges
    lim=(inner-0.003)*t+(outer-0.005)*(1-t)
    if vr[i]>lim:
        d=Vo[i]-_c; hv=d@_ax; rad=d-hv*_ax
        Vo[i]=_c+hv*_ax+rad*(lim/vr[i]); moved+=1
print('skin verts tucked under collar',moved)
attrs={}
attrs['POSITION']=add_acc(Vo,5126,'VEC3',minmax=True)
attrs['NORMAL']=add_acc(No.astype(np.float32),5126,'VEC3')
attrs['TEXCOORD_0']=add_acc(UVo.astype(np.float32),5126,'VEC2')
attrs['JOINTS_0']=add_acc(Jo.astype(np.uint8),5121,'VEC4')
attrs['WEIGHTS_0']=add_acc(Wo.astype(np.float32),5126,'VEC4')
FF=Fout.astype(np.uint32)
ind=add_acc(FF.ravel(),5125,'SCALAR',target=34963)
imgs=[]
for name,data in [('basecolor',bcj),('roughness_metal',rmj),('normal',nmj)]:
    imgs.append({'name':name,'mimeType':'image/jpeg','bufferView':add_view(data)})
# ---------- accessories from the split model (rigid pieces, animated in the page) ----------
PJ,PBIN=load(os.path.join(ROOT,'source','parts_split.glb'))
ACC={'tripo_part_7':'Acc_Collar','tripo_part_10':'Acc_TagLink','tripo_part_17':'Acc_TagRing','tripo_part_8':'Acc_Tag',
     'tripo_part_6':'Acc_Hoop','tripo_part_16':'Acc_HoopLink','tripo_part_11':'Acc_Star','tripo_part_18':'Acc_StudL','tripo_part_18.001':'Acc_StudR','tripo_part_9':'Acc_EarRing'}
acc_meshes=[]; acc_mats=[]; acc_imgs=[]
for n in PJ['nodes']:
    if n.get('name') not in ACC: continue
    pr=PJ['meshes'][n['mesh']]['primitives'][0]; M=nodeM(n)
    P=acc(PJ,PBIN,pr['attributes']['POSITION']).astype(np.float64)@M[:3,:3].T+M[:3,3]
    N=acc(PJ,PBIN,pr['attributes']['NORMAL']).astype(np.float64)@M[:3,:3].T
    UVa=acc(PJ,PBIN,pr['attributes']['TEXCOORD_0']).astype(np.float32)
    I=acc(PJ,PBIN,pr['indices']).astype(np.uint32)
    pm=PJ['materials'][pr['material']]; ti=pm['pbrMetallicRoughness']['baseColorTexture']['index']
    im=img(PJ,PBIN,PJ['textures'][ti]['source']).convert('RGB'); sz=min(im.size[0],512 if ACC[n['name']]=='Acc_Collar' else 256)
    acc_imgs.append(jpeg(im,sz,90))
    acc_mats.append({'name':ACC[n['name']]+'_Mat','pbrMetallicRoughness':{'baseColorTexture':{'index':3+len(acc_imgs)-1},
        'metallicFactor':pm['pbrMetallicRoughness'].get('metallicFactor',1.0),'roughnessFactor':pm['pbrMetallicRoughness'].get('roughnessFactor',1.0)}})
    acc_meshes.append((ACC[n['name']],P.astype(np.float32),N.astype(np.float32),UVa,I))
# skeleton nodes
nodes=[{'name':'Bullterrier','children':[]},{'name':'Body_Mesh','mesh':0,'skin':0}]
base=len(nodes); nodes[0]['children'].append(1)
for k in names:
    par,pos=B[k]; ppos=B[par][1] if par else np.zeros(3)
    nodes.append({'name':k,'translation':(pos-ppos).tolist()})
for k in names:
    par,_=B[k]
    if par: nodes[base+idx[par]].setdefault('children',[]).append(base+idx[k])
    else: nodes[0]['children'].append(base+idx[k])
acc_nodes=[]
for k,(nm,P,N,UVa,I) in enumerate(acc_meshes):
    at={'POSITION':add_acc(P,5126,'VEC3',minmax=True),'NORMAL':add_acc(N,5126,'VEC3'),'TEXCOORD_0':add_acc(UVa,5126,'VEC2')}
    ia=add_acc(I,5125,'SCALAR',target=34963)
    acc_nodes.append({'name':nm,'mesh':1+k,'_prim':{'attributes':at,'indices':ia,'material':1+k}})
ibm=np.zeros((len(names),16),np.float32)
for k in names:
    M=np.eye(4); M[:3,3]=-B[k][1]; ibm[idx[k]]=M.T.ravel()
ibmA=add_acc(ibm,5126,'MAT4',target=None)
mat=j['materials'][0]; mat['name']='Bullterrier_Vinyl'
mat['pbrMetallicRoughness']['baseColorFactor']=[1,1,1,1]
mat.pop('extensions',None)
for k,data in enumerate(acc_imgs): imgs.append({'name':acc_meshes[k][0]+'_basecolor','mimeType':'image/jpeg','bufferView':add_view(data)})
for an in acc_nodes:
    nodes.append({'name':an['name'],'mesh':an['mesh']}); nodes[0]['children'].append(len(nodes)-1)
doc={'asset':{'version':'2.0','generator':'bullterrier-rig'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':nodes,
 'meshes':[{'name':'Bullterrier','primitives':[{'attributes':attrs,'indices':ind,'material':0}]}]+[{'name':an['name'],'primitives':[an['_prim']]} for an in acc_nodes],
 'skins':[{'name':'Bullterrier_Rig','joints':[base+idx[k] for k in names],'inverseBindMatrices':ibmA,'skeleton':base}],
 'materials':[mat]+acc_mats,'textures':[{'source':i,'sampler':0} for i in range(3+len(acc_imgs))],
 'samplers':[{'magFilter':9729,'minFilter':9987,'wrapS':10497,'wrapT':10497}],'images':imgs,
 'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':len(out)}]}
js=json.dumps(doc,separators=(',',':')).encode()
while len(js)%4: js+=b' '
while len(out)%4: out.append(0)
glb=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(out))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(out),0x004E4942)+bytes(out)
open(os.path.join(ROOT,'dog.glb'),'wb').write(glb)
print('bones',names); print('size MB',round(len(glb)/1e6,2))

