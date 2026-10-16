import json,struct,numpy as np,io
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
def load(path):
    b=open(path,'rb').read()
    L=struct.unpack('<I',b[12:16])[0]; j=json.loads(b[20:20+L]); bs=20+L; BL=struct.unpack('<I',b[bs:bs+4])[0]; BIN=b[bs+8:bs+8+BL]
    return j,BIN
def acc(j,BIN,i):
    a=j['accessors'][i]; bv=j['bufferViews'][a['bufferView']]; off=bv.get('byteOffset',0)+a.get('byteOffset',0)
    n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]; dt={5126:np.float32,5125:np.uint32,5123:np.uint16,5121:np.uint8}[a['componentType']]
    arr=np.frombuffer(BIN,dtype=dt,count=a['count']*n,offset=off)
    return arr.reshape(-1,n) if n>1 else arr
def img(j,BIN,i):
    bv=j['bufferViews'][j['images'][i]['bufferView']]; o=bv.get('byteOffset',0)
    return Image.open(io.BytesIO(BIN[o:o+bv['byteLength']]))
