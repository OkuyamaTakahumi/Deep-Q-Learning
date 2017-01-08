# coding:utf-8
#%matplotlib nbagg
#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime

def to_plot(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def to_grayscale(img):
  grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return grayed
def draw_Meiro(meiro,title):
  if np.max(meiro) <= 1.0:
    meiro*=255
  meiro=meiro.astype(np.uint8)
  plt.title(title,size=10)
  plt.tick_params(labelbottom="off")
  plt.tick_params(labelleft="off")
  if len(meiro.shape)==3 :
    if meiro.shape[0]==1:
      h,w=meiro.shape[1:]
      meiro=meiro.reshape(h,w)
      draw_Meiro(meiro,title)
    elif meiro.shape[0]==3:
      meiro = meiro.transpose(1,2,0)
      draw_Meiro(meiro,title)
    else:
      plt.imshow(to_plot(meiro), interpolation="none")
  else:
      plt.gray()
      #meiro=meiro[::-1,:]
      plt.imshow(meiro, interpolation="none")
def draw_Meiro_set(Meiroset,titleset,yoko=8,tate=5):
  minisetsize=tate*yoko
  for val in range(0,len(Meiroset),minisetsize):
    Meiro_mini_set=list(Meiroset)[val:val+minisetsize]
    title_mini_set=list(titleset)[val:val+minisetsize]
    #tate=int(len(Meiro_mini_set)/yoko)
    plt.figure(figsize=(10,10))
    for idx in xrange(len(Meiro_mini_set)):
      Meiro = Meiro_mini_set[idx]
      title = title_mini_set[idx]
      plt.subplot(tate, yoko, idx+1)
      draw_Meiro(Meiro,title)
    plt.show()
def Make_animation(Meiroset):
    fig = plt.figure()
    ims=[]
    for idx in xrange(len(Meiroset)):
      Meiro = Meiroset[idx]
      if np.max(Meiro) <= 1.0:
        Meiro*=255
      Meiro=Meiro.astype(np.uint8)
      plt.gray()
      im = [plt.imshow(Meiro, interpolation="none")]
      ims.append(im)
    ani = animation.ArtistAnimation(fig, ims)
    hiduke = datetime.datetime.now().strftime("%m-%d-%H-%M")
    print u"gifを保存中"
    ani.save("GIF/%s.gif"%(hiduke), writer="imagemagick")
