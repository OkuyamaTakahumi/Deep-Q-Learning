# coding:utf-8
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.externals import joblib
from chainer import cuda, Variable, FunctionSet,\
   optimizers,Chain,serializers
import chainer.functions  as F
import cv2

#import sys
import datetime
#import os
from collections import deque
import copy

import Visualize

class Forward_Enviroment():
  def __init__(self,Train):
    # parameters
    #self.name = os.path.splitext(os.path.basename(__file__))[0]
    self.screen_n_y = 16
    self.screen_n_x = 16
    self.player_length = 1
    #self.frame_rate = 5
    self.Train=Train
    if not Train:
      self.player_x = np.random.randint(self.screen_n_x - self.player_length)

  def reset(self):
    # reset player position
    self.player_y = self.screen_n_y - 1
    if self.Train:
      self.player_x = np.random.randint(self.screen_n_x - self.player_length)
    #self.player_x=7

    # reset goal position
    self.goal_y = 0
    self.goal_x = np.random.randint(self.screen_n_x)
    #self.goal_x = 2
    # reset other variables
    self.reward = 0
    self.ep_end = False
    self.draw()
    return self.screen

  def step(self, action):
    # update player position
    if action == 'left':
      # move left
      self.player_x = max(0, self.player_x - 1)
    elif action == 'right':
      # move right
      self.player_x = min(self.player_x + 1, self.screen_n_x - self.player_length)
    elif action == 'up':
      # do nothing
      pass
    # update goal position
    self.goal_y += 1
    # xlision detection
    self.reward = 0
    self.ep_end = False
    if self.goal_y == self.player_y:
        self.ep_end = True
        # self.player_x == self.goal_xと同じ意味
        if self.player_x == self.goal_x :
            # goal
            self.reward = 10
        else:
            # 衝突
            self.reward = -10
    self.draw()
    return self.screen, self.reward, self.ep_end
  def draw(self):
      # reset screen
      self.screen = np.zeros((self.screen_n_y, self.screen_n_x))
      kabe = np.ones((self.screen_n_x))/2.0
      self.screen[self.goal_y]=kabe
      # draw goal
      self.screen[self.goal_y, self.goal_x] = 0
      # draw player
      self.screen[self.player_y, self.player_x:self.player_x + self.player_length] = 1
  def reset_kokuhuku(self):
    # reset player position
    self.player_y = self.screen_n_y - 1
    if self.Train:
      self.player_x = np.random.randint(self.screen_n_x - self.player_length)
    self.player_x = 7

    # reset goal position
    self.goal_y = 0
    #self.goal_x = np.random.randint(self.screen_n_x)
    self.goal_x = np.random.randint(0,3)
    #self.goal_x = 0
    # reset other variables
    self.reward = 0
    self.ep_end = False
    self.draw()
    return self.screen
  def draw_Kantyou(self):
    n_y = self.screen_n_y
    n_x = self.screen_n_x

    yubi_y = self.player_y + 16
    yubi_x = self.player_x + 16 -13

    ue_ketu_y = self.goal_y+16
    ketu_y = self.goal_y+16
    ketu_x = self.goal_x +16 -8



    Yubi = cv2.imread('Yubi.jpg')
    Ketu=cv2.imread('Ketu.jpg')

    ueKetu = Ketu[0:14 , : , :]
    sitaKetu = Ketu[14:16 , : , :]

    screen = np.zeros((16+n_y+16,16+n_x+16 ,3))


    #screen[ketu_y-14:ketu_y , ketu_x:ketu_x+16 , :] = ueKetu
    screen[yubi_y:yubi_y+16 , yubi_x:yubi_x+16 , :] = Yubi
    #screen[ketu_y:ketu_y+2 , ketu_x:ketu_x+16 , :] = sitaKetu
    screen[ketu_y-14:ketu_y+2 , ketu_x:ketu_x+16 , :] = Ketu

    screen[yubi_y,yubi_x+13,:] = Yubi[0,13,:]
    screen[yubi_y+1 , yubi_x+12:yubi_x+15 ,:] = Yubi[1 , 12:15 ,:]

    return screen


class DQN_NN(object):
  def __init__(self,n_act):
    self.N_input = 64
    N_output = n_act
    #N_unit = (self.N_input-1)*2
    N_unit = 64
    self.model = FunctionSet(
      l1=F.Linear(self.N_input,N_unit),
      #l2=F.Linear(N_unit, N_unit),
      #l3=F.Linear(N_unit, N_unit),
      l4=F.Linear(N_unit, N_output,initialW=np.zeros((N_output, N_unit), dtype=np.float32)))
  def Q_func(self,x):
    N,h,w=x.shape
    x=x.reshape(N,h*w)
    x = Variable(x)
    h = F.leaky_relu(self.model.l1(x))
    #h = F.leaky_relu(self.model.l2(h))
    #h = F.leaky_relu(self.model.l3(h))
    y = self.model.l4(h)
    return y
class DQN_CNN(object):
  def __init__(self,n_act):
    N_output = n_act
    self.model = FunctionSet(
      conv1=F.Convolution2D(1, 16, 3, pad=1),
      conv2=F.Convolution2D(16, 16, 3, pad=1),
      l1=F.Linear(1024, 256),
      l2=F.Linear(256, N_output))
  def Q_func(self,x):
    N,h,w=x.shape
    x=x.reshape(N,1,h,w)
    x = Variable(x)
    h = F.relu(self.model.conv1(x))
    h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
    h = F.relu(self.model.l1(h))
    y = self.model.l2(h)
    return y

class Agent():
  def __init__(self,n_act,Train):
    self.n_act = n_act
    #self.NN = DQN_NN(n_act)
    self.NN = DQN_CNN(n_act)
    self.target_NN = copy.deepcopy(self.NN)
    self.optimizer = optimizers.Adam()
    self.optimizer.setup(self.NN.model)
    self.mem_size = 1000 # Experience Replayのために覚えておく経験の数
    self.memory = deque(maxlen=self.mem_size)
    self.gamma = 0.99 # 割引率
    self.batch_size = 32 # Experience Replayの際のミニバッチの大きさ
    #self.train_freq = 100 # ニューラルネットワークの学習間隔
    self.target_update_freq = 1 # ターゲットネットワークの同期間隔
    # ε-greedy
    self.epsilon_min = 0.1 # εの最小値
    if Train:
      self.epsilon = 1.0# εの初期値->小さいほどgreedy
      self.epsilon_decay = 0.001 # εの減衰値
    else:
      self.epsilon = 0 # εの初期値->小さいほどgreedy
      self.epsilon_decay = 0 # εの減衰値

    #self.exploration = 1000 # εを減衰し始めるまでのステップ数(今回はメモリーが貯まるまで)

  def get_action_test(self, st):
    #各行動の価値が含まれた(n_act,)のarrayを返す
    if np.random.rand() < self.epsilon:
      return np.random.rand(self.n_act)
      #return np.array([np.random.rand(), 0, 6.0/7])
    else:
      #-----NN用のデータ準備[float32->reshape(N,1,h,w)]
      st = np.array([st], dtype=np.float32)
      #st = st.reshape(1,9)
      Q = self.NN.Q_func(st)
      return Q.data[0]

  def load_model(self,model_name):
    serializers.load_npz('./modelKeep/'+model_name,self.NN.model)

def test(action_list,model_name):
  n_act = len(action_list)
  env=Forward_Enviroment(False)
  agent=Agent(n_act,False)
  agent.load_model(model_name)
  step=1
  n_episode=50
  meiroset=deque()
  shoutotu=0
  kaihi=0
  up=0
  right=0
  left=0
  for i_episode in range(n_episode):
    st=env.reset()
    #meiroset.append(st)
    meiroset.append(env.draw_Kantyou())
    ep_end = False
    while not ep_end:
      #act_i_array=agent.get_action_test(st)
      #act_i= act_i_array.argmax()
      act_i = np.random.randint(0,n_act)
      #act_i = 0
      action=action_list[act_i]
      if action == 'left':
        left+=1
      elif action == 'right':
        right+=1
      elif action == 'up':
        up+=1
      observation,reward,ep_end=env.step(action)
      st = observation
      #meiroset.append(st)
      meiroset.append(env.draw_Kantyou())
    if reward < 0:
      shoutotu+=1
      #print u'衝突!!'
    else:
      kaihi+=1
      #print u'回避!!'


  print u"テスト回数 = %d"%(n_episode)
  print u"衝突回数 = %d , 回避回数 = %d"%(shoutotu,kaihi)
  kaihiritu = float(kaihi)/n_episode*100
  print u"回避率 = "+str(kaihiritu)+"%"
  print u"UP :%d回 , RIGHT :%d回 , LEFT :%d回\n"%(up,right,left)
  Visualize.Make_animation(list(meiroset)[0:25*16])

if __name__ == '__main__':
  action_list=['up','right','left']
  #model=['model_8_forward_MLP']
  #model=['model_8_forward_CNN','model_8_forward_CNN_burebure','model_8_forward_CNN_manyUP']
  model=['model_16_forward_98%']
  for model_name in model:
    print "----------------------------------------------"
    print u'MODEL : ',model_name
    test(action_list,model_name)
    print "----------------------------------------------"
