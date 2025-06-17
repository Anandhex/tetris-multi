import random
import numpy as np
import torch
from keras.models import Sequential
from keras.layers import Dense
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import os
from datetime import datetime

class TetrisPiece:
    """Static definitions of all tetromino shapes and rotations."""
    PIECES = {
        'I': [
            [[1,1,1,1]],
            [[1],[1],[1],[1]]
        ],
        'O': [
            [[1,1],[1,1]]
        ],
        'T': [
            [[0,1,0],[1,1,1]],
            [[1,0],[1,1],[1,0]],
            [[1,1,1],[0,1,0]],
            [[0,1],[1,1],[0,1]]
        ],
        'S': [
            [[0,1,1],[1,1,0]],
            [[1,0],[1,1],[0,1]]
        ],
        'Z': [
            [[1,1,0],[0,1,1]],
            [[0,1],[1,1],[1,0]]
        ],
        'J': [
            [[1,0,0],[1,1,1]],
            [[1,1],[1,0],[1,0]],
            [[1,1,1],[0,0,1]],
            [[0,1],[0,1],[1,1]]
        ],
        'L': [
            [[0,0,1],[1,1,1]],
            [[1,0],[1,0],[1,1]],
            [[1,1,1],[1,0,0]],
            [[1,1],[0,1],[0,1]]
        ]
    }

    @staticmethod
    def get_shape(piece_type: str, rotation: int) -> list:
        rotations = TetrisPiece.PIECES.get(piece_type, [[[1]]])
        return rotations[rotation % len(rotations)]

class TetrisEnvironment:
    """Simulates Tetris board dynamics without rendering."""
    def __init__(self, height=20, width=10):
        self.height = height
        self.width = width

    def decode_action(self, action: int, piece_type: str) -> tuple:
        rotation = action // self.width
        target = action % self.width
        shape = TetrisPiece.get_shape(piece_type, rotation)
        # find leftmost block index
        leftmost = min(c for r in shape for c, v in enumerate(r) if v)
        position = target - leftmost
        return position, rotation

    def can_place(self, board: list, shape: list, pos: int) -> tuple:
        b = np.array(board).reshape(self.height, self.width)
        ph, pw = len(shape), len(shape[0])
        if pos < 0 or pos+pw > self.width or ph>self.height:
            return False, -1
        for drop in range(self.height - ph + 1):
            coll = False
            for i in range(ph):
                for j in range(pw):
                    if shape[i][j]:
                        if b[drop+i, pos+j]: coll=True; break
                if coll: break
            if coll:
                return (False, -1) if drop==0 else (True, drop-1)
        return True, self.height - ph

    def place(self, board: list, shape: list, pos: int, row: int) -> list:
        b = np.array(board).reshape(self.height, self.width).copy()
        for i in range(len(shape)):
            for j in range(len(shape[0])):
                if shape[i][j]: b[row+i, pos+j]=1
        return b.flatten().tolist()

    def clear_lines(self, board: list) -> tuple:
        b = np.array(board).reshape(self.height, self.width)
        new = [list(r) for r in b if not all(r)]
        cleared = self.height - len(new)
        for _ in range(cleared): new.insert(0, [0]*self.width)
        return sum(1 for _ in range(cleared)), np.array(new).flatten().tolist()

    def column_heights(self, board: list) -> list:
        b = np.array(board).reshape(self.height, self.width)
        return [self.height - np.argmax(col!=0) if any(col!=0) else 0 for col in b.T]

    def count_holes(self, board: list) -> int:
        b = np.array(board).reshape(self.height, self.width)
        holes=0
        for col in b.T:
            filled = False
            for cell in col:
                if cell: filled=True
                elif filled: holes+=1
        return holes

    def bumpiness(self, heights: list) -> int:
        return sum(abs(heights[i]-heights[i+1]) for i in range(len(heights)-1))

class DQNAgent:
    """Deep Q-Network agent for Tetris."""
    def __init__(self, state_size=4, mem_size=10000, discount=0.95,
                 epsilon=1.0, min_epsilon=0.01, decay_steps=10000,
                 hidden=[32,32], activations=['relu','relu','linear'],
                 loss='mse', opt='adam', log_dir=None, model_file=None):
        self.state_size=state_size
        self.memory=deque(maxlen=mem_size)
        self.discount=discount
        self.epsilon=epsilon; self.min_epsilon=min_epsilon
        self.epsilon_decay=(epsilon-min_epsilon)/decay_steps
        self.env=TetrisEnvironment()
        self.pieces=list(TetrisPiece.PIECES.keys())
        self.model=self._build_model(hidden, activations, loss, opt)
        self.writer=SummaryWriter(log_dir or f"runs/tetris_{datetime.now():%Y%m%d_%H%M%S}")

    def _build_model(self, hidden, acts, loss, opt):
        m=Sequential()
        m.add(Dense(hidden[0], input_dim=self.state_size, activation=acts[0]))
        for h,a in zip(hidden[1:], acts[1:]): m.add(Dense(h,activation=a))
        m.add(Dense(1, activation=acts[-1]))
        m.compile(loss=loss, optimizer=opt)
        return m

    def simulate(self, state: dict, action: int) -> dict:
        board=state['board'][:]
        idx=int(state['currentPiece'][0])
        shape=self.env.decode_action(action,self.pieces[idx])[1]  # wrong
        # decode correctly:
        pos, rot=self.env.decode_action(action,self.pieces[idx])
        shape=TetrisPiece.get_shape(self.pieces[idx], rot)
        can,row=self.env.can_place(board,shape,pos)
        if not can:
            sim=state.copy(); sim.update({'invalid':True,'linesCleared':0,'action':action}); return sim
        new_b=self.env.place(board,shape,pos,row)
        lines,new_b=self.env.clear_lines(new_b)
        heights=self.env.column_heights(new_b)
        holes=self.env.count_holes(new_b)
        bump=self.env.bumpiness(heights)
        return {'board':new_b,'linesCleared':lines,'heights':heights,
                'holes':holes,'bumpiness':bump,'action':action,'invalid':False}

    def evaluate(self, state, action):
        sim=self.simulate(state, action)
        if sim['invalid']: return [0,0,0,0]
        return [sim['linesCleared'], sim['holes'], sim['bumpiness'], sum(sim['heights'])]

    def get_actions(self, state):
        idx=int(state['currentPiece'][0]); piece=self.pieces[idx]
        acts=[]; feats=[]
        for rot in range(len(TetrisPiece.PIECES[piece])):
            shape=TetrisPiece.get_shape(piece,rot)
            pw=len(shape[0])
            left=min(c for r in shape for c,v in enumerate(r) if v)
            for tgt in range(self.env.width):
                pos=tgt-left
                can,_=self.env.can_place(state['board'],shape,pos)
                if not can: continue
                action=rot*self.env.width+tgt
                acts.append(action)
                feats.append(self.evaluate(state,action))
        return list(zip(acts,feats))

    def act(self, state_vec):
        if np.random.rand()<self.epsilon: return np.random.rand()
        return float(self.model.predict(state_vec.reshape(1,-1),verbose=0))

    def remember(self,s,n,r,d): self.memory.append((s,n,r,d))
    def replay(self,batch=32, epochs=1):
        if len(self.memory)<batch: return
        samp=random.sample(self.memory,batch)
        ns=[x[1] for x in samp]
        qn=self.model.predict(np.array(ns)).flatten()
        x=[]; y=[]
        for i,(s,_,r,d) in enumerate(samp):
            target=r if d else r+self.discount*qn[i]
            x.append(s); y.append(target)
        self.model.fit(np.array(x),np.array(y),batch_size=batch,epochs=epochs,verbose=0)
        if self.epsilon>self.min_epsilon: self.epsilon-=self.epsilon_decay

# Utility functions for state creation
def make_empty_state(piece_idx, h=20, w=10):
    return {'board':[0]*(h*w),'currentPiece':[piece_idx,0],'height':h,'width':w}

def make_custom_state(board_list,piece_idx,h=20,w=10):
    return {'board':board_list.copy(),'currentPiece':[piece_idx,0],'height':h,'width':w}

# Example test
if __name__=='__main__':
    agent=DQNAgent()
    state=make_empty_state(0)
    actions=agent.get_actions(state)
    print(f"Generated {len(actions)} possible moves for I-piece on empty board.")
