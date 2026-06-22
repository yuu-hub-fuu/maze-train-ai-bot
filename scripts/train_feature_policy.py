from __future__ import annotations

import argparse
from collections import deque
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "ATTACK_BOSS", "STOP"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
TILES = ["#", ".", "S", "E", "B", "G", "T", "@"]; TILE_TO_ID = {t:i for i,t in enumerate(TILES)}


def read_jsonl(path):
    rows=[]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r=json.loads(line)
                if r.get('action') in ACTION_TO_ID:
                    rows.append(r)
    return rows


def cell(grid, pos):
    r,c=pos
    if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]): return '#'
    return grid[r][c]


def visible_cell(grid, state, pos):
    ch=cell(grid,pos)
    tup=[pos[0],pos[1]]
    if ch=='G' and tup in state.get('collected',[]): return '.'
    if ch=='T' and tup in state.get('triggered',[]): return '.'
    if ch=='B' and state.get('boss_defeated', False): return '.'
    return ch


def bfs_dist(grid, state, start, targets):
    q=deque([(tuple(start),0)]); seen={tuple(start)}
    while q:
        cur,d=q.popleft()
        if d>0 and visible_cell(grid,state,cur) in targets:
            return d
        for dr,dc in DELTAS.values():
            nxt=(cur[0]+dr,cur[1]+dc)
            if nxt in seen or cell(grid,nxt)=='#': continue
            seen.add(nxt); q.append((nxt,d+1))
    return 99


def one_hot(idx, n):
    v=[0.0]*n
    if 0 <= idx < n: v[idx]=1.0
    return v


def features(rec):
    grid=rec['maze']['grid']; state=rec['state']; pos=state['pos']
    gold=float(state['gold']); steps=float(state['steps']); boss_cost=float(rec['maze'].get('boss_cost',150))
    feat=[gold/250.0, steps/100.0, (boss_cost-gold)/250.0, 1.0 if state.get('boss_defeated') else 0.0]
    for rr in range(pos[0]-1,pos[0]+2):
        for cc in range(pos[1]-1,pos[1]+2):
            ch='@' if [rr,cc]==pos else visible_cell(grid,state,(rr,cc))
            feat += one_hot(TILE_TO_ID.get(ch,1), len(TILES))
    for action in ACTIONS:
        if action in DELTAS:
            dr,dc=DELTAS[action]; nxt=(pos[0]+dr,pos[1]+dc); ch=visible_cell(grid,state,nxt)
            feat += one_hot(TILE_TO_ID.get(ch,1), len(TILES))
            feat.append(0.0 if ch=='#' else 1.0)
        elif action=='ATTACK_BOSS':
            boss_positions=[(r,c) for r,row in enumerate(grid) for c,ch in enumerate(row) if ch=='B']
            near=any(abs(pos[0]-r)+abs(pos[1]-c)<=1 for r,c in boss_positions)
            feat += [1.0 if near and not state.get('boss_defeated') else 0.0]* (len(TILES)+1)
        else:
            feat += [0.0]*(len(TILES)+1)
    for targets in [{'G'},{'T'},{'B'},{'E'}]:
        d=bfs_dist(grid,state,pos,targets)
        feat.append(min(d,99)/99.0)
    return feat


class FeatureDataset(Dataset):
    def __init__(self, rows):
        self.x=[torch.tensor(features(r), dtype=torch.float32) for r in rows]
        self.y=[torch.tensor(ACTION_TO_ID[r['action']], dtype=torch.long) for r in rows]
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i], self.y[i]


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256,128), nn.ReLU(), nn.Linear(128,len(ACTIONS)))
    def forward(self,x): return self.net(x)


def eval_model(model, loader, device):
    model.eval(); ok=tot=0; loss_sum=0.0; ce=nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device); logits=model(x); loss=ce(logits,y)
            loss_sum += loss.item()*y.numel(); ok += (logits.argmax(-1)==y).sum().item(); tot += y.numel()
    return {'loss': loss_sum/max(1,tot), 'accuracy': ok/max(1,tot)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data', default='train/sft_records.jsonl'); ap.add_argument('--out', default='artifacts/torch_policy/maze_mlp_policy.pt'); ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--batch-size', type=int, default=128); args=ap.parse_args()
    random.seed(42); torch.manual_seed(42)
    rows=read_jsonl(args.data); random.shuffle(rows); split=int(len(rows)*0.9); train=rows[:split]; val=rows[split:]
    train_ds=FeatureDataset(train); val_ds=FeatureDataset(val); in_dim=train_ds.x[0].numel()
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True); val_loader=DataLoader(val_ds,batch_size=args.batch_size)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=MLP(in_dim).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); ce=nn.CrossEntropyLoss()
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); best={'accuracy':-1}
    print(json.dumps({'device':str(device),'cuda':torch.cuda.is_available(),'rows':len(rows),'in_dim':in_dim},ensure_ascii=False), flush=True)
    for epoch in range(1,args.epochs+1):
        model.train(); total=0; loss_sum=0.0
        for x,y in train_loader:
            x=x.to(device); y=y.to(device); opt.zero_grad(); logits=model(x); loss=ce(logits,y); loss.backward(); opt.step(); loss_sum += loss.item()*y.numel(); total += y.numel()
        valm=eval_model(model,val_loader,device); row={'epoch':epoch,'train_loss':loss_sum/max(1,total),**valm}; print(json.dumps(row,ensure_ascii=False), flush=True)
        if valm['accuracy']>best['accuracy']:
            best=valm; torch.save({'model':model.state_dict(),'actions':ACTIONS,'in_dim':in_dim,'metrics':best}, out)
    out.with_suffix('.metrics.json').write_text(json.dumps({'best':best,'rows':len(rows),'in_dim':in_dim},ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved '+str(out), flush=True)

if __name__=='__main__': main()
