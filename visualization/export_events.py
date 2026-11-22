"""Export actual Gymnasium transitions for the Three.js playback viewer.

Run: uv run python -m visualization.export_events --seed 42 --policy heuristic
"""
import argparse
import json
from pathlib import Path
from environment.custom_env import MediFlightEnv, SITES

BASE = Path(__file__).resolve().parent

def snapshot(env):
    return {
        'time': round(env.time, 3), 'battery': round(env.battery, 3),
        'weather': ('calm','breezy','windy')[env.weather],
        'completed': env.completed, 'missed': env.missed,
        'distance': round(env.total_distance, 3),
        'orders': [None if o is None else {k:round(v,3) if isinstance(v,float) else v for k,v in o.items()} for o in env.orders],
    }

def heuristic(env):
    # Deadline-aware dispatch baseline; not a learned policy.
    feasible=[]
    for i,o in enumerate(env.orders):
        if o is None or o['weight'] > env.capacity_kg: continue
        distance=2*env._distance(i)
        wind=[1.,1.2,1.5][env.weather]
        energy=distance*2*wind+3
        if env.battery < energy+5: continue
        travel=distance/.75*wind+3
        arrival=env.time+(travel-3)/2
        slack=o['deadline']-arrival
        feasible.append((slack, -o['urgency'], i+1))
    if feasible:
        return min(feasible)[2]
    if env.battery < 85: return env.n+2
    return 0

def export(seed=42, policy='heuristic', output=None, max_steps=200):
    env=MediFlightEnv()
    env.reset(seed=seed)
    rng=__import__('numpy').random.default_rng(seed+1000)
    data={
      'version':1,'title':'MediFlight AI — Medical Delivery Replay',
      'metadata':{'seed':seed,'policy':policy,'horizon':env.horizon,'model':'single-drone round-trip dispatcher','units':'km, minutes, battery %'},
      'base':{'name':'Medical Distribution Center','x':0,'y':0},
      'sites':[{'name':s.name,'kind':s.kind,'x':s.x,'y':s.y} for s in SITES],
      'initial':snapshot(env),'events':[]
    }
    for step in range(max_steps):
        before=snapshot(env)
        action=heuristic(env) if policy=='heuristic' else int(rng.integers(env.action_space.n))
        _,reward,terminated,truncated,info=env.step(action)
        after=snapshot(env)
        # Successful or late dispatches include the full outward/return journey.
        flight=info['event'] in ('delivered','late_delivery')
        data['events'].append({
          'step':step,'action':action,'destination':action-1 if flight else None,
          'flight':flight,'event':info['event'],'reward':round(reward,3),
          'start_time':before['time'],'end_time':after['time'],
          'before':before,'after':after,
        })
        if terminated or truncated:break
    data['summary']={'steps':len(data['events']),'reward':round(sum(e['reward'] for e in data['events']),2),**snapshot(env)}
    out=Path(output) if output else BASE/'events.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(f'Exported {len(data["events"])} transitions to {out}')
    print('Summary:',data['summary']['completed'],'on-time deliveries,',data['summary']['missed'],'missed, reward',data['summary']['reward'])
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--policy',choices=['heuristic','random'],default='heuristic')
    p.add_argument('--output',type=Path)
    p.add_argument('--max-steps',type=int,default=200)
    args=p.parse_args()
    export(args.seed,args.policy,args.output,args.max_steps)
