from scripts import imitate_mj, video_mj

#def train(mode='ga', env='Acrobot-v0', data='imitation_runs/trajs/trajs_acrobot.h5', num_trajs=1):
def train(args):
    args = vars(args)
    #args = {'mode':mode, 'env':env, 'data':data, 'num_trajs':num_trajs}
    args['log'] = "imitation_runs/checkpoints/alg=%s,task=%s,num_trajs=%d,run=0.h5"%(args['mode'], args['env'], args['num_trajs'])
    imitate_mj.main(args)

def run(args):
    #video_mj.main("imitation_runs/checkpoints/alg=ga,task=Acrobot-v0,num_trajs=1,run=0.h5/snapshots/iter0000280", "Acrobot-v0")
    video_mj.main("imitation_runs/checkpoints/alg=%s,task=%s,num_trajs=%d,run=0.h5/snapshots/iter0000280"%(args.mode, args.env, args.num_trajs), args.env)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train and/or run an imitation learning algorithm.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--mode', type=str, default='ga', help='Which imitation learning algorithm to run {ga}')
    parser.add_argument('--env', type=str, default='Acrobot-v0', help='Which OpenAI Gym environment to use')
    parser.add_argument('--data', type=str, default='imitation_runs/trajs/trajs_acrobot.h5', help='Trajectory file')
    parser.add_argument('--num_trajs', type=int, default=1, help='Number of trajectories')
    args = parser.parse_args()
    if args.train:
        train(args)
    else:
        run(args)

if __name__ == '__main__':
    main()
