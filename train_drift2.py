import argparse
import os
import sys
import yaml

from datetime import datetime
from pytorch_lightning import Trainer, callbacks, loggers
import wandb

from src.lightning import Drift2
from src.utils import disable_rdkit_logging, set_deterministic, Logger

def setup_wandb():
    """Setup wandb login using API key from file"""
    try:
        # Try to read API key from file
        api_key_file = 'wandb_api_key.txt'
        if os.path.exists(api_key_file):
            with open(api_key_file, 'r') as f:
                api_key = f.read().strip()
            wandb.login(key=api_key)
            print("Successfully logged into Weights & Biases")
        else:
            print(f"Warning: {api_key_file} not found. Please create it with your API key or login manually.")
            print("You can also set WANDB_API_KEY environment variable.")
    except Exception as e:
        print(f"Warning: Failed to login to wandb: {e}")
        print("Training will continue without wandb logging.")

def find_last_checkpoint(checkpoints_dir):
    epoch2fname = [
        (int(fname.split('=')[1].split('.')[0]), fname)
        for fname in os.listdir(checkpoints_dir)
        if fname.endswith('.ckpt') and '=' in fname
    ]
    latest_fname = max(epoch2fname, key=lambda t: t[0])[1]
    return os.path.join(checkpoints_dir, latest_fname)

def main(args):
    # Setup wandb login
    setup_wandb()
    
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
    run_name = f'{args.exp_name}_bs{args.batch_size}_{start_time}'
    # run_name = f'{os.path.splitext(os.path.basename(args.config))[0]}_{pwd.getpwuid(os.getuid())[0]}_{args.exp_name}_bs{args.batch_size}_{start_time}'
    experiment = run_name if args.resume is None else args.resume
    checkpoints_dir = os.path.join(args.checkpoints, experiment)
    os.makedirs(os.path.join(args.logs, "general_logs", experiment),exist_ok=True)
    sys.stdout = Logger(logpath=os.path.join(args.logs, "general_logs", experiment, f'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(args.logs, "general_logs", experiment, f'log.log'), syspart=sys.stderr)

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

    samples_dir = os.path.join(args.logs, 'samples', experiment)

    set_deterministic(args.seed)
    torch_device = 'cuda:0' if args.device == 'gpu' else 'cpu'
    wandb_logger = loggers.WandbLogger(
        save_dir=args.logs,
        project=args.project,
        name=experiment,
        id=experiment,
        resume='must' if args.resume is not None else 'allow',
        entity=args.wandb_entity,
    )

    drift2 = Drift2(
        hidden_nf=args.nf,
        activation=args.activation,
        n_layers=args.n_layers,
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
        batch_size=args.batch_size,
        lr=args.lr,
        torch_device=torch_device,
        test_epochs=args.test_epochs,
        n_stability_samples=args.n_stability_samples,
        log_iterations=args.log_iterations,
        samples_dir=samples_dir,
        data_augmentation=args.data_augmentation,
        table_name=args.table_name,
        use_affinity_mode=args.use_affinity_mode,
    )
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=experiment + '_{epoch:02d}',
        monitor='loss/val',
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
    )
    trainer = Trainer(
        max_epochs=args.n_epochs,
        logger=wandb_logger,
        callbacks=checkpoint_callback,
        accelerator=args.device,
        devices=1,
        num_sanity_val_steps=0,
        enable_progress_bar=args.enable_progress_bar,
    )

    if args.resume is None:
        last_checkpoint = None
    else:
        last_checkpoint = find_last_checkpoint(checkpoints_dir)
        print(f'Training will be resumed from the latest checkpoint {last_checkpoint}')

    print('Start training')
    trainer.fit(model=drift2, ckpt_path=last_checkpoint)

if __name__ == '__main__':
    # get the name of the folder of the current file
    p = argparse.ArgumentParser(description=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/train_pdbbind.yml')
    p.add_argument('--device', action='store', type=str, default='cpu')
    p.add_argument('--checkpoints', action='store', type=str, default='checkpoints')
    p.add_argument('--logs', action='store', type=str, default='logs')
    p.add_argument('--log_iterations', type=int, default=20)

    p.add_argument('--exp_name', type=str, default='YourName')
    p.add_argument('--n_layers', type=int, default=6, help='number of layers')
    p.add_argument('--nf', type=int, default=128, help='hidden dimension')
    p.add_argument('--activation', type=str, default='silu', help='activation function')
    p.add_argument('--sin_embedding', type=eval, default=False, help='whether using or not the sin embedding')
    p.add_argument('--normalization_factor', type=float, default=1, help='Normalize the sum aggregation of EGNN')
    p.add_argument('--aggregation_method', type=str, default='sum', help='"sum" or "mean"')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--test_epochs', type=int, default=20)
    p.add_argument('--n_stability_samples', type=int, default=500, help='Number of samples to compute the stability')
    p.add_argument('--data_augmentation', type=eval, default=False, help='whether to use data augmentation')
    p.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--wandb_entity', type=str, default='geometric', help='Entity (project) name')
    p.add_argument('--enable_progress_bar', action='store_true', help='Enable progress bar')
    p.add_argument('--n_epochs', type=int, default=200)
    p.add_argument('--table_name', type=str, default='pdbbind_dataset', help='Database table name for dataset')
    p.add_argument('--use_affinity_mode', action='store_true', help='Use affinity mode: positive scores should equal binding affinities')

    disable_rdkit_logging()

    args = p.parse_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list) and key != 'normalize_factors':
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}
    main(args=args)
