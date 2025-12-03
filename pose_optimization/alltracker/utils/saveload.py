import pathlib
import os
import torch

def save(ckpt_dir, module, optimizer, scheduler, global_step, keep_latest=2, model_name='model'):
    pathlib.Path(ckpt_dir).mkdir(exist_ok=True, parents=True)
    prev_ckpts = list(pathlib.Path(ckpt_dir).glob('%s-*pth' % model_name))
    prev_ckpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
    if len(prev_ckpts) > keep_latest-1:
        for f in prev_ckpts[keep_latest-1:]:
            f.unlink()
    save_path = '%s/%s-%09d.pth' % (ckpt_dir, model_name, global_step)
    save_dict = {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    print(f"saving {save_path}")
    torch.save(save_dict, save_path)
    return False

def load(fabric, ckpt_path, model, optimizer=None, scheduler=None, model_ema=None, step=0, model_name='model', ignore_load=None, strict=True, verbose=True, weights_only=False):
    if verbose:
        print('reading ckpt from %s' % ckpt_path)
    if not os.path.exists(ckpt_path):
        print('...there is no full checkpoint in %s' % ckpt_path)
        print('-- note this function no longer appends "saved_checkpoints/" before the ckpt_path --')
        assert(False)
    else:
        if os.path.isfile(ckpt_path):
            path = ckpt_path
            print('...found checkpoint %s' % (path))
        else:
            prev_ckpts = list(pathlib.Path(ckpt_path).glob('%s-*pth' % model_name))
            prev_ckpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
            if len(prev_ckpts):
                path = prev_ckpts[0]
                # e.g., './checkpoints/2Ai4_5e-4_base18_1539/model-000050000.pth'
                # OR ./whatever.pth
                step = int(str(path).split('-')[-1].split('.')[0])
                if verbose:
                    print('...found checkpoint %s; (parsed step %d from path)' % (path, step))
            else:
                print('...there is no full checkpoint here!')
                return 0
        if fabric is not None:
            checkpoint = fabric.load(path)
        else:
            checkpoint = torch.load(path, weights_only=weights_only)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        assert ignore_load is None # not ready yet
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=strict)
    return step
                        
                                            
                                                                                                                                                                                    
