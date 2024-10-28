import torch


def save_checkpoint(model, path, validation_id, epoch, optimizer, scheduler):
    """
    save checkpoint (optimizer and model)
    :param validation_id:
    :param model:
    :return:
    """
    path_model = path+'model_'+str(validation_id)+'.pt'

    save_model_state(model, path_model)

    if validation_id:
        torch.save({
                'validation_id': validation_id,
                'epoch': epoch,
                'model_state_dict_path': path_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }, path+'checkpoint.tar')

def save_model_state(model, path):
    print('Saving model: ' + path)
    model = model.module if type(model) is torch.nn.DataParallel else model
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, optimizer, scheduler):
    checkpoint = torch.load(path)
    load_model_state(model, checkpoint['model_state_dict_path'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def load_model_state(model, state_path):
    state = torch.load(state_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state)
