import time
import os
import torch
import torch.optim as optimizer
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from lib.cnn.predict import predict
from lib.evaluation import evaluate
from lib.cnn.utils import save_model_state, save_checkpoint, load_checkpoint
from lib.metrics import F1Score

def fit(model, train, validation, iterations=(90, 130, 150, 170, 180),  log_modulo=500, val_modulo=5, loss=torch.nn.CrossEntropyLoss(),
        lr=0.1,  gamma=0.1,  momentum=0.9, batch_size=124,  n_workers=8,  validation_size=1000,
        metrics=(F1Score(),), device_id=0, output_path='./'):
    """
    This function performs a model training procedure.
    :param model: the model that should be trained
    :param train: The training set
    :param validation: The validation set
    :param iterations: This is a list of epochs numbers, it indicates when to changes learning rate
    :param log_modulo: Indicates after how many batches the loss is printed
    :param val_modulo: Indicates after how many epochs should be done a validation
    :param lr: The learning rate
    :param gamma: The coefficient to apply when decreasing the learning rate
    :param momentum: The momentum
    :param batch_size: The mini batch size
    :param n_workers: The number of parallel job for input preparation
    :param validation_size: The maximum number of occurrences to use in validation
    :param metrics: The list of evaluation metrics to use in validation
    """

    # training parameters
    max_iterations = iterations[-1]
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batch_size,  num_workers=n_workers)

    print('Training...')

    opt = optimizer.SGD(model.parameters(), lr=lr, momentum=momentum)

    scheduler = MultiStepLR(opt, milestones=list(iterations), gamma=gamma)

    # number of batches in the ml
    epoch_size = len(train_loader)

    # one log per epoch if value is -1
    log_modulo = epoch_size if log_modulo == -1 else log_modulo

    restart = False
    if os.path.isfile(output_path+'checkpoint.tar'):
        restart = True
        checpoint_epoch = load_checkpoint(model, output_path+'checkpoint.tar', opt, scheduler)
    
    if not restart:
        with open(output_path+'loss.txt', 'w') as f:
            f.write('-' * 5 + ' Start ' + '-' * 5 + '\n')
    

    if torch.cuda.is_available():
        # check if GPU is available
        print("Training on GPU")
        model = model.to(torch.device('cuda:0'))
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    else:
        print("Training on CPU")

    for epoch in range(max_iterations):
        if restart and epoch <= checpoint_epoch:
            continue

        model.train()

        # printing new epoch
        print('-' * 5 + ' Epoch ' + str(epoch + 1) + '/' + str(max_iterations) +
              ' (lr: ' + str(scheduler.get_lr()) + ') ' + '-' * 5)
        with open(output_path+'loss.txt', 'a') as f:
            f.write('-' * 5 + ' Epoch ' + str(epoch + 1) + '/' + str(max_iterations) +
              ' (lr: ' + str(scheduler.get_lr()) + ') ' + '-' * 5 + '\n')

        running_loss = 0.0

        for idx, data in enumerate(train_loader):
            # get the inputs
            if len(data) > 2:
                inputs, labels = data[0:len(data)-1], data[-1]
            else:
                inputs, labels = data
            #with open(output_path+'loss.txt', 'a') as f:
            #    f.write(str(labels) + '\n')
            #    f.write(str(torch.isnan(inputs).any()) + '\n')

            # wrap labels in Variable as input is managed through a decorator
            if type(inputs) is tuple or type(inputs) is list:
                inputs = [Variable(input.cuda(device_id)) if torch.cuda.is_available() else Variable(input) for input in inputs]
            else:
                inputs = Variable(inputs.cuda(device_id)) if torch.cuda.is_available() else Variable(inputs)
            labels = Variable(labels.cuda(device_id)) if torch.cuda.is_available() else Variable(labels)

            # zero the parameter gradients
            opt.zero_grad()
            outputs = model(inputs)
            #with open(output_path+'loss.txt', 'a') as f:
            #    f.write(str(outputs) + '\n')
            loss_value = loss(outputs, labels)
            loss_value.backward()

            opt.step()

            # print loss
            running_loss += loss_value.item()
            if idx % log_modulo == log_modulo - 1:  # print every log_modulo mini-batches
                print('[%d, %5d] loss: %.5f' % (epoch + 1, idx + 1, running_loss / log_modulo))
                with open(output_path+'loss.txt', 'a') as f:
                    f.write('[%d, %5d] loss: %.5f \n' % (epoch + 1, idx + 1, running_loss / log_modulo))
                running_loss = 0.0

        # end of epoch update of learning rate scheduler
        scheduler.step(epoch + 1)

        # validation of the model
        if epoch % val_modulo == val_modulo - 1:
            validation_id = str(int((epoch + 1) / val_modulo))

            # save model checkpoint
            save_checkpoint(model, path=output_path, validation_id=str(validation_id), epoch=epoch, optimizer=opt, scheduler=scheduler)

            # predict
            predictions, labels = predict(
                model, validation, batch_size=batch_size*2, validation_size=validation_size, n_workers=0, device_id=device_id
            )

            # evatuate
            res = '\n[validation_id:' + validation_id + ']\n' + evaluate(predictions, labels, metrics)

            print(res)

    save_model_state(model, path=output_path+'model_final.pt')
    os.remove(output_path+'checkpoint.tar')
    # final validation
    print('Final validation: ')

    predictions, labels = predict(model, validation, batch_size=batch_size*2, validation_size=-1, n_workers=0)

    res = evaluate(predictions, labels, metrics, final=True)

    print(res)

    return predictions
