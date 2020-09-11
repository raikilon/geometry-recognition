import numpy as np
import torch
import torch.nn.functional
import torch.nn.parallel
import torch.optim
import torch.utils.data


def execute_inference(model, input_val, target_val, args):
    target_val = target_val.to(device=args.device, non_blocking=True)
    input_val = input_val.to(device=args.device, non_blocking=True)

    # number of samples
    nsamp = int(target_val.size(0) / args.nview)

    # compute output
    output = model(input_val)
    # 400,820 (20 views * 20 samples, 41 classes for 20 views) to 8000 41 (20x20x20 everything is flatten,
    # 41 classes)
    output = output.view(-1, args.num_classes + 1)

    _, scores = generate_custom_targets(target_val, output, args)

    output_val = torch.zeros((nsamp, args.num_classes))
    max_score = []
    for n in range(nsamp):
        # best score between views and classes and get the view candidate index
        j_max = int(np.argmax(scores[:, :, n]) / scores.shape[1])
        # get class scores for the view candidate
        output_val[n] = torch.from_numpy(scores[j_max, :, n])
        max_score.append(np.max(scores[j_max, :, n]))

    output_val = output_val.to(device=args.device)
    target = target_val[0:-1:args.nview]

    cor = np.sum(np.equal(np.array(max_score) < args.threshold, target.detach().cpu().numpy()))
    cor_score = cor * (100 / nsamp)

    _, pred = output_val.topk(1, dim=1, sorted=True)
    pred = pred.t()

    predictions = pred[0].detach().cpu().numpy()

    return predictions, max_score, target.detach().cpu().numpy(), cor_score


def execute_batch(model, criterion, input_val, target_val, args):
    """
    Execute a single batch to compute the loss and various statistics

    Parameters
    ----------
    model : RotaitonNet model
    criterion : Pytorch criterion (CrossEntropy for RotationNet)
    input_val : Input value from a batch of a data loader
    target_val : Target value from a batch of a data loader
    args :  Input args from the parser

    Returns
    -------
    Loss of the model, accuracy (top1 and top5), predicted classes and targets (not the same of the input targets)
    """
    target_val = target_val.to(device=args.device)
    input_val = input_val.to(device=args.device)

    # number of samples
    nsamp = int(target_val.size(0) / args.nview)

    # compute output
    output = model(input_val)
    # 400,820 (20 views * 20 samples, 41 classes for 20 views) to 8000 41 (20x20x20 everything is flatten,
    # 41 classes)
    output = output.view(-1, args.num_classes + 1)

    target, scores = generate_custom_targets(target_val, output, args)

    loss = criterion(output, target)

    output_val = torch.zeros((nsamp, args.num_classes))
    # args.vcand.shape[0], args.num_classes, nsamp
    # for each sample #n, determine the best pose that maximizes the score (for the top class)
    for n in range(nsamp):
        if args.pooling == "max":
            # best score between views and classes and get the view candidate index
            j_max = int(np.argmax(scores[:, :, n]) / scores.shape[1])
            # get class scores for the view candidate
            output_val[n] = torch.from_numpy(scores[j_max, :, n])
            # print(np.max(scores[j_max, :, n]))
        else:
            # average score instead of max
            output_val[n] = torch.from_numpy(np.mean(scores[:, :, n], axis=0))

    output_val = output_val.to(device=args.device)

    # 20 (for 20 samples)
    target = target_val[0:-1:args.nview]

    maxk = 5
    # get the sorted top k for each sample
    _, pred = output_val.topk(maxk, dim=1, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    prec = []
    for k in (1, 5):
        correct_k = correct[:k].view(-1).float().sum(0)
        res = correct_k.mul_(100.0 / nsamp)
        prec.append(res.detach().cpu().numpy())

    predictions = pred[0].detach().cpu().numpy()
    real_target = target.detach().cpu().numpy()
    # cleaning variable for out of memory problems
    del pred, target, input_val, scores, target_val, correct, output_val, output
    torch.cuda.empty_cache()
    return loss, prec, predictions, real_target


def execute_batch_aligned(model, criterion, input_val, target_val, args):
    """

    """
    target_val = target_val.to(device=args.device)
    input_val = input_val.to(device=args.device)

    # number of samples
    nsamp = int(target_val.size(0) / args.nview)

    # compute output
    output = model(input_val)
    # 400,820 (20 views * 20 samples, 41 classes for 20 views) to 8000 41 (20x20x20 everything is flatten,
    # 41 classes)
    output = output.view(-1, args.num_classes + 1)

    _, scores = generate_custom_targets(target_val[:, 0], output, args)

    # 8000 -> 400 x 20 outputs for each views
    target_ = torch.LongTensor(target_val.size(0) * args.nview)

    # initialize target labels with "incorrect view label"
    for j in range(target_.size(0)):
        target_[j] = args.num_classes
    for n in range(target_val.size(0)):
        idx, view = target_val[n]
        target_[n * args.nview + view - 1] = idx

    loss = criterion(output, target_.to(device=args.device))

    output_val = torch.zeros((nsamp, args.num_classes))
    # args.vcand.shape[0], args.num_classes, nsamp
    # for each sample #n, determine the best pose that maximizes the score (for the top class)
    for n in range(nsamp):
        # best score between views and classes and get the view candidate index
        j_max = int(np.argmax(scores[:, :, n]) / scores.shape[1])
        # get class scores for the view candidate
        output_val[n] = torch.from_numpy(scores[j_max, :, n])

    output_val = output_val.to(device=args.device)

    # 20 (for 20 samples)
    target = target_val[:, 0][0:-1:args.nview]

    maxk = 5
    # get the sorted top k for each sample
    _, pred = output_val.topk(maxk, dim=1, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    prec = []
    for k in (1, 5):
        correct_k = correct[:k].view(-1).float().sum(0)
        res = correct_k.mul_(100.0 / nsamp)
        prec.append(res.detach().cpu().numpy())

    predictions = pred[0].detach().cpu().numpy()
    real_target = target.detach().cpu().numpy()
    # cleaning variable for out of memory problems
    del pred, target, input_val, scores, target_val, correct, output_val, output
    torch.cuda.empty_cache()
    return loss, prec, predictions, real_target


def generate_custom_targets(target_val, output, args):
    """
    Generate custom targets for RotationNet. This target take into consideration the view candidate which gives the
    best performance.

    Parameters
    ----------
    target_val : Target value from a batch of a data loader
    output : Output of the model
    args : Input args from the parser

    Returns
    -------
    Correct targets and scores of equation 5 (see RotationNet paper)
    """

    # number of samples
    nsamp = int(target_val.size(0) / args.nview)

    # 8000 -> 400 x 20 outputs for each views
    target_ = torch.LongTensor(target_val.size(0) * args.nview)

    # compute scores and decide target labels
    output_ = torch.nn.functional.log_softmax(output, dim=1)

    # divide object scores by the scores for "incorrect view label" (see Eq.(5))
    output_ = output_[:, :-1] - torch.t(output_[:, -1].repeat(1, output_.size(1) - 1).view(output_.size(1) - 1, -1))
    # reshape output matrix
    output_ = output_.view(-1, args.nview * args.nview, args.num_classes)
    # TODO check if necessary
    output_ = output_.data.cpu().numpy()
    # 400,40,20 (views*views -> one object, classes and samples)
    output_ = output_.transpose(1, 2, 0)
    # initialize target labels with "incorrect view label"
    for j in range(target_.size(0)):
        target_[j] = args.num_classes
    # compute scores for all the candidate poses (see Eq.(5))
    # sum of all histograms differences (we substracted the wrong view to all bins) for each view rotation
    # candidate
    scores = np.zeros((args.vcand.shape[0], args.num_classes, nsamp))
    for j in range(args.vcand.shape[0]):
        for k in range(args.vcand.shape[1]):
            scores[j] = scores[j] + output_[args.vcand[j][k] * args.nview + k]
    # for each sample #n, determine the best pose that maximizes the score for the target class (see Eq.(2))
    for n in range(nsamp):
        # take views candidate with higher score (in training we know the target class)
        j_max = np.argmax(scores[:, target_val[n * args.nview], n])
        # assign target labels
        for k in range(args.vcand.shape[1]):
            # we assign to the right view as a target the right class. In the others the target is the wrong
            # view class
            target_[n * args.nview * args.nview + args.vcand[j_max][k] * args.nview + k] = target_val[n * args.nview]

    target = target_.to(device=args.device)
    return target, scores
