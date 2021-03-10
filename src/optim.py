import torch
import torch.optim


def optimize(nodes, links, width, height,
             xs_init, ys_init, n_iter=1, lr_xy=0,
             lr_scheduler_milestones=None,
             output_iter=None, verbose=True):
    """Globally minimize loss by adjusting affine transforms.

    The loss function is the mean of distance between true relative
    affine transformations between pairs of images, as estimated in the
    stitching procedures, and estimated relative affine transformations,
    as calculated from parameterized affine transformations that are
    optimized over.

    Args:
        nodes (list of int): ids of images in the graph
        links (dict): {(int, int): affine.Affine}
            transforms between pairs of images that are estimated in the
            stitching procedures
        width, height (list of int [len(nodes),]): (width, height) of images
        xs_init, ys_init (list of float [len(nodes),]):
            initial values for x and y translation shifts
            these refer to the centroids
        n_iter (int): number of iterations
        lr_xy (float): param specific learning rates
            for the adam optimizer
        lr_scheduler_milestones (NoneType or list of int): iterations when
            learning rate is decayed by 0.1, disabled if None
        output_iter (list of int): iterations where loss and affines are
            returned as outputs, if None, this defaults to [last iteration]
        verbose (bool)

    Returns:
        tuple (list of int, list of float, list of list of affine.Affine):
            iterations, losses at those iterations,
            transforms estimated at those iterations
    """
    if output_iter is None:
        # output results from last iteration
        output_iter = [n_iter - 1]

    # prepare for init
    rel_true_tensor = []  # collects true relative links
    trans_i_idx = []  # collects integer indices for i in link(i, j)
    trans_j_idx = []  # collects integer indices for j in link(i, j)
    for (i_idx, j_idx), trans in links.items():
        if trans is not None:
            rel_true_tensor.append(torch.from_numpy(trans).float().T)
            trans_i_idx.append(nodes.index(i_idx))
            trans_j_idx.append(nodes.index(j_idx))

    if len(rel_true_tensor) == 0:
        raise ValueError('No links available for optimization.')
    rel_true_tensor = torch.stack(rel_true_tensor)  # [n_links, 2]

    # initialize leaf nodes
    xs = torch.tensor(xs_init, dtype=torch.float, requires_grad=True)
    ys = torch.tensor(ys_init, dtype=torch.float, requires_grad=True)

    # initialize optimizer and scheduler
    optimizer_xy = torch.optim.Adam([xs, ys], lr=lr_xy)
    if lr_scheduler_milestones is not None:
        lr_scheduler_xy = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_xy, lr_scheduler_milestones)

    # prepare to collect output
    if (n_iter - 1) not in output_iter:
        output_iter.append(n_iter - 1)
    output_loss = []
    output_trans = []
    output_params = {}

    # iterate n_iter times
    for k in range(n_iter):
        optimizer_xy.zero_grad()
        # compute absolute translations
        trans = torch.stack([  # [n_nodes, 2]
            xs, ys]).T

        # extract i, j translations to estimate relative translations
        trans_i = trans[trans_i_idx, ...]  # [n_links, 2]
        trans_j = trans[trans_j_idx, ...]  # [n_links, 2]

        # get estimated relative translations
        rel_est_tensor = trans_i - trans_j  # [n_links, 2]

        # compute loss on each link (between true/estimated relative translations)
        # loss is the mean squared distance between points
        # in the true versus estimated relative translation
        losses = ((rel_est_tensor - rel_true_tensor) ** 2
                  ).sum(axis=0)  # -> [1, 2]
        loss = losses.sum()  # -> [1,]

        if verbose:
            if (k + 1) % 200 == 0:
                print('Iter: {}; Loss: {:.3f}'.format(k, loss.item()))
        # output affines and loss
        if k in output_iter:
            output_loss.append(loss.item())
            output_tran = [img_trans.T for img_trans in trans.detach().numpy()]
            output_trans.append(output_tran)
        # back propagate
        loss.backward()
        optimizer_xy.step()
        if lr_scheduler_milestones is not None:
            # learning rate update
            lr_scheduler_xy.step()

    # output transformation params
    output_params['loc'] = list(zip(xs.tolist(), ys.tolist()))

    return output_iter, output_loss, output_trans, output_params
