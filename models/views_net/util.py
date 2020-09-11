import torch
import numpy as np
from pytorch3d.structures import join_meshes_as_batch
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes, Textures
from pytorch3d.io import load_obj


def execute_batch(model, criterion, input_val, target_val, device):
    target_val = target_val.to(device=device)
    input_val = input_val.to(device=device)
    nsamp = target_val.size(0)

    # compute output
    output = model(input_val)

    loss = criterion(output, target_val)

    maxk = 3
    # get the sorted top k for each sample
    _, pred = output.topk(maxk, dim=1, sorted=True)
    pred = pred.t()
    correct = pred.eq(target_val.contiguous().view(1, -1).expand_as(pred))

    prec = []
    for k in (1, 3):
        correct_k = correct[:k].view(-1).float().sum(0)
        res = correct_k.mul_(100.0 / nsamp)
        prec.append(res.detach().cpu().numpy())

    predictions = pred[0].detach().cpu().numpy()
    real_target = target_val.detach().cpu().numpy()
    # cleaning variable for out of memory problems
    del pred, input_val, target_val, correct, output
    torch.cuda.empty_cache()
    return loss, prec, predictions, real_target


def render_shape(model, R, T, args, vertices):
    verts, faces_idx, _ = load_obj(args.obj_path)
    faces = faces_idx.verts_idx
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb)
    # center = verts.mean(0)
    # verts = verts - center
    # scale = max(verts.abs().max(0)[0])
    # verts = verts / scale
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )
    if args.lights:
        images = []
        for i in range(model.nviews):
            if args.net_version == 1:
                model.lights.location = model.camera_position[i]
            else:
                model.lights.location = vertices[i]
            imgs = model.renderer(meshes_world=mesh.to(device=args.device).clone(), R=R[None, i], T=T[None, i],
                                  lights=model.lights).cpu().detach().numpy()
            images.extend(imgs)
    else:
        # model.lights.location = model.light_position
        meshes = mesh.extend(args.nviews)
        images = model.renderer(meshes.to(device=args.device), R=R, T=T)  # , lights=model.lights)
        images = images.detach().cpu().numpy()

    return images


def my_collate(batch):
    meshes, targets = zip(*batch)
    meshes = join_meshes_as_batch(meshes, include_textures=True)

    targets = torch.tensor(targets)
    return [meshes, targets]


def plot_camera_scene(cameras, device):
    """
    Plots a set of predicted cameras `cameras` and their corresponding
    ground truth locations `cameras_gt`. The plot is named with
    a string passed inside the `status` argument.
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.clear()
    color = "#812CE5"
    cam_wires_canonical = get_camera_wireframe().to(device=device)[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    plot_radius = ((1 + np.sqrt(5)) / 2) * 3
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([-plot_radius, plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    # plt.show()
    return plt


def get_camera_wireframe(scale: float = 0.3):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * torch.tensor([-2, 1.5, 4])
    b = 0.5 * torch.tensor([2, 1.5, 4])
    c = 0.5 * torch.tensor([-2, -1.5, 4])
    d = 0.5 * torch.tensor([2, -1.5, 4])
    C = torch.zeros(3)
    F = torch.tensor([0, 0, 3])
    camera_points = [a, b, d, c, a, C, b, d, C, c, C, F]
    lines = torch.stack([x.float() for x in camera_points]) * scale
    return lines


def image_grid(images, rows=None, cols=None, fill: bool = True, show_axes: bool = False):
    plt.figure()
    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        ax.imshow(im[..., :3])

        if not show_axes:
            ax.set_axis_off()
    # draw the figure first...
    fig.canvas.draw()
    # plt.show()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
