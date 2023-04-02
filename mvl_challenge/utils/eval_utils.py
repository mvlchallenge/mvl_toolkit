from mvl_challenge.utils.spherical_utils import phi_coords2xyz
from shapely.geometry import Polygon
import torch.nn.functional as F


def compute_L1_loss(y_est, y_ref):
        return F.l1_loss(y_est, y_ref)     
    
def compute_weighted_L1(y_est, y_ref, std, min_std=1E-2):
    return F.l1_loss(y_est/(std + min_std)**2, y_ref/(std + min_std)**2) 


def eval_2d3d_iuo_from_tensors(est_bon, gt_bon, losses, ch=1):
    est_bearing_ceiling = phi_coords2xyz(est_bon[:, 0, :].squeeze())
    est_bearing_floor = phi_coords2xyz(est_bon[:, 1, :].squeeze())
    gt_bearing_ceiling = phi_coords2xyz(gt_bon[:, 0, :].squeeze())
    gt_bearing_floor = phi_coords2xyz(gt_bon[:, 1, :].squeeze())

    iou2d, iou3d = get_2d3d_iou(ch, est_bearing_floor, gt_bearing_floor, est_bearing_ceiling, gt_bearing_ceiling)
    losses["2DIoU"].append(iou2d)
    losses["3DIoU"].append(iou3d)


def eval_2d3d_iuo(phi_coords_est, phi_coords_gt_bon, ch=1):
    est_bearing_ceiling = phi_coords2xyz(phi_coords_est[0])
    est_bearing_floor = phi_coords2xyz(phi_coords_est[1])
    gt_bearing_ceiling = phi_coords2xyz(phi_coords_gt_bon[0])
    gt_bearing_floor = phi_coords2xyz(phi_coords_gt_bon[1])
    return get_2d3d_iou(ch, est_bearing_floor, gt_bearing_floor, est_bearing_ceiling, gt_bearing_ceiling)
    # Project bearings into a xz plane, ch: camera height

def get_2d3d_iou(ch, est_bearing_floor, gt_bearing_floor, est_bearing_ceiling, gt_bearing_ceiling):
    est_scale_floor = ch / est_bearing_floor[1, :]
    est_pcl_floor = est_scale_floor * est_bearing_floor

    gt_scale_floor = ch / gt_bearing_floor[1, :]
    gt_pcl_floor = gt_scale_floor * gt_bearing_floor

    # Calculate height
    est_scale_ceiling = est_pcl_floor[2] / est_bearing_ceiling[2]
    est_pcl_ceiling = est_scale_ceiling * est_bearing_ceiling
    est_h = abs(est_pcl_ceiling[1, :].mean() - ch)

    gt_scale_ceiling = gt_pcl_floor[2] / gt_bearing_ceiling[2]
    gt_pcl_ceiling = gt_scale_ceiling * gt_bearing_ceiling
    gt_h = abs(gt_pcl_ceiling[1, :].mean() - ch)
    try:
        est_poly = Polygon(zip(est_pcl_floor[0], est_pcl_floor[2]))
        gt_poly = Polygon(zip(gt_pcl_floor[0], gt_pcl_floor[2]))

        if not gt_poly.is_valid:
            print("[ERROR] Skip ground truth invalid")
            return -1, -1

        # 2D IoU
        try:
            area_dt = est_poly.area
            area_gt = gt_poly.area
            area_inter = est_poly.intersection(gt_poly).area
            iou2d = area_inter / (area_gt + area_dt - area_inter)
        except:
            iou2d = 0

        # 3D IoU
        try:
            area3d_inter = area_inter * min(est_h, gt_h)
            area3d_pred = area_dt * est_h
            area3d_gt = area_gt * gt_h
            iou3d = area3d_inter / (area3d_pred + area3d_gt - area3d_inter)
        except:
            iou3d = 0
    except:
        iou2d = 0
        iou3d = 0

    return iou2d, iou3d
