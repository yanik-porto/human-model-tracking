import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import cv2
from copy import deepcopy

from .viz_utils import create_cube_mesh
from .matrix import *
from .scene3d import Scene3D

class Renderer(Scene3D):
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None,  img_res_height=None):
        super().__init__(focal_length=focal_length, img_res=img_res, img_res_height=img_res_height)
        height = img_res if img_res_height is None else img_res_height
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=height,
                                       point_size=1.0)
        self.faces = faces

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i]), (2,0,1))).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, joints=None):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        cam_trans = deepcopy(camera_translation)
        cam_trans[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = cam_trans
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])

        scene.add(camera, pose=camera_pose)


        # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        # light_pose = np.eye(4)

        # light_pose[:3, 3] = np.array([0, -1, 1])
        # scene.add(light, pose=light_pose)

        # light_pose[:3, 3] = np.array([0, 1, 1])
        # scene.add(light, pose=light_pose)

        # light_pose[:3, 3] = np.array([1, 1, 2])
        # scene.add(light, pose=light_pose)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_lighting(scene, camera_node)

        if joints is not None:
            self.add_joints(joints, scene)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
        
        # output_img *= 255
        # output_img = output_img.astype(np.uint8)

        output_img = cv2.normalize(output_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # if joints is not None:
        #     output_img = self.render_joints(joints, camera_translation, output_img)

        return output_img
    
    def add_joints(self, joints, scene):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.2, 0.3, 0.8, 1.0))

        rot = rotation_3d_x(np.radians(180))
        jHomo = np.concatenate((joints, np.ones((joints.shape[0], 1))), axis=1).transpose()
        jRotated = rot @ jHomo
        joints = jRotated.transpose()[:, :3]

        for i, joint in enumerate(joints):
            mesh = create_cube_mesh(joint)
            mesh = pyrender.Mesh.from_trimesh(mesh, material)

            scene.add(mesh, 'joint_' + str(i))

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def Get3Dfrom2D(List2D, K, R, t, d=1.75):
    # List2D : n x 2 array of pixel locations in an image
    # K : Intrinsic matrix for camera
    # R : Rotation matrix describing rotation of camera frame
    #     w.r.t world frame.
    # t : translation vector describing the translation of camera frame
    #     w.r.t world frame
    # [R t] combined is known as the Camera Pose.

    List2D = np.array(List2D)
    List3D = []
    # t.shape = (3,1)

    for p in List2D:
        # Homogeneous pixel coordinate
        p = np.array([p[0], p[1], 1]).T; p.shape = (3,1)
        # print("pixel: \n", p)

        # Transform pixel in Camera coordinate frame
        pc = np.linalg.inv(K) @ p
        # print("pc : \n", pc, pc.shape)

        # Transform pixel in World coordinate frame
        pw = t + (R@pc)
        # print("pw : \n", pw, t.shape, R.shape, pc.shape)

        # Transform camera origin in World coordinate frame
        cam = np.array([0,0,0]).T; cam.shape = (3,1)
        cam_world = t + R @ cam
        # print("cam_world : \n", cam_world)

        # Find a ray from camera to 3d point
        vector = pw - cam_world
        unit_vector = vector / np.linalg.norm(vector)
        # print("unit_vector : \n", unit_vector)
        
        # Point scaled along this ray
        p3D = cam_world + d * unit_vector
        # print("p3D : \n", p3D)
        List3D.append(p3D)

    return List3D