import matplotlib.pyplot as plt
import numpy as np


def plot_3d_results(tranlation_total, orientation_total, X_4xN_casadi, X_new):
    camera_initial_rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Multiple 3D Frames")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-5, 5])
    ax.set_ylim([-0, 10])
    ax.set_zlim([-5, 5])

    origin = np.array([0, 0, 0])
    global_x = np.array([1, 0, 0])
    global_y = np.array([0, 1, 0])
    global_z = np.array([0, 0, 1])

    # --- Plot the global frame (red, green, blue) at the origin ---
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        global_x[0],
        global_x[1],
        global_x[2],
        length=1,
        color="red",
    )
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        global_y[0],
        global_y[1],
        global_y[2],
        length=1,
        color="green",
    )
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        global_z[0],
        global_z[1],
        global_z[2],
        length=1,
        color="blue",
    )

    # --- Plot the "initial camera" frame (camera_initial_rotation) ---
    #   We'll call this "Camera 1"
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        camera_initial_rotation[0, 0],
        camera_initial_rotation[1, 0],
        camera_initial_rotation[2, 0],
        length=0.5,
        color="red",
    )
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        camera_initial_rotation[0, 1],
        camera_initial_rotation[1, 1],
        camera_initial_rotation[2, 1],
        length=0.5,
        color="green",
    )
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        camera_initial_rotation[0, 2],
        camera_initial_rotation[1, 2],
        camera_initial_rotation[2, 2],
        length=0.5,
        color="blue",
    )

    # Label the initial camera frame near the origin
    ax.text(
        0, 0, 0, "Camera 1", color="black", fontsize=8
    )  # x,y,z position in 3D  # the text

    # --- Now loop over subsequent frames and label them: Camera 2, Camera 3, etc. ---
    for k in range(len(orientation_total)):
        # The position of the camera in world coords
        points = (
            camera_initial_rotation @ orientation_total[k].T @ (-tranlation_total[k])
        )

        # The rotation in world coords
        full_rotation = camera_initial_rotation @ orientation_total[k].T

        # Plot each camera’s local axes
        ax.quiver(
            points[0],
            points[1],
            points[2],
            full_rotation[0, 0],
            full_rotation[1, 0],
            full_rotation[2, 0],
            length=0.5,
            color="red",
        )
        ax.quiver(
            points[0],
            points[1],
            points[2],
            full_rotation[0, 1],
            full_rotation[1, 1],
            full_rotation[2, 1],
            length=0.5,
            color="green",
        )
        ax.quiver(
            points[0],
            points[1],
            points[2],
            full_rotation[0, 2],
            full_rotation[1, 2],
            full_rotation[2, 2],
            length=0.5,
            color="blue",
        )

        # Label each subsequent camera frame:
        camera_label = f"Camera {k + 2}"
        ax.text(
            points[0], points[1], points[2], camera_label, color="black", fontsize=8
        )

    # --- Plot the 3D points (blue spheres) ---
    points_projected_to_world = camera_initial_rotation @ X_4xN_casadi[0:3, :] * 0.5
    points_projected_to_world_augmentation = (
        camera_initial_rotation @ X_new[0:3, :] * 0.5
    )

    ax.scatter(
        points_projected_to_world_augmentation[0, :],
        points_projected_to_world_augmentation[1, :],
        points_projected_to_world_augmentation[2, :],
        color="red",
        marker="o",
        s=5,
    )
    ax.scatter(
        points_projected_to_world[0, :],
        points_projected_to_world[1, :],
        points_projected_to_world[2, :],
        color="green",
        marker="o",
        s=8,
    )
    ax.view_init(elev=90, azim=-90)
    plt.savefig("3d_camera_poses.pdf", bbox_inches="tight")
    plt.show()
