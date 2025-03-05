import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.spatial import KDTree
from kd_tree import CustomKDTree

def generate_gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2
    for x in range(size):
        for y in range(size):
            x_dist = x - center
            y_dist = y - center
            kernel[x, y] = np.exp(-(x_dist ** 2 + y_dist ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def apply_gaussian_blur(image, kernel_size=10, sigma=0.5):
    kernel = generate_gaussian_kernel(kernel_size, sigma)
    if len(image.shape) == 3:
        blurred = np.zeros_like(image)
        for channel in range(image.shape[2]):
            blurred[:, :, channel] = convolve2d(
                image[:, :, channel], kernel, mode='same', boundary='symm'
            )
    else:
        blurred = convolve2d(image, kernel, mode='same', boundary='symm')
    return blurred.astype(np.uint8)

def custom_red_specific_edge_detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 120], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 120, 120], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

#hsv stylshi gadaviyvane rgb da miaxlovebiti witlebi rom gavfiltro shemovitana sxvadasva  witlebi


    #mask filterit matricit gadavuyvebi maticas (videos ) aris marica romelic sheicavs 11 ebs da 0 ebs da 1 ebi weria iq sadacdainaxacs
    # witel fers
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)


   # noise is mosashoreblad
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel) # es gaps and holes avsebs
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel) # es ashorebs small noise magalitad zedmeti witeli wertil



    red_channel = frame[:, :, 2]  # it is like greyscale matrix for red intensity value
    red_masked = cv2.bitwise_and(red_channel, red_channel, mask=red_mask) # amoigebs adgilebs sadac maskas da red channel matricebs ertmanets
    ## rom gadaadebs 1 ebis sadac iqneba iset matricas dagvibrunebs
    red_blurred = apply_gaussian_blur(red_masked, kernel_size=7, sigma=1.0) # blurrrrrr again for better edge detection



    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) # changes in horizontal direction
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) # detects changes along the vertical direction
    # gvexmareba edgebis amocnobistvis sobel matrixxxx


    grad_x = convolve2d(red_blurred, sobel_x, mode='same', boundary='symm')
    grad_y = convolve2d(red_blurred, sobel_y, mode='same', boundary='symm')
    # Applies the Sobel filters to the blurred image (red_blurred) to compute gradients.

    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2) # computes strength of edges at each pixel
    # grey sclaes aketebs
    edges = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # gradacias  shewyobads xxdis mere vizualizaciistvis

    _, sharp_edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY) # Isolates stronger edges, discarding weaker ones.
    kernel = np.ones((3, 3), np.uint8)
    sharp_edges = cv2.morphologyEx(sharp_edges, cv2.MORPH_CLOSE, kernel)
    sharp_edges = cv2.morphologyEx(sharp_edges, cv2.MORPH_OPEN, kernel)
    # noise reduction again

    return sharp_edges
"""
 KDTREEEEEEE
"""




def find_clusters_kdtree(edge_map, distance_threshold=10):
    points = np.column_stack(np.where(edge_map > 0))

    kdtree = CustomKDTree(points)

    visited = set()
    clusters = []

    def bfs(start_point_idx):
        cluster = []
        queue = [start_point_idx]
        visited.add(start_point_idx)

        while queue:
            idx = queue.pop(0)
            cluster.append(points[idx])

            neighbors = kdtree.query_ball_point(points[idx], distance_threshold)
            for neighbor_idx in neighbors:
                if neighbor_idx not in visited:
                    queue.append(neighbor_idx)
                    visited.add(neighbor_idx)

        return cluster

    for idx in range(len(points)):
        if idx not in visited:
            clusters.append(bfs(idx))

    return clusters
# bfs it edzebs
# mezoblebs edzebs da ajgupebs yovel 10 pixelshi rac aris da 1 obqtad agiqvams
# This function clusters edge points into groups using a KD-tree for efficient neighbor searches.


def label_clusters_on_frame(frame, clusters):

    output_frame = frame.copy()
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        x_min, y_min = cluster[:, 1].min(), cluster[:, 0].min()
        x_max, y_max = cluster[:, 1].max(), cluster[:, 0].max()

        cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"obj{i + 1}"
        cv2.putText(output_frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return output_frame

def calculate_speed(previous_positions, current_positions, fps, scaling_factor):

    speeds = []
    if not previous_positions or not current_positions:
        return speeds

    prev_pos_array = np.array(previous_positions)
    curr_pos_array = np.array(current_positions)

    if prev_pos_array.size == 0 or curr_pos_array.size == 0:
        return speeds

    if len(prev_pos_array.shape) == 1:
        prev_pos_array = prev_pos_array.reshape(1, -1)
    if len(curr_pos_array.shape) == 1:
        curr_pos_array = curr_pos_array.reshape(1, -1)

    tree = CustomKDTree(prev_pos_array)
    distances, indices = tree.query(curr_pos_array)

    for dist in distances.flatten():
        real_distance = dist * scaling_factor
        speed = real_distance * fps
        speeds.append(speed)

    return speeds
# calculates speed / frame per second
# wamshi frames sheccvla rac ufro magalia mit ufro smooth aris modzraoba
# wamshi rac ufro coda frames vdeb mit ufro didi dashoreba maqvs objectebs shoris
# returns the listoof speeds for rach object

def process_video_with_average_speed(video_path, target_fps=12, scaling_factor=0.01):
    cap = cv2.VideoCapture(video_path)
    target_width = 640
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(fps / target_fps))
    previous_positions = []
    frame_count = 0
    object_avg_speeds = []

# gadavuyevit videos
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1 # jedavda da frame shevamciree
            continue

        height, width = frame.shape[:2]
        scale = target_width / width
        frame_resized = cv2.resize(frame, (target_width, int(height * scale)))
          # cota cota optimizacia , jedavda
        red_edges = custom_red_specific_edge_detection(frame_resized)

        clusters = find_clusters_kdtree(red_edges)
          # vigeb yovel framze red edgebs da vshvebi clusterings
        current_positions = [
            (int(np.mean([p[1] for p in cluster])), int(np.mean([p[0] for p in cluster])))
            for cluster in clusters
        ]
        # gavige clusterebis current

        speeds = calculate_speed(previous_positions, current_positions, target_fps, scaling_factor)

        if len(speeds) > len(object_avg_speeds):
            object_avg_speeds.extend(speeds[len(object_avg_speeds):])
        for i, speed in enumerate(speeds):
            object_avg_speeds[i] = ((frame_count // frame_skip - 1) * object_avg_speeds[i] + speed) / (
                frame_count // frame_skip
            )

        labeled_frame = frame_resized.copy()
        for i, cluster in enumerate(clusters):
            cluster = np.array(cluster)
            x_min, y_min = cluster[:, 1].min(), cluster[:, 0].min()
            x_max, y_max = cluster[:, 1].max(), cluster[:, 0].max()

            avg_speed = object_avg_speeds[i] if i < len(object_avg_speeds) else 0
            cv2.rectangle(labeled_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"Avg Speed: {avg_speed:.2f} m/s" # abavs avg speed visualizacias videoze
            cv2.putText(labeled_frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('Average Speed Calculation', labeled_frame)

        previous_positions = current_positions

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #working vid
    #input_video = "cp1_vid.mp4"
    input_video = "bouncing_stars_advanced.mp4"
    #not properly working vid
    #input_video = "cp1_vid2.mp4"

    # process_video_with_grid_clustering(input_video)
    # process_video_with_kdtree_clustering(input_video)
    process_video_with_average_speed(input_video,scaling_factor=0.003)