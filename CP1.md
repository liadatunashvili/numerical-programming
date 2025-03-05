


## Introduction

In this paper I will be describing 
## 1. Algorithm

### Key Components
#### 1. Edge detection and Processing (Mostly based on red label)

To solve our problem and detect the coca-cola bottles we base our edge detection on mostly red label of coke. For edge detection I used Sobel filter as well as Gaussian blur (To fill out noise and hide the label)
For Gaussian blur I created 2 methods:

Generating the kernel, basically a blur matrix
```python
def generate_gaussian_kernel(size, sigma):  
    kernel = np.zeros((size, size))  
    center = size // 2  
    for x in range(size):  
        for y in range(size):  
            x_dist = x - center  
            y_dist = y - center  
            kernel[x, y] = np.exp(-(x_dist ** 2 + y_dist ** 2) / (2 * sigma ** 2))  
    return kernel / np.sum(kernel)
```
And then applying such blur matrix to the image, where I use convolution for faster operation (by going through array normally I had a lot of performance issues)
```python
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
```

Next part is the edge detection (for us we focus on detecting the red for simplicity)
Here by converting image to HVS color space and then creating a mask (that takes into consideration different range of red) we get an easier picture to detect the edges from, because now we just have red masked.

We apply blur for such mask and then get better image to detect the edges of the red labels.

after that we use Sobel operators for the edge detection (here we also use the convolution operation for optimization).  and lastly we return such points 

```python
def custom_red_specific_edge_detection(frame):  
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    lower_red1 = np.array([0, 120, 120], dtype=np.uint8)  
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)  
    lower_red2 = np.array([170, 120, 120], dtype=np.uint8)  
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)  
  
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  
    red_mask = cv2.bitwise_or(mask1, mask2)  
  
    kernel = np.ones((5, 5), np.uint8)  
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)  
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)  
  
    red_channel = frame[:, :, 2]  
    red_masked = cv2.bitwise_and(red_channel, red_channel, mask=red_mask)  
    red_blurred = apply_gaussian_blur(red_masked, kernel_size=7, sigma=1.0)  
  
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)  
  
    grad_x = convolve2d(red_blurred, sobel_x, mode='same', boundary='symm')  
    grad_y = convolve2d(red_blurred, sobel_y, mode='same', boundary='symm')  
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)  
    edges = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  
  
    _, sharp_edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)  
    kernel = np.ones((3, 3), np.uint8)  
    sharp_edges = cv2.morphologyEx(sharp_edges, cv2.MORPH_CLOSE, kernel)  
    sharp_edges = cv2.morphologyEx(sharp_edges, cv2.MORPH_OPEN, kernel)  
  
    return sharp_edges
```

#### 2. Clustering using KD-tree
To continue with the Clustering here I had to implement the KD-tree for better and more optimized clustering of the objects in order for program to understand the separate objects:

Here's the small explanation of the implementation:

this method finds all points within the specified radius of the center point and returns such indices
```python
def query_ball_point(self, center_point, radius)
```

with this function we optimize the search for such points and branch out quicker
```python
def _query_ball_point_recursive(self, node, center_point, radius, depth, result_indices)
```

Finds  k nearest neighbours for given points and returns distances and indices , also handles multiple queries speeding up out algorithm more.
```python
def query(self, points, k=1)
```

This just implementation for reccursive search
```python
def _query_recursive(self, node, point, k, heap, depth)
```

Now let's return to our original algorithm and continue from there

Here we pass the edges we got from previous edge detection algorithm and also define the distance (it's like a radius for neighbouring points). here we run normal bfs algorithm thet quickly helps us identify and trverse through all neighbouting points . In the end it returns us the clusters
```python
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
```

Labeling clusters, is just small method for making it more understandable and eye pleasing when running the program. It just labels such clusters as objn

```python
def label_clusters_on_frame(frame, clusters):  
  
    output_frame = frame.copy()  
    for i, cluster in enumerate(clusters):  
        cluster = np.array(cluster)  
        x_min, y_min = cluster[:, 1].min(), cluster[:, 0].min()  
        x_max, y_max = cluster[:, 1].max(), cluster[:, 0].max()  
  
        # Draw bounding box and label  
        cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  
        label = f"obj{i + 1}"  
        cv2.putText(output_frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  
    return output_frame
```

#### 3. Speed

Now for the good part, detecting the speed , here we take into consideration the fps, and the distance that object traveled in these 

```python
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
  
    # Use KDTree  
    tree = CustomKDTree(prev_pos_array)  
    distances, indices = tree.query(curr_pos_array)  
  
    for dist in distances.flatten():  # Flatten because query returns 2D array  
        real_distance = dist * scaling_factor  
        speed = real_distance * fps  
        speeds.append(speed)  
  
    return speeds
```


#### 4. Video Processing

So this is the part where everything is combined, for taking up small resources we decrease the size of the video as well as set the scaling factor (translating meters to pixels). For decreasing the fps we skip the frames. 
```python
def process_video_with_average_speed(video_path, target_fps=12, scaling_factor=0.01):  
    cap = cv2.VideoCapture(video_path)  
    target_width = 640  
    fps = cap.get(cv2.CAP_PROP_FPS)  
    frame_skip = max(1, int(fps / target_fps))  
    previous_positions = []  
    frame_count = 0  
    object_avg_speeds = []  
```

Here the logic is following, for each frame we iterate we get red_edges and find the clusters and then fetch the current positions from such edges. after that we calculate the speed and lastly we put this information over the label and display it.

```python
while True:  
        ret, frame = cap.read()  
        if not ret:  
            break  
  
        if frame_count % frame_skip != 0:  
            frame_count += 1  
            continue  
  
        height, width = frame.shape[:2]  
        scale = target_width / width  
        frame_resized = cv2.resize(frame, (target_width, int(height * scale)))  
  
        red_edges = custom_red_specific_edge_detection(frame_resized)  
  
        clusters = find_clusters_kdtree(red_edges)  
  
        current_positions = [  
            (int(np.mean([p[1] for p in cluster])), int(np.mean([p[0] for p in cluster])))  
            for cluster in clusters  
        ]  
  
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
            label = f"Avg Speed: {avg_speed:.2f} m/s"            cv2.putText(labeled_frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  
  
        cv2.imshow('Average Speed Calculation', labeled_frame)  
  
        previous_positions = current_positions  
  
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  
  
        frame_count += 1  
  
    cap.release()  
    cv2.destroyAllWindows()
```

## 2. Results

Now let's look at the results, for video that it worked and for one that it did not work and the reasons why so.

Working video 
Here the speeds are displayed, maybe we can do more tweaking but for now this is correct. Thanks to the bright label, it is easier for us to identify the objects
![Pasted image 20241119104254.png](Pasted%20image%2020241119104254.png)

Now let's talk about the video that in fact may not work. Reasons for such video could be,
- damaged video 
- very low quality video
- discolored video
- or the bottles with no red labels
- or a video where these bottles intersect making it harder for our clustering algorithm to identify an object

Here's the example video where it does not work properly detecting extra objects

![Pasted image 20241119110417.png](Pasted%20image%2020241119110417.png)
![Pasted image 20241119110519.png](Pasted%20image%2020241119110519.png)

and also there's a moment where it is considered as 1 obect instead of 2 separate ones
![Pasted image 20241119110615.png](Pasted%20image%2020241119110615.png)

cp1_vid - working video
cp2_vid - not working properly
### 3. Real World application

So in real word we could use this technology to determine plastic pollution of let's say the KIU pond. As we know most of the plastic waste come from such bottles which have bright labels on it, so maybe small adjustment to the code would help to identify such other plastic waste. 

Determining the speed could help us identify if it is being thrown, to differ items that are put and than taken away again

Maybe we could also use this technology to determine the locations which are being polluted mostly, by people throwing their trash on the ground and then put the trashcans according to the data.

The limitation to this is that the detection of plastics happens using bright labels, we could miss out on the waste that does not have labels on it, or vice-versa identify other objects that are bright as a trash


