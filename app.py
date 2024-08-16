from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def calculate_gsd(sensor_width, altitude, focal_length, image_width):
    return (sensor_width * altitude) / (focal_length * image_width)

def calculate_area(gsd, points):
    real_world_points = [(x * gsd, y * gsd) for x, y in points]
    n = len(real_world_points)
    area = 0.5 * abs(sum(real_world_points[i][0] * real_world_points[(i + 1) % n][1] - real_world_points[i][1] * real_world_points[(i + 1) % n][0] for i in range(n)))
    return area

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        sensor_width = float(request.form['sensor_width'])
        altitude = float(request.form['altitude'])
        focal_length = float(request.form['focal_length'])

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        global points, resized_image
        image = cv2.imread(image_path)
        
        # Resize image to fit the screen
        screen_res = 1280, 720
        scale_width = screen_res[0] / image.shape[1]
        scale_height = screen_res[1] / image.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(image.shape[1] * scale)
        window_height = int(image.shape[0] * scale)
        resized_image = cv2.resize(image, (window_width, window_height))
        
        img_height_pixels, img_width_pixels = image.shape[:2]
        gsd = calculate_gsd(sensor_width, altitude, focal_length, img_width_pixels)
        
        points = []
        cv2.imshow("Select Area to Measure", resized_image)
        cv2.setMouseCallback("Select Area to Measure", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(points) < 3:
            return "At least 3 points are required to form a polygon"

        points = [(int(x / scale), int(y / scale)) for x, y in points]
        area = calculate_area(gsd, points)
        
        square_meters_to_acres = 1 / 4046.86
        area_acres = area * square_meters_to_acres
        
        mask = np.zeros_like(image)
        points_np = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points_np], (255, 255, 255))
        masked_image = cv2.bitwise_and(image, mask)

        x, y, w, h = cv2.boundingRect(points_np)
        cropped_image = masked_image[y:y+h, x:x+w]
        
        cropped_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "cropped_image.jpg")
        cv2.imwrite(cropped_image_path, cropped_image)
        
        return render_template('index.html', gsd=gsd, area=area, area_acres=area_acres, image_file=image_file.filename)
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def click_event(event, x, y, flags, params):
    global points, resized_image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(resized_image, (x, y), 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(resized_image, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow("Select Area to Measure", resized_image)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
