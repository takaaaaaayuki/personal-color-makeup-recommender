from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import os
from collections import Counter

app = Flask(__name__)

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 春夏秋冬のパーソナルカラーの平均値
spring_rgb = np.array([165.19098375, 142.15050382, 123.1936164])
spring_hsv = np.array([12.94740521, 69.81190518, 165.19098375])
spring_lab = np.array([153.28789097, 133.86658512, 141.4636962])

summer_rgb = np.array([142.28434407, 121.08098306, 106.0596081])
summer_hsv = np.array([11.95754234, 66.50773829, 142.28434407])
summer_lab = np.array([131.32119562, 133.88528728, 139.24688143])

autumn_rgb = np.array([147.5982676, 112.64980718, 98.76022682])
autumn_hsv = np.array([8.68605308, 83.13820777, 147.5982676])
autumn_lab = np.array([127.19425256, 139.6711078, 141.10983096])

winter_rgb = np.array([137.75042384, 116.06619312, 104.76075685])
winter_hsv = np.array([9.93889295, 67.30391417, 137.75042384])
winter_lab = np.array([126.57111213, 134.8772909, 137.51677412])

# 既存データセットの統計値
dataset_stats = {
    'summer': {
        'iris': {
            'mean': np.array([8.33458133, 12.36732892, 33.51589202]),
            'median': np.array([6.28571429, 13., 36.55555556]),
            'min': np.array([0., 0., 0.]),
            'max': np.array([51.27777778, 27., 54.625]),
            'mode': np.array([0.0, 0.0, 0.0])
        },
        'sclera': {
            'mean': np.array([121.26469305, 136.20166702, 176.95406745]),
            'median': np.array([121.2212766, 136.24312896, 179.31412894]),
            'min': np.array([64.73537604, 83.70560748, 99.78967495]),
            'max': np.array([186.54442344, 192.34215501, 233.65625]),
            'mode': np.array([125.95481336, 132.08644401, 163.66994106])
        }
    },
    'winter': {
        'iris': {
            'mean': np.array([8.31014989, 12.16148263, 30.33020668]),
            'median': np.array([7.48333333, 13.26144036, 33.23106061]),
            'min': np.array([0., 0., 0.]),
            'max': np.array([26.02777778, 27.11111111, 55.71428571]),
            'mode': np.array([0.0, 0.0, 0.0])
        },
        'sclera': {
            'mean': np.array([123.46480656, 141.00293814, 181.10098007]),
            'median': np.array([122.83821167, 141.7552994, 180.02218288]),
            'min': np.array([71.45957447, 95.21702128, 129.5260771]),
            'max': np.array([181.96875, 199.84504132, 233.76446281]),
            'mode': np.array([80.37642586, 107.74524715, 154.65874525])
        }
    },
    'autumn': {
        'iris': {
            'mean': np.array([9.66486941, 13.28414286, 35.32358168]),
            'median': np.array([8.29166667, 13.86835749, 36.84375]),
            'min': np.array([0., 0., 0.]),
            'max': np.array([38., 27., 63.]),
            'mode': np.array([0.0, 0.0, 0.0])
        },
        'sclera': {
            'mean': np.array([128.63041886, 144.16956902, 185.19845233]),
            'median': np.array([129.74, 146.9441382, 187.65964112]),
            'min': np.array([72.90715884, 80.49492901, 105.68762677]),
            'max': np.array([191.8704, 192.87642586, 231.11790157]),
            'mode': np.array([97.86481481, 119.75185185, 167.92592593])
        }
    },
    'spring': {
        'iris': {
            'mean': np.array([8.17262554, 12.20020359, 35.01788183]),
            'median': np.array([7.32539683, 12.18333333, 37.59411765]),
            'min': np.array([0., 0., 0.]),
            'max': np.array([34., 26., 69.]),
            'mode': np.array([0.0, 0.0, 0.0])
        },
        'sclera': {
            'mean': np.array([119.88681702, 136.911578, 180.27225129]),
            'median': np.array([120.38569059, 138.80199899, 181.93047849]),
            'min': np.array([67.27060653, 81.40207972, 100.13171577]),
            'max': np.array([204.683391, 204.35640138, 237.31314879]),
            'mode': np.array([141.30288462, 139.06891026, 165.96634615])
        }
    }
}

# 人物の写真から背景を切り取り、肌の色を抽出する関数
def extract_skin_color(image_path):
    # 画像の読み込み
    image = cv2.imread(image_path)
    
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 顔認識器の読み込み
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 顔の検出
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # 顔が検出されなかった場合
    if len(faces) == 0:
        print("顔が検出されませんでした。")
        return None, None, None
    
    # 最初に検出された顔の座標を取得
    x, y, w, h = faces[0]
    
    # 顔の部分を切り取り
    face_image = image[y:y+h, x:x+w]
    
    # 肌の色を抽出するために、YCrCb色空間に変換
    ycrcb = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)
    
    # 肌の色の範囲を定義
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    # 肌の色のマスクを作成
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # 肌の色領域を抽出
    skin = cv2.bitwise_and(face_image, face_image, mask=skin_mask)
    
    # 抽出された肌の色の画像を保存（デバッグ用）
    cv2.imwrite("skin_color_extracted.jpg", cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
    
    # 肌の色の画像からRGB、HSV、Lab値を計算
    skin_rgb = np.mean(skin, axis=(0, 1))
    skin_hsv = np.mean(cv2.cvtColor(skin, cv2.COLOR_BGR2HSV), axis=(0, 1))
    skin_lab = np.mean(cv2.cvtColor(skin, cv2.COLOR_BGR2Lab), axis=(0, 1))
    
    return skin_rgb, skin_hsv, skin_lab

# 与えられたRGB、HSV、Lab値に最も近い季節を返す関数
def find_nearest_season_rgb(rgb):
    differences_rgb = {
        'spring': np.abs(spring_rgb - rgb).sum(),
        'summer': np.abs(summer_rgb - rgb).sum(),
        'autumn': np.abs(autumn_rgb - rgb).sum(),
        'winter': np.abs(winter_rgb - rgb).sum()
    }
    nearest_rgb = min(differences_rgb, key=differences_rgb.get)
    return nearest_rgb

def find_nearest_season_hsv(hsv):
    differences_hsv = {
        'spring': np.abs(spring_hsv - hsv).sum(),
        'summer': np.abs(summer_hsv - hsv).sum(),
        'autumn': np.abs(autumn_hsv - hsv).sum(),
        'winter': np.abs(winter_hsv - hsv).sum()
    }
    nearest_hsv = min(differences_hsv, key=differences_hsv.get)
    return nearest_hsv

def find_nearest_season_lab(lab):
    differences_lab = {
        'spring': np.abs(spring_lab - lab).sum(),
        'summer': np.abs(summer_lab - lab).sum(),
        'autumn': np.abs(autumn_lab - lab).sum(),
        'winter': np.abs(winter_lab - lab).sum()
    }
    nearest_lab = min(differences_lab, key=differences_lab.get)
    return nearest_lab

# 新しい画像から瞳と白目の色を抽出する関数
def extract_eye_colors(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    iris_colors = []
    sclera_colors = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]

            eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)

            iris_color = cv2.mean(eye_roi, mask=mask)
            white_mask = cv2.bitwise_not(mask)
            sclera_color = cv2.mean(eye_roi, mask=white_mask)

            iris_colors.append(iris_color[:3])
            sclera_colors.append(sclera_color[:3])

        break

    return iris_colors[0], sclera_colors[0]  # 最初の目の色のみを返す

# 統計値から最も近い季節を見つける関数
def find_closest_season(iris_color, sclera_color):
    min_distance = float('inf')
    closest_season = None

    for season, stats in dataset_stats.items():
        iris_stats = stats['iris']
        sclera_stats = stats['sclera']

        iris_distances = []
        sclera_distances = []

        for stat_type, stat_values in iris_stats.items():
            stat_value = np.array(stat_values)
            distance = np.linalg.norm(iris_color - stat_value)
            iris_distances.append(distance)

        for stat_type, stat_values in sclera_stats.items():
            stat_value = np.array(stat_values)
            distance = np.linalg.norm(sclera_color - stat_value)
            sclera_distances.append(distance)

        total_distance = sum(iris_distances) + sum(sclera_distances)

        if total_distance < min_distance:
            min_distance = total_distance
            closest_season = season

    return closest_season

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        print("No file part in the request")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return redirect(request.url)
    
    if file:
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        
        print(f"File saved at {image_path}")
        
        try:
            # 画像から肌の色を抽出し、RGB、HSV、Lab値を計算
            skin_rgb, skin_hsv, skin_lab = extract_skin_color(image_path)

            # 肌の色から最も近い季節を求める
            nearest_rgb = find_nearest_season_rgb(skin_rgb)
            nearest_hsv = find_nearest_season_hsv(skin_hsv)
            nearest_lab = find_nearest_season_lab(skin_lab)

            # 瞳と白目の色を抽出
            iris_color, sclera_color = extract_eye_colors(image_path)
            closest_season_eye = find_closest_season(iris_color, sclera_color)

            # 各季節の結果をカウント
            results = [nearest_rgb, nearest_hsv, nearest_lab, closest_season_eye]
            season_counts = Counter(results)
            overall_season = season_counts.most_common(1)[0][0]

            # おすすめのメイク道具を選定
            makeup_recommendation = ""
            if overall_season == 'spring':
                makeup_recommendation = "春の季節には、桃色や明るいピンクのチークやリップがおすすめです。"
            elif overall_season == 'summer':
                makeup_recommendation = "夏の季節には、クールなトーンのピンクやベージュのチークやリップがおすすめです。"
            elif overall_season == 'autumn':
                makeup_recommendation = "秋の季節には、温かみのあるオレンジやレンガ色のチークやリップがおすすめです。"
            elif overall_season == 'winter':
                makeup_recommendation = "冬の季節には、クールなピンクやワインレッドのチークやリップがおすすめです。"

            return render_template('result.html', 
                                   skin_rgb=skin_rgb, 
                                   skin_hsv=skin_hsv, 
                                   skin_lab=skin_lab,
                                   nearest_rgb=nearest_rgb,
                                   nearest_hsv=nearest_hsv,
                                   nearest_lab=nearest_lab,
                                   iris_color=iris_color,
                                   sclera_color=sclera_color,
                                   closest_season_eye=closest_season_eye,
                                   overall_season=overall_season,
                                   makeup_recommendation=makeup_recommendation)
        except Exception as e:
            print(f"An error occurred: {e}")
            return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)