from flask import Flask, render_template, request, jsonify
import os
import re
import wave
import base64
from google.cloud import speech, storage
from google.cloud.speech_v1 import RecognitionAudio, RecognitionConfig, SpeechClient
from pydub import AudioSegment
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Set the environment variable for Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/tikitaka/static/json/caps-418105-5baf160d389f.json"

def get_wav_sample_rate(file_path):
    with wave.open(file_path, 'rb') as wf:
        return wf.getframerate()

def convert_to_mono(input_path, output_path):
    stereo_audio = AudioSegment.from_wav(input_path)
    mono_audio = stereo_audio.set_channels(1)
    mono_audio.export(output_path, format="wav")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    return f"gs://{bucket_name}/{destination_blob_name}"

def analyze_transcript(transcript):
    habit_words = ['어', '그', '저', '음']
    negative_words = ['근데', '여튼', '암튼', '그래가지고']
    habit_counts = {word: transcript.count(word) for word in habit_words}
    negative_counts = {word: transcript.count(word) for word in negative_words}
    return habit_counts, negative_counts

def process_audio_file(bucket_name, file_path):
    mono_audio_path = "mono_audio.wav"
    convert_to_mono(file_path, mono_audio_path)
    gcs_uri = upload_to_gcs(bucket_name, mono_audio_path, "mono_audio.wav")

    client = SpeechClient()
    audio = RecognitionAudio(uri=gcs_uri)
    config = RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=get_wav_sample_rate(mono_audio_path),
        language_code="ko-KR",
        enable_word_time_offsets=True
    )
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=600)
    transcript = "".join(result.alternatives[0].transcript for result in response.results)
    habit_counts, negative_counts = analyze_transcript(transcript)
    return habit_counts, negative_counts

def run_quickstart(bucket_name, file_paths):
    total_habit_counts = {word: 0 for word in ['어', '그', '저', '음']}
    total_negative_counts = {word: 0 for word in ['근데', '여튼', '암튼', '그래가지고']}
    
    for file_path in file_paths:
        habit_counts, negative_counts = process_audio_file(bucket_name, file_path)
        for word, count in habit_counts.items():
            total_habit_counts[word] += count
        for word, count in negative_counts.items():
            total_negative_counts[word] += count
    
    feedback = ""
    total_habit_usage = sum(total_habit_counts.values())
    total_negative_usage = sum(total_negative_counts.values())

    if total_habit_usage > 0 or total_negative_usage > 0:
        if total_habit_usage > 0:
            feedback += f"총 {total_habit_usage}번의 습관어를 사용하셨습니다. 효과적인 커뮤니케이션을 위해 불필요한 습관어 사용을 줄이는 것이 좋습니다. 면접에서는 명확하고 자신감 있는 표현이 중요합니다. 다음 팁을 참고하세요:\n"
            feedback += "1. 잠시 멈추기: 습관어를 말할 때 잠시 멈추고 생각할 시간을 가지세요.\n"
            feedback += "2. 연습하기: 자주 사용하는 습관어를 인식하고 이를 의식적으로 줄이기 위해 연습해보세요.\n"
            feedback += "3. 대체어 찾기: 습관어 대신 사용할 수 있는 적절한 단어를 찾아보세요.\n"
        if total_negative_usage > 0:
            feedback += f"총 {total_negative_usage}번의 부정어를 사용하셨습니다. 더 나은 커뮤니케이션을 위해 부정어 사용을 줄이는 것이 좋습니다. 명확하고 긍정적인 표현이 중요합니다. 다음 팁을 참고하세요:\n"
            feedback += "1. 부정어 피하기: 부정어 대신 긍정적인 표현을 사용해보세요.\n"
            feedback += "2. 구체적 예시 사용하기: 부정어를 사용할 때 구체적인 예시를 들어 명확하게 설명해보세요.\n"
            feedback += "3. 연습하기: 부정어를 인식하고 이를 의식적으로 줄이기 위해 연습해보세요.\n"
    else:
        feedback += "훌륭합니다! 면접 동안 습관어와 부정어를 전혀 사용하지 않으셨습니다. 이는 명확하고 자신감 있는 커뮤니케이션 능력을 잘 보여줍니다. 다음 팁을 참고하여 더욱 발전해보세요:\n"
        feedback += "1. 계속 연습하기: 현재의 좋은 습관을 유지하고, 계속해서 연습하여 더욱 유창한 표현을 유지하세요.\n"
        feedback += "2. 자신감 유지하기: 지금처럼 자신감 있게 말하는 태도를 계속 유지하세요.\n"
        feedback += "3. 구체적 예시 사용하기: 답변을 할 때 구체적인 예시를 사용하면 더욱 효과적입니다.\n"
        feedback += "지금처럼 명확하고 자신감 있게 면접을 진행하시면 좋은 결과가 있을 것입니다. 훌륭한 성과를 계속 이어나가세요!\n"
    
    return total_habit_counts, total_negative_counts, feedback

results_storage = {}

def detectPupil(eye):
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    _, inverse_eye = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(inverse_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    if contours:
        (x, y, w, h) = cv2.boundingRect(contours[0])
        pupil_center = (x + w // 2, y + h // 2)
        eye_center = eye.shape[1] // 2
        deviation = pupil_center[0] - eye_center
        distance_ratio = abs(deviation) / (eye.shape[1] // 2)
        accuracy = 1 - distance_ratio
        return pupil_center, accuracy
    return None, 0

@app.route('/detect_face', methods=['POST'])
def detect_face():
    face_cascade_path = "C:/tikitaka/static/xml/haarcascade_frontalface_alt.xml"
    eye_cascade_path = "C:/tikitaka/static/xml/haarcascade_eye.xml"
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    data = request.get_json()
    images = data['images']

    accuracies = []
    warning_count = 0

    for image_data in images:
        image_data = base64.b64decode(image_data.split(',')[1])
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                eye = roi_color[ey:ey+eh, ex:ex+ew]
                pupil, accuracy = detectPupil(eye)
                if pupil:
                    accuracies.append(accuracy)
                else:
                    accuracies.append(0.0)
        
        if len(faces) > 1:
            warning_count += 1

    #0.2 미만은 산점도에서 제거
    filtered_accuracies = [acc for acc in accuracies if acc >= 0.2]
    
    with open('accuracies.pkl', 'wb') as f:
        pickle.dump(filtered_accuracies, f)
    
    categories = {'매우 좋음': 0, '좋음': 0, '보통': 0, '미흡': 0}
    for acc in filtered_accuracies:
        if acc >= 0.95:
            categories['매우 좋음'] += 1
        elif acc >= 0.8:
            categories['좋음'] += 1
        elif acc >= 0.6:
            categories['보통'] += 1
        elif acc > 0.2:  # 0.2보다 크고 0.6보다 작은 경우
            categories['미흡'] += 1
    
    plt.figure()  # 새로운 Figure 객체 생성 
    plt.scatter(range(len(filtered_accuracies)), filtered_accuracies, c=filtered_accuracies, cmap='coolwarm')
    plt.xlabel('Frame')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over frames')
    plt.colorbar(label='Accuracy')
    plt.axhline(y=0.95, color='r', linestyle='--', label='very good')
    plt.axhline(y=0.8, color='g', linestyle='--', label='good')
    plt.axhline(y=0.6, color='b', linestyle='--', label='normal')
    plt.ylim(0.05, 1.05)
    plt.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    
    total = len(filtered_accuracies)
    max_category = max(categories, key=categories.get)
    max_percent = (categories[max_category] / total * 100) if total > 0 else 0
    
    if max_category == "매우 좋음":
        max_category_message = "당신의 시선 처리 능력은 정말 탁월합니다! 매 순간을 선명하게 파악하고, 뛰어난 집중력으로 면접 과정을 효과적으로 진행할 것으로 기대됩니다. 이러한 높은 수준의 성과는 당신의 자신감을 높여주고 면접관에게도 긍정적인 인상을 남길 것입니다."
    elif max_category == "좋음":
        max_category_message = "시선 처리 능력이 상당히 좋습니다! 대체로 안정적이고 정확한 시선 추적 능력을 보여주며, 이는 면접 중에도 면접관과의 원활한 대화를 이끌어낼 것으로 예상됩니다. 조금 더 연습하면 더욱 일관된 결과를 얻을 수 있을 것입니다."
    elif max_category == "보통":
        max_category_message = "시선 처리 능력은 보통 수준입니다. 어려운 상황에서 약간의 어려움이 있을 수 있지만, 기본적으로는 잘하고 있습니다. 조금 더 연습하고 신중히 주의를 기울이면, 시선을 효과적으로 관리하는 방법을 개선할 수 있을 것입니다."
    else:
        max_category_message = "시선 처리 능력이 부족하게 나타났습니다. 이는 면접 과정에서 실수로 이어질 수 있으므로 주의가 필요합니다. 추가적인 연습과 함께, 전문가의 조언을 구하거나 시각적인 트레이닝을 받는 것이 도움이 될 수 있습니다."

    result = {
        "img_base64": img_base64,
        "summary": max_category_message,
        "warning_count": warning_count,
        "frame_count": total  # 추가된 프레임 수 정보
    }
    
    # 결과를 저장
    results_storage['latest'] = result
    
    return jsonify(result)


@app.route('/get_face_detection_result', methods=['GET'])
def get_face_detection_result():
    result = results_storage.get('latest', {})
    return jsonify(result)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/test_ready')
def test_ready():
    return render_template('test_ready.html')

@app.route('/test_start')
def test_start():
    return render_template('test_start.html')

@app.route('/test_result')
def test_result():
    return render_template('test_result.html')

@app.route('/upload_wav', methods=['POST'])
def upload_wav():
    if 'wav_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['wav_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        bucket_name = "hallym-caps"
        file_paths = [file_path]
        total_habit_counts, total_negative_counts, feedback = run_quickstart(bucket_name, file_paths)
        
        return jsonify({"habit_counts": total_habit_counts, "negative_counts": total_negative_counts, "feedback": feedback})

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(host='0.0.0.0', port=5000, debug=True)