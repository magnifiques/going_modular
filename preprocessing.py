import os
import cv2
import hashlib
import imutils

PROJECT_DIR = 'dataset'
LABELS = os.listdir(os.path.join(PROJECT_DIR, 'Training'))
IMG_SIZE = 256

def compute_hash(file):
    hasher = hashlib.md5()
    with open(file, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def crop_img(img):
    """
    Detects the extreme points on the image and crops the rectangular region around them.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply thresholding and remove noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours and select the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    # Get the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    cropped_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    
    return cropped_img

def process_images(data_type, hash_dict):
    data_path = os.path.join(PROJECT_DIR, data_type)
    for label in LABELS:
        folder_path = os.path.join(data_path, label)
        save_path = os.path.join('cleaned', data_type, label)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".jpg"):
                    file_path = os.path.join(root, file)
                    image = cv2.imread(file_path)
                    new_img = crop_img(image)
                    resized_img = cv2.resize(new_img, (IMG_SIZE, IMG_SIZE))
                    
                    temp_save_path = os.path.join(save_path, file)
                    cv2.imwrite(temp_save_path, resized_img)
                    
                    # Compute hash and check for duplicates
                    file_hash = compute_hash(temp_save_path)
                    if file_hash in hash_dict:
                        print(f"Removing duplicate (hash : {file_hash}) : {temp_save_path}")
                        os.remove(temp_save_path)
                    else:
                        hash_dict[file_hash] = [temp_save_path]

if __name__ == '__main__':
    hash_dict = {}
    
    for data_type in ['Training', 'Testing']:
        process_images(data_type, hash_dict)
