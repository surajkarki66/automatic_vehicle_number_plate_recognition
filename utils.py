import csv
import tempfile
import cv2
import string
import easyocr
import ffmpeg
import ast
import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
from typing import Any, Dict, Tuple, List, Optional, Union

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int: Dict[str, str] = {
    'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char: Dict[str, str] = {
    '0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}


def write_csv(results: Dict[int, Dict[int, Dict[str, any]]], output_path: str) -> None:
    """
    Write the results to a CSV file.

    Args:
        results (Dict[int, Dict[int, Dict[str, any]]]): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score',
                                                'license_number', 'license_number_score'))

        for frame_nmr, cars in results.items():
            for car_id, car_data in cars.items():
                if 'car' in car_data and 'license_plate' in car_data and 'text' in car_data['license_plate']:
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        car_id,
                        '[{} {} {} {}]'.format(*car_data['car']['bbox']),
                        '[{} {} {} {}]'.format(
                            *car_data['license_plate']['bbox']),
                        car_data['license_plate']['bbox_score'],
                        car_data['license_plate']['text'],
                        car_data['license_plate']['text_score']
                    ))


def license_complies_format(text: str) -> bool:
    """
    Check if the license plate text complies with the required format (UK License Plate).

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    return (
        text[0] in string.ascii_uppercase or text[0] in dict_int_to_char and
        text[1] in string.ascii_uppercase or text[1] in dict_int_to_char and
        text[2] in string.digits or text[2] in dict_char_to_int and
        text[3] in string.digits or text[3] in dict_char_to_int and
        text[4] in string.ascii_uppercase or text[4] in dict_int_to_char and
        text[5] in string.ascii_uppercase or text[5] in dict_int_to_char and
        text[6] in string.ascii_uppercase or text[6] in dict_int_to_char
    )


def format_license(text: str) -> str:
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    return ''.join(mapping.get(i, {}).get(c, c) for i, c in enumerate(text))


def read_license_plate(license_plate_crop) -> Tuple[Optional[str], Optional[float]]:
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (any): Cropped image containing the license plate.

    Returns:
        Tuple[Optional[str], Optional[float]]: Formatted license plate text and confidence score.
    """
    detections = reader.readtext(license_plate_crop)

    for bbox, text, score in detections:
        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate: Tuple[int, int, int, int, float, int],
            vehicle_track_ids: List[Tuple[int, int, int, int, int]]) -> Tuple[int, int, int, int, int]:
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (Tuple[int, int, int, int, float, int]): License plate bounding box.
        vehicle_track_ids (List[Tuple[int, int, int, int, int]]): List of vehicle bounding boxes.

    Returns:
        Tuple[int, int, int, int, int]: Vehicle bounding box and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1


def interpolate_bounding_boxes(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Interpolates missing bounding boxes for cars and license plates in video frames.

    Args:
        data (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains
                                     information about frame number, car ID, bounding boxes,
                                     and optional license plate details.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing interpolated bounding box values
                              and associated metadata as strings.
    """
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array(
        [list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array(
        [list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    for car_id in unique_car_ids:
        frame_numbers_ = [p['frame_nmr'] for p in data if int(
            float(p['car_id'])) == int(float(car_id))]
        print(frame_numbers_, car_id)

        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i - 1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(
                        prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack(
                        (prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack(
                        (prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(
                        interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row: Dict[str, str] = {
                'frame_nmr': str(frame_number),
                'car_id': str(car_id),
                'car_bbox': ' '.join(map(str, car_bboxes_interpolated[i])),
                'license_plate_bbox': ' '.join(map(str, license_plate_bboxes_interpolated[i]))
            }

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = next(
                    (p for p in data if int(p['frame_nmr']) == frame_number and int(
                        float(p['car_id'])) == int(float(car_id))),
                    None
                )
                if original_row:
                    row['license_plate_bbox_score'] = original_row.get(
                        'license_plate_bbox_score', '0')
                    row['license_number'] = original_row.get(
                        'license_number', '0')
                    row['license_number_score'] = original_row.get(
                        'license_number_score', '0')

            interpolated_data.append(row)

    return interpolated_data


def convert_to_mp4(input_file: Union[str, bytes], output_file: str) -> None:
    """
    Converts a video file to MP4 format using H.264 and AAC codecs.

    Args:
        input_file (Union[str, bytes]): Path to the input video file or a byte stream.
        output_file (str): Path to save the converted MP4 file.

    Returns:
        None
    """
    ffmpeg.input(input_file).output(
        output_file,
        vcodec='libx264',  # Video codec (H.264 is widely supported)
        acodec='aac',      # Audio codec (AAC is widely supported)
        strict='-2',       # Allows non-strict behavior (for better compatibility)
        preset='fast',     # Balance between speed and quality
        crf=23,            # Constant Rate Factor for a good quality-to-size ratio
        movflags='faststart'  # Move the moov atom to the start for better streaming compatibility
    ).overwrite_output().run()
    
    

def draw_border(
    img: np.ndarray, 
    top_left: Tuple[int, int], 
    bottom_right: Tuple[int, int], 
    color: Tuple[int, int, int] = (0, 255, 0), 
    thickness: int = 10, 
    line_length_x: int = 200, 
    line_length_y: int = 200
) -> np.ndarray:
    """
    Draws corner borders on an image.

    Parameters:
    - img: np.ndarray -> The image on which to draw the border.
    - top_left: Tuple[int, int] -> Coordinates (x1, y1) of the top-left corner.
    - bottom_right: Tuple[int, int] -> Coordinates (x2, y2) of the bottom-right corner.
    - color: Tuple[int, int, int] -> Border color in BGR format (default is green).
    - thickness: int -> Thickness of the border lines (default is 10).
    - line_length_x: int -> Length of horizontal lines (default is 200).
    - line_length_y: int -> Length of vertical lines (default is 200).

    Returns:
    - np.ndarray: The image with the border drawn.
    """

    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # -- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # -- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def process_video(csv_path: str, video_path: str, output_path: str) -> str:
    """
    Process a video with bounding box data, interpolate the bounding boxes, 
    and save the processed video to the specified output path.

    Parameters:
    - csv_path (str): Path to the CSV file containing bounding box data.
    - video_path (str): Path to the input video file.
    - output_path (str): Path where the processed video will be saved.

    Returns:
    - str: Path to the processed video.
    """
    # Step 1: Read the CSV file containing bounding box data
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    
    # Step 2: Interpolate the bounding box data
    interpolated_data = interpolate_bounding_boxes(data)
    
    # Define the field names for the new CSV file
    fieldnames = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
    
    # Step 3: Save the interpolated data to a temporary CSV file
    temp_csv_path = tempfile.mktemp(suffix=".csv")

    with open(temp_csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(interpolated_data)

    # Step 4: Read the temporary CSV file into a DataFrame
    results = pd.read_csv(temp_csv_path)

    # Step 5: Post-process the video with the results and save it to the specified output path
    new_output_path = post_processing_video(video_path, results, output_path)

    # Step 6: Return the path to the processed video
    return new_output_path

def post_processing_video(
    video_path: str, 
    results: pd.DataFrame, 
    output_path: str
) -> None:
    """
    Processes a video by drawing bounding boxes on detected cars and their license plates,
    and overlays the recognized license plate number.

    Parameters:
    - video_path (str): Path to the input video file.
    - results (pd.DataFrame): DataFrame containing detection results with car IDs, bounding boxes, 
      license plate numbers, confidence scores, and frame numbers.
    - output_path (str): Path to save the processed output video.

    Returns:
    - None: The processed video is saved to output_path.
    """

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    license_plate: Dict[int, Dict[str, Any]] = {}

    for car_id in np.unique(results['car_id']):
        max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        license_plate[car_id] = {
            'license_crop': None,
            'license_plate_number': results[
                (results['car_id'] == car_id) & 
                (results['license_number_score'] == max_)
            ]['license_number'].iloc[0]
        }
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, results[
            (results['car_id'] == car_id) & 
            (results['license_number_score'] == max_)
        ]['frame_nmr'].iloc[0])
        
        ret, frame = cap.read()

        x1, y1, x2, y2 = ast.literal_eval(
            results[
                (results['car_id'] == car_id) & 
                (results['license_number_score'] == max_)
            ]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
        )

        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

        license_plate[car_id]['license_crop'] = license_crop

    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read frames
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                # Draw car
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                    df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
                )
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)

                # Draw license plate
                x1, y1, x2, y2 = ast.literal_eval(
                    df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
                )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                # Crop license plate
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

                H, W, _ = license_crop.shape

                try:
                    frame[int(car_y1) - H - 100:int(car_y1) - 100,
                          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                    frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                    (text_width, text_height), _ = cv2.getTextSize(
                        license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.3,
                        17
                    )

                    cv2.putText(frame,
                                license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4.3,
                                (0, 0, 0),
                                17)

                except:
                    pass

            out.write(frame)
            frame = cv2.resize(frame, (1280, 720))

    out.release()
    cap.release()
    new_output_path = tempfile.mktemp(suffix=".mp4")
    convert_to_mp4(output_path, new_output_path)
    
    return new_output_path