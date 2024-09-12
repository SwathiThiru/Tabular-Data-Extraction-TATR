from collections import OrderedDict, defaultdict
import json
import argparse
import sys
import xml.etree.ElementTree as ET
import os
import yaml
from os import listdir
from os.path import join, split
import random
from scipy import stats
import csv
from io import StringIO

import openpyxl
from openpyxl import Workbook
import xlsxwriter
import math
import time
#import easyocr

import torch
import fitz
import pytesseract
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
from fitz import Rect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from pdf2image import convert_from_path
from finalstep import Datasheet

from main import get_model
import postprocess
#sys.path.append('../detr/models')
sys.path.append('../')
from detr.models import build_model

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_class_map(data_type):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6,
            'table row header': 7,
            'table projected column header': 8,
            'table name': 9
        }
    elif data_type == 'detection':
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map

detection_class_thresholds = {
    "table": 0.8,
    "table rotated": 0.7,
    "no object": 10
}

structure_class_thresholds = {
    "table": 0.5,
    "table column": 0.5,
    "table row": 0.5,
    "table column header": 0.3,
    "table projected row header": 0.5,
    "table spanning cell": 0.3,
    "no object": 10,
    "table row header": 0.3,
    "table projected column header": 0.5,
    "table name": 0.9
}

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir',
                        help="Directory for input images")
    parser.add_argument('--words_dir',
                        help="Directory for input words")
    parser.add_argument('--out_dir',
                        help="Output directory")
    parser.add_argument('--model_dir',
                        help="Directory containing the models")
    parser.add_argument('--mode',
                        help="The processing to apply to the input image and tokens",
                        choices=['detect', 'recognize', 'extract'])
    parser.add_argument('--structure_config_path',
                        help="Filepath to the structure model config file")
    parser.add_argument('--structure_model_path', help="The path to the structure model")
    parser.add_argument('--detection_config_path',
                        help="Filepath to the detection model config file")
    parser.add_argument('--detection_model_path', help="The path to the detection model")
    parser.add_argument('--detection_device', default="cuda")
    parser.add_argument('--structure_device', default="cuda")
    parser.add_argument('--crops', '-p', action='store_true',
                        help='Output cropped data from table detections')
    parser.add_argument('--objects', '-o', action='store_true',
                        help='Output objects')
    parser.add_argument('--cells', '-l', action='store_true',
                        help='Output cells list')
    parser.add_argument('--html', '-m', action='store_true',
                        help='Output HTML')
    parser.add_argument('--csv', '-c', action='store_true',
                        help='Output CSV')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--visualize', '-z', action='store_true',
                        help='Visualize output')
    parser.add_argument('--crop_padding', type=int, default=10,
                        help="The amount of padding to add around a detected table when cropping.")

    return parser.parse_args()


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)

    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        return intersection.get_area() / bbox1_area

    return 0


def align_headers(headers, rows):
    """
    Adjust the header boundary to be the convex hull of the rows it intersects
    at least 50% of the height of.

    For now, we are not supporting tables with multiple headers, so we need to
    eliminate anything besides the top-most header.
    """

    aligned_headers = []

    for row in rows:
        row['column header'] = False

    header_row_nums = []
    for header in headers:
        for row_num, row in enumerate(rows):
            row_height = row['bbox'][3] - row['bbox'][1]
            min_row_overlap = max(row['bbox'][1], header['bbox'][1])
            max_row_overlap = min(row['bbox'][3], header['bbox'][3])
            overlap_height = max_row_overlap - min_row_overlap
            if overlap_height / row_height >= 0.5:
                header_row_nums.append(row_num)

    if len(header_row_nums) == 0:
        return aligned_headers

    header_rect = Rect()
    if header_row_nums[0] > 0:
        header_row_nums = list(range(header_row_nums[0] + 1)) + header_row_nums

    last_row_num = -1
    for row_num in header_row_nums:
        if row_num == last_row_num + 1:
            row = rows[row_num]
            row['column header'] = True
            header_rect = header_rect.include_rect(row['bbox'])
            last_row_num = row_num
        else:
            # Break as soon as a non-header row is encountered.
            # This ignores any subsequent rows in the table labeled as a header.
            # Having more than 1 header is not supported currently.
            break

    header = {'bbox': list(header_rect)}
    aligned_headers.append(header)

    return aligned_headers


def refine_table_structure(table_structure, class_thresholds):
    """
    Apply operations to the detected table structure objects such as
    thresholding, NMS, and alignment.
    """
    rows = table_structure["rows"]
    columns = table_structure['columns']

    # Process the headers
    column_headers = table_structure['column headers']
    column_headers = postprocess.apply_threshold(column_headers, class_thresholds["table column header"])
    column_headers = postprocess.nms(column_headers)
    column_headers = align_headers(column_headers, rows)

    # Process the row headers
    row_headers = table_structure['row headers']
    row_headers = postprocess.apply_threshold(row_headers, class_thresholds["table row header"])
    row_headers = postprocess.nms(row_headers)
    row_headers = align_headers(row_headers, columns)

    '''# Process spanning cells  ~~ original
    spanning_cells = [elem for elem in table_structure['spanning cells'] if not elem['projected row header']]
    projected_row_headers = [elem for elem in table_structure['spanning cells'] if elem['projected row header']]
    spanning_cells = postprocess.apply_threshold(spanning_cells, class_thresholds["table spanning cell"])
    projected_row_headers = postprocess.apply_threshold(projected_row_headers,
                                                        class_thresholds["table projected row header"])
    spanning_cells += projected_row_headers'''

    # Process spanning cells
    print(table_structure['spanning cells'])
    spanning_cells = [elem for elem in table_structure['spanning cells'] if not elem['projected row header'] and not elem['projected column header']]
    projected_row_headers = [elem for elem in table_structure['spanning cells'] if elem['projected row header']]
    projected_column_headers = [elem for elem in table_structure['spanning cells'] if elem['projected column header']]
    spanning_cells = postprocess.apply_threshold(spanning_cells, class_thresholds["table spanning cell"])
    projected_row_headers = postprocess.apply_threshold(projected_row_headers,
                                                        class_thresholds["table projected row header"])
    projected_column_headers = postprocess.apply_threshold(projected_column_headers,
                                                        class_thresholds["table projected column header"])
    spanning_cells += projected_row_headers
    # Align before NMS for spanning cells because alignment brings them into agreement
    # with rows and columns first; if spanning cells still overlap after this operation,
    # the threshold for NMS can basically be lowered to just above 0
    spanning_cells = postprocess.align_supercells(spanning_cells, rows, columns)
    spanning_cells = postprocess.nms_supercells(spanning_cells)

    postprocess.header_supercell_tree(spanning_cells)

    table_structure['columns'] = columns
    table_structure['rows'] = rows
    table_structure['spanning cells'] = spanning_cells
    table_structure['column headers'] = column_headers
    table_structure['row headers'] = row_headers

    return table_structure

def extract_words_from_images(input_folder, output_folder):
    '''This function takes 2 paths as input:
        for all image file in the input_folder_path it will extract the words
        using an OCR-Pytesserct and saves the extracted file in the Output_folder_path '''

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # Identify all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', 'PNG', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for image_file in image_files:
        try:
            # Finding path to input images
            image_path = os.path.join(input_folder, image_file)
            with Image.open(image_path).convert('RGB') as img:
                # improve the input image
                img = img.filter(ImageFilter.SHARPEN)
                # Use Tesseract to do OCR on the image
                extracted_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                words_data = [
                    {"bbox": [extracted_data['left'][i], extracted_data['top'][i],
                              extracted_data['left'][i] + extracted_data['width'][i],
                              extracted_data['top'][i] + extracted_data['height'][i]],
                     "text": extracted_data['text'][i].strip(),
                     "line_num": extracted_data['line_num'][i],
                     "block_num": extracted_data['block_num'][i]}
                    for i in range(len(extracted_data['text'])) if extracted_data['text'][i].strip()
                ]

                # Construct the full path to the output JSON file (same name as image file with .json extension)
                output_json_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + "_words.json")

                # Write the extracted word data to a JSON file
                with open(output_json_path, 'w') as json_file:
                    json.dump(words_data, json_file)
                    print("OCR file created")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

def outputs_to_objects(outputs, img_size, class_idx2name):
    m = outputs['pred_logits'].softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = class_idx2name[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects


def objects_to_crops(img, tokens, objects, class_thresholds, padding=20):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]

        cropped_img = img.crop(bbox)

        # Applying image filters to enhance quality
        cropped_img = cropped_img.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(cropped_img)
        cropped_img = enhancer.enhance(2)

        # Scaling the cropped image up for better resolution
        #scale_factor = 2
        #width, height = cropped_img.size
        #cropped_img = cropped_img.resize((width * scale_factor, height * scale_factor), Image.LANCZOS)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0] - bbox[0],
                             token['bbox'][1] - bbox[1],
                             token['bbox'][2] - bbox[0],
                             token['bbox'][3] - bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


def objects_to_structures(objects, tokens, class_thresholds):
    """
    Process the bounding boxes produced by the table structure recognition model into
    a *consistent* set of table structures (rows, columns, spanning cells, headers).
    This entails resolving conflicts/overlaps, and ensuring the boxes meet certain alignment
    conditions (for example: rows should all have the same width, etc.).
    """

    tables = [obj for obj in objects if obj['label'] == 'table']
    table_structures = []

    for table in tables:
        table_objects = [obj for obj in objects if iob(obj['bbox'], table['bbox']) >= 0.5]
        table_tokens = [token for token in tokens if iob(token['bbox'], table['bbox']) >= 0.5]

        structure = {}

        table_name = [obj for obj in table_objects if obj['label'] == 'table name']
        columns = [obj for obj in table_objects if obj['label'] == 'table column']
        rows = [obj for obj in table_objects if obj['label'] == 'table row']
        column_headers = [obj for obj in table_objects if obj['label'] == 'table column header']
        row_headers = [obj for obj in table_objects if obj['label'] == 'table row header']

        spanning_cells = [obj for obj in table_objects if obj['label'] == 'table spanning cell']
        for obj in spanning_cells:
            obj['projected row header'] = False
            obj['projected column header'] = False

        projected_row_headers = [obj for obj in table_objects if obj['label'] == 'table projected row header']
        for obj in projected_row_headers:
            obj['projected row header'] = True
            obj['projected column header'] = False
        spanning_cells += projected_row_headers

        projected_column_headers = [obj for obj in table_objects if obj['label'] == 'table projected column header']
        for obj in projected_column_headers:
            obj['projected column header'] = True
            obj['projected row header'] = False
        spanning_cells += projected_column_headers

        for obj in rows:
            obj['column header'] = False
            for header_obj in column_headers:
                if iob(obj['bbox'], header_obj['bbox']) >= 0.5:
                    obj['column header'] = True

        for obj in columns:
            obj['row header'] = False
            for header_obj in row_headers:
                if iob(obj['bbox'], header_obj['bbox']) >= 0.5:
                    obj['row header'] = True

        # Refine table structures
        rows = postprocess.refine_rows(rows, table_tokens, class_thresholds['table row'])
        columns = postprocess.refine_columns(columns, table_tokens, class_thresholds['table column'])

        # Shrink table bbox to just the total height of the rows
        # and the total width of the columns
        row_rect = Rect()
        for obj in rows:
            row_rect.include_rect(obj['bbox'])
        column_rect = Rect()
        for obj in columns:
            column_rect.include_rect(obj['bbox'])
        table['row_column_bbox'] = [column_rect[0], row_rect[1], column_rect[2], row_rect[3]]
        table['bbox'] = table['row_column_bbox']

        # Process the rows and columns into a complete segmented table
        columns = postprocess.align_columns(columns, table['row_column_bbox'])
        rows = postprocess.align_rows(rows, table['row_column_bbox'])

        structure['table name'] = table_name
        structure['rows'] = rows
        structure['columns'] = columns
        structure['column headers'] = column_headers
        structure['row headers'] = row_headers
        structure['spanning cells'] = spanning_cells

        if len(rows) > 0 and len(columns) > 1:
            structure = refine_table_structure(structure, class_thresholds)

        table_structures.append(structure)

    return table_structures

def structure_to_cells(table_structure, tokens):
    """
    Assuming the row, column, spanning cell, and header bounding boxes have
    been refined into a set of consistent table structures, process these
    table structures into table cells. This is a universal representation
    format for the table, which can later be exported to Pandas or CSV formats.
    Classify the cells as header/access cells or data cells
    based on if they intersect with the header bounding box.
    """
    table_name_header = table_structure['table name']
    columns = table_structure['columns']
    rows = table_structure['rows']
    spanning_cells = table_structure['spanning cells']
    cells = []
    subcells = []

    # Identify complete cells and subcells
    for column_num, column in enumerate(columns):
        for row_num, row in enumerate(rows):
            column_rect = Rect(list(column['bbox']))
            row_rect = Rect(list(row['bbox']))
            cell_rect = row_rect.intersect(column_rect)
            table_name_header = 'table name' in row and row['table name']
            row_header = 'row header' in column and column['row header']
            column_header = 'column header' in row and row['column header']
            cell = {'bbox': list(cell_rect), 'column_nums': [column_num], 'row_nums': [row_num],
                    'column header': column_header, 'row header': row_header, 'table name':table_name_header}

            cell['subcell'] = False
            for spanning_cell in spanning_cells:
                spanning_cell_rect = Rect(list(spanning_cell['bbox']))
                if (spanning_cell_rect.intersect(cell_rect).get_area()
                    / cell_rect.get_area()) > 0.5:
                    cell['subcell'] = True
                    break

            if cell['subcell']:
                subcells.append(cell)
            else:
                # cell text = extract_text_inside_bbox(table_spans, cell['bbox'])
                # cell['cell text'] = cell text
                cell['projected row header'] = False
                cell['projected column header'] = False
                cells.append(cell)

    for spanning_cell in spanning_cells:
        spanning_cell_rect = Rect(list(spanning_cell['bbox']))
        cell_columns = set()
        cell_rows = set()
        cell_rect = None
        column_header = True
        row_header = True
        for subcell in subcells:
            subcell_rect = Rect(list(subcell['bbox']))
            subcell_rect_area = subcell_rect.get_area()
            if (subcell_rect.intersect(spanning_cell_rect).get_area()
                / subcell_rect_area) > 0.5:
                if cell_rect is None:
                    cell_rect = Rect(list(subcell['bbox']))
                else:
                    cell_rect.include_rect(Rect(list(subcell['bbox'])))
                cell_rows = cell_rows.union(set(subcell['row_nums']))
                cell_columns = cell_columns.union(set(subcell['column_nums']))
                # By convention here, all subcells must be classified
                # as header cells for a spanning cell to be classified as a header cell;
                # otherwise, this could lead to a non-rectangular header region
                column_header = column_header and 'column header' in subcell and subcell['column header']
                row_header = row_header and 'row header' in subcell and subcell['row header']
        if len(cell_rows) > 0 and len(cell_columns) > 0:
            cell = {'bbox': list(cell_rect), 'column_nums': list(cell_columns), 'row_nums': list(cell_rows),
                    'column header': column_header, 'row header': row_header,
                    'projected row header': spanning_cell['projected row header'],
                    'projected column header': spanning_cell['projected column header'],
                    'table name':table_name_header}
            cells.append(cell)

    # Compute a confidence score based on how well the page tokens
    # slot into the cells reported by the model
    _, _, cell_match_scores = postprocess.slot_into_containers(cells, tokens)
    try:
        mean_match_score = sum(cell_match_scores) / len(cell_match_scores)
        min_match_score = min(cell_match_scores)
        confidence_score = (mean_match_score + min_match_score) / 2
    except:
        confidence_score = 0

    # Dilate rows and columns before final extraction
    # dilated_columns = fill_column_gaps(columns, table_bbox)
    dilated_columns = columns
    # dilated_rows = fill_row_gaps(rows, table_bbox)
    dilated_rows = rows
    for cell in cells:
        column_rect = Rect()
        for column_num in cell['column_nums']:
            column_rect.include_rect(list(dilated_columns[column_num]['bbox']))
        row_rect = Rect()
        for row_num in cell['row_nums']:
            row_rect.include_rect(list(dilated_rows[row_num]['bbox']))
        cell_rect = column_rect.intersect(row_rect)
        cell['bbox'] = list(cell_rect)

    span_nums_by_cell, _, _ = postprocess.slot_into_containers(cells, tokens, overlap_threshold=0.001,
                                                               unique_assignment=True, forced_assignment=False)

    for cell, cell_span_nums in zip(cells, span_nums_by_cell):
        cell_spans = [tokens[num] for num in cell_span_nums]
        # TODO: Refine how text is extracted; should be character-based, not span-based;
        # but need to associate
        cell['cell text'] = postprocess.extract_text_from_spans(cell_spans, remove_integer_superscripts=False)
        cell['spans'] = cell_spans

    # Adjust the row, column, and cell bounding boxes to reflect the extracted text
    num_rows = len(rows)
    rows = postprocess.sort_objects_top_to_bottom(rows)
    num_columns = len(columns)
    columns = postprocess.sort_objects_left_to_right(columns)
    min_y_values_by_row = defaultdict(list)
    max_y_values_by_row = defaultdict(list)
    min_x_values_by_column = defaultdict(list)
    max_x_values_by_column = defaultdict(list)
    for cell in cells:
        min_row = min(cell["row_nums"])
        max_row = max(cell["row_nums"])
        min_column = min(cell["column_nums"])
        max_column = max(cell["column_nums"])
        for span in cell['spans']:
            min_x_values_by_column[min_column].append(span['bbox'][0])
            min_y_values_by_row[min_row].append(span['bbox'][1])
            max_x_values_by_column[max_column].append(span['bbox'][2])
            max_y_values_by_row[max_row].append(span['bbox'][3])
    for row_num, row in enumerate(rows):
        if len(min_x_values_by_column[0]) > 0:
            row['bbox'][0] = min(min_x_values_by_column[0])
        if len(min_y_values_by_row[row_num]) > 0:
            row['bbox'][1] = min(min_y_values_by_row[row_num])
        if len(max_x_values_by_column[num_columns - 1]) > 0:
            row['bbox'][2] = max(max_x_values_by_column[num_columns - 1])
        if len(max_y_values_by_row[row_num]) > 0:
            row['bbox'][3] = max(max_y_values_by_row[row_num])
    for column_num, column in enumerate(columns):
        if len(min_x_values_by_column[column_num]) > 0:
            column['bbox'][0] = min(min_x_values_by_column[column_num])
        if len(min_y_values_by_row[0]) > 0:
            column['bbox'][1] = min(min_y_values_by_row[0])
        if len(max_x_values_by_column[column_num]) > 0:
            column['bbox'][2] = max(max_x_values_by_column[column_num])
        if len(max_y_values_by_row[num_rows - 1]) > 0:
            column['bbox'][3] = max(max_y_values_by_row[num_rows - 1])
    for cell in cells:
        row_rect = Rect()
        column_rect = Rect()
        for row_num in cell['row_nums']:
            row_rect.include_rect(list(rows[row_num]['bbox']))
        for column_num in cell['column_nums']:
            column_rect.include_rect(list(columns[column_num]['bbox']))
        cell_rect = row_rect.intersect(column_rect)
        if cell_rect.get_area() > 0:
            cell['bbox'] = list(cell_rect)
            pass

    return cells, confidence_score

def cells_to_csv(cells):
    if len(cells) > 0:
        num_columns = max([max(cell['column_nums']) for cell in cells]) + 1
        num_rows = max([max(cell['row_nums']) for cell in cells]) + 1
    else:
        return

    header_cells = [cell for cell in cells if cell['column header']]
    if len(header_cells) > 0:
        max_header_row = max([max(cell['row_nums']) for cell in header_cells])
    else:
        max_header_row = -1

    table_array = np.empty([num_rows, num_columns], dtype="object")
    if len(cells) > 0:
        for cell in cells:
            for row_num in cell['row_nums']:
                for column_num in cell['column_nums']:
                    table_array[row_num, column_num] = cell["cell text"]

    header = table_array[:max_header_row + 1, :]
    flattened_header = []
    for col in header.transpose():
        flattened_header.append(' | '.join(OrderedDict.fromkeys(col)))
    df = pd.DataFrame(table_array[max_header_row + 1:, :], index=None, columns=flattened_header)
    #df.to_excel('ex.xlsx', index=None)

    return df.to_csv(index=None)

def cells_to_html(cells):
    cells = sorted(cells, key=lambda k: min(k['column_nums']))
    cells = sorted(cells, key=lambda k: min(k['row_nums']))

    table = ET.Element("table")
    current_row = -1

    for cell in cells:
        this_row = min(cell['row_nums'])

        attrib = {}
        colspan = len(cell['column_nums'])
        if colspan > 1:
            attrib['colspan'] = str(colspan)
        rowspan = len(cell['row_nums'])
        if rowspan > 1:
            attrib['rowspan'] = str(rowspan)
        if this_row > current_row:
            current_row = this_row
            if cell['column header']:
                cell_tag = "th"
                row = ET.SubElement(table, "thead")
            else:
                cell_tag = "td"
                row = ET.SubElement(table, "tr")
        tcell = ET.SubElement(row, cell_tag, attrib=attrib)
        tcell.text = cell['cell text']

    return str(ET.tostring(table, encoding="unicode", short_empty_elements=False))

def visualize_detected_tables(img, det_tables, out_path):
    plt.imshow(img, interpolation="lanczos")
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        else:
            continue

        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor='none', facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=0,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                             label='Table', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                             label='Table (rotated)', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
               fontsize=10, ncol=2)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

    return

def visualize_cells(img, cells, out_path):
    plt.imshow(img, interpolation="lanczos")
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()

    for cell in cells:
        bbox = cell['bbox']

        if cell['column header']:
            facecolor = (0.9, 0, 0.9)
            edgecolor = (0.9, 0, 0.9)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif cell['projected row header']:
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif cell['row header']:
            facecolor = (1, 0, 0)
            edgecolor = (1, 0, 0)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif cell['projected column header']:
            facecolor = (0.49, 0.15, 0.8)
            edgecolor = (0.49, 0.15, 0.8)
            linewidth = 2
            hatch = '//////'
        elif cell['table name']:
            facecolor = (0, 0.8, 0)
            edgecolor = (0, 0.8, 0)
            linewidth = 2
            hatch = '//////'
        else:
            facecolor = (0.3, 0.74, 0.8)
            edgecolor = (0.3, 0.7, 0.6)
            alpha = 0.3
            linewidth = 2
            hatch = '\\\\\\\\\\\\'

        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor='none', facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=0,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(0.3, 0.74, 0.8), edgecolor=(0.3, 0.7, 0.6),
                             label='Data cell', hatch='\\\\\\\\\\\\', alpha=0.3),
                       Patch(facecolor=(1, 0, 0), edgecolor=(1, 0, 0),
                             label='Row header cell', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.9, 0, 0.9), edgecolor=(0.9, 0, 0.9),
                             label='Column header cell', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                             label='Projected row header cell', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0, 0.8, 0), edgecolor=(0, 0.8, 0),
                             label='Table name cell', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.49, 0.15, 0.8), edgecolor=(0.49, 0.15, 0.8),
                             label='Projected column header cell', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
               fontsize=10, ncol=3)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

    return

class TableExtractionPipeline(object):
    def __init__(self, det_device=None, str_device=None,
                 det_model=None, str_model=None,
                 det_model_path=None, str_model_path=None,
                 det_config_path=None, str_config_path=None,
                 model_dir=None, in_dir=None, out_dir=None):

        self.det_device = det_device
        self.str_device = str_device
        self.model_dir = model_dir
        self.in_dir = in_dir
        self.out_dir = out_dir

        self.det_class_name2idx = get_class_map('detection')
        self.det_class_idx2name = {v: k for k, v in self.det_class_name2idx.items()}
        self.det_class_thresholds = detection_class_thresholds

        self.str_class_name2idx = get_class_map('structure')
        self.str_class_idx2name = {v: k for k, v in self.str_class_name2idx.items()}
        self.str_class_thresholds = structure_class_thresholds

        if not det_config_path is None:
            with open(det_config_path, 'r') as f:
                det_config = json.load(f)
            det_args = type('Args', (object,), det_config)
            det_args.device = det_device
            self.det_model, _, _ = build_model(det_args)
            print("Detection model initialized.")

            if not det_model_path is None:
                self.det_model.load_state_dict(torch.load(det_model_path,
                                                          map_location=torch.device(det_device)))
                self.det_model.to(det_device)
                self.det_model.eval()
                print("Detection model weights loaded.")
            else:
                self.det_model = None

        if not str_config_path is None:
            with open(str_config_path, 'r') as f:
                str_config = json.load(f)
            str_args = type('Args', (object,), str_config)
            str_args.device = str_device
            self.str_model, _, _ = build_model(str_args)
            print("Structure model initialized.")

            if not str_model_path is None:
                self.str_model.load_state_dict(torch.load(str_model_path,
                                                          map_location=torch.device(str_device)))
                self.str_model.to(str_device)
                self.str_model.eval()
                print("Structure model weights loaded.")
            else:
                self.str_model = None

    def __call__(self, page_image, page_tokens=None):
        return self.extract(self, page_image, page_tokens)

    def detect(self, img, tokens=None, out_objects=True, out_crops=False, crop_padding=10):
        out_formats = {}
        if self.det_model is None:
            print("No detection model loaded.")
            return out_formats

        # Transform the image how the model expects it
        img_tensor = detection_transform(img)

        # Run input image through the model
        outputs = self.det_model([img_tensor.to(self.det_device)])

        # Post-process detected objects, assign class labels
        objects = outputs_to_objects(outputs, img.size, self.det_class_idx2name)
        if out_objects:
            out_formats['objects'] = objects
        if not out_crops:
            return out_formats

        # Crop image and tokens for detected table
        if out_crops:
            tables_crops = objects_to_crops(img, tokens, objects, self.det_class_thresholds,
                                            padding=crop_padding)
            out_formats['crops'] = tables_crops

        return out_formats

    def recognize(self, img, tokens=None, out_objects=False, out_cells=False,
                  out_html=False, out_csv=False):
        out_formats = {}
        if self.str_model is None:
            print("No structure model loaded.")
            return out_formats

        if not (out_objects or out_cells or out_html or out_csv):
            print("No output format specified")
            return out_formats

        # Transform the image how the model expects it
        img_tensor = structure_transform(img)

        # Run input image through the model
        outputs = self.str_model([img_tensor.to(self.str_device)])

        # Post-process detected objects, assign class labels
        objects = outputs_to_objects(outputs, img.size, self.str_class_idx2name)
        if out_objects:
            out_formats['objects'] = objects
        if not (out_cells or out_html or out_csv):
            return out_formats

        # Further process the detected objects so they correspond to a consistent table
        tables_structure = objects_to_structures(objects, tokens, self.str_class_thresholds)

        # Enumerate all table cells: grid cells and spanning cells
        tables_cells = [structure_to_cells(structure, tokens)[0] for structure in tables_structure]
        if out_cells:
            out_formats['cells'] = tables_cells
        if not (out_html or out_csv):
            return out_formats

        # Convert cells to HTML
        if out_html:
            tables_htmls = [cells_to_html(cells) for cells in tables_cells]
            out_formats['html'] = tables_htmls

        # Convert cells to CSV, including flattening multi-row column headers to a single row
        if out_csv:
            tables_csvs = [cells_to_csv(cells) for cells in tables_cells]
            out_formats['csv'] = tables_csvs
            out_formats['xlsx'] = tables_csvs
            #tables_csvs_df = pd.DataFrame(tables_csvs)
            #writer = pd.ExcelWriter(os.path.join(), engine='xlsxwriter')
            #tables_csvs_df.to_excel(writer, sheet_name='1', index=False)
            #writer.save()

        return out_formats

    def extract(self, img, tokens=None, out_objects=True, out_crops=False, out_cells=False,
                out_html=False, out_csv=False, crop_padding=10):

        detect_out = self.detect(img, tokens=tokens, out_objects=False, out_crops=True,
                                 crop_padding=crop_padding)
        cropped_tables = detect_out['crops']

        extracted_tables = []
        for table in cropped_tables:
            img = table['image']
            tokens = table['tokens']

            extracted_table = self.recognize(img, tokens=tokens, out_objects=out_objects,
                                             out_cells=out_cells, out_html=out_html, out_csv=out_csv)
            extracted_table['image'] = img
            extracted_table['tokens'] = tokens
            extracted_tables.append(extracted_table)

        return extracted_tables

    def perform_final_step(self, excel_folder_path, pdf_folder_path):
        """
        This function will perform the final step as mentioned in the
        documentation. It will extract and structure the values from
        the excel files using regex patterns and return them as a
        dictionary. This will be the major function that will perform
            the final step for all the excel files in all the folders i.e.
            baseline, camelot and tabula.
            """

        print("Performing final Step")
        all_files = self.fs_folder(
            path_to_excel_folder=excel_folder_path,
            path_to_pdf_folder=pdf_folder_path
        )
        self.extracted_values = all_files
        print("final Step completed")

    def fs_folder(
            self,
            path_to_excel_folder: str,
            path_to_pdf_folder: str
    ) -> dict:
        """
        This is an internal function that will perform the final step
        for all the excel files in the folder specified.
        """

        list_of_files = get_list_of_files_with_ext(
            path_to_folder=path_to_excel_folder,
            ext=".xlsx",
            verbose=True
        )

        all_files_extracted = {}

        # Go through all the files one by one and get the
        # values that were extracted
        for file in list_of_files:
            #filename = str(basename(file)).rsplit(sep=".")[0]
            filename = os.path.splitext(file)[0]

            # Just run the function for now
            extracted_vals = self.fs_file(
                path_to_excel_file=file,
                path_to_pdf_file=filename + ".pdf"
            )

            all_files_extracted[filename] = extracted_vals

        return all_files_extracted

    def save_to_excel(
            self,
            path: str = "extracted_data.xlsx"
    ) -> None:
        """
        This is the path where the excel file containing the extracted
        values will be saved.

        Args:
            path:
                This is the path to the excel file where the extracted
                values will be saved.
        """

        print()

        print("Saving to excel")
        print("---------------")

        final_list = []

        for name, prop_type in self.extracted_values.items():

            # Get the values
            thermal = prop_type.get("thermal")
            electrical = prop_type.get("electrical")

            year = prop_type.get("misc").get("year")

            length = prop_type.get("mech").get("length")
            width = prop_type.get("mech").get("width")

            # Assuming equal lengths of extracted arrays of values
            if electrical is not None:
                value_count = []

                for value_type, value_list in electrical.items():
                    if value_list is not None:
                        value_count.append(len(value_list))

                # most_freq_count = mode(value_count)
                value_count_cleaned = np.nan_to_num(value_count, nan=np.nan)

                # Filter out NaN values
                value_count_cleaned = value_count_cleaned[~np.isnan(value_count_cleaned)]

                if len(value_count_cleaned) == 0:
                    most_freq_count = np.nan
                else:
                    # Calculate the most frequent count
                    most_freq_count = int(stats.mode(value_count_cleaned)[0])
            else:
                most_freq_count = 1

            if math.isnan(most_freq_count):
                curr_file_list = [[]]
            else:
                name = name.split("\\")[-1]
                curr_file_list = [[name] * most_freq_count]
                curr_file_list.append([year] * most_freq_count)

                curr_file_list.append([length] * most_freq_count)
                curr_file_list.append([width] * most_freq_count)

                # Adding electrical properties
                elec_prop_types = ["eff", "pmpp", "vmpp", "impp", "voc", "isc", "ff"]

                for prop in elec_prop_types:

                    if electrical is not None:
                        vals = electrical.get(prop)
                    else:
                        vals = None

                    if vals is None:
                        curr_file_list.append([""] * most_freq_count)
                    else:
                        curr_file_list.append(vals)

                # Adding thermal properties
                thermal_prop_types = ["isc", "pmpp", "voc"]

                for prop in thermal_prop_types:

                    if thermal is not None:
                        vals = thermal.get(prop)
                    else:
                        vals = None

                    if vals is None:
                        curr_file_list.append([""] * most_freq_count)
                    else:
                        curr_file_list.append(vals * most_freq_count)

            # Transpose the list
            curr_file_list = list(map(list, zip(*curr_file_list)))

            final_list.extend(curr_file_list)

        # Create a dataframe
        final_df = pd.DataFrame(final_list,
                                columns=[
                                    "name",
                                    "year",
                                    "length",
                                    "width",
                                    "E/eff",
                                    "E/pmpp",
                                    "E/vmpp",
                                    "E/impp",
                                    "E/voc",
                                    "E/isc",
                                    "E/ff",
                                    "T/isc",
                                    "T/pmpp",
                                    "T/voc"
                                ]
                                )


        final_df.to_excel(path)
        print("final excel created")
        # print(final_df)

        # Write to the excel file
        #with pd.ExcelWriter(path=path, mode='w') as writer:
            #final_df.to_excel(writer,index=False)

        #wb = openpyxl.Workbook()
        #ws = wb.active
        #ws.append(final_df)
        #wb.save(path)

    def fs_file(
            self,
            path_to_excel_file: str,
            path_to_pdf_file: str
    ) -> dict:
        """
        This is an internal function that will perform the final step
        for the excel file specified and return a dictionary of items
        that were extracted. It will combine the electrical and thermal
        properties together into a single dictionary and just keep the
        values that were extracted and not the rows where they were
        found on.
        """

        # Load the yaml file that contains all the patterns for
        # detecting the correct columns and the values
        with open(self.model_dir + "/patterns.yaml", "r", encoding='utf-8') as stream:
            try:
                patterns = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)

        # Get type specific patterns
        elec_patterns = patterns.get("electrical")
        therm_patterns = patterns.get("temperature")
        mech_patterns = patterns.get("mechanical")

        curr_ds = Datasheet(
            path_to_pdf=path_to_pdf_file,
            path_to_excel=path_to_excel_file,
            path_to_clf=self.model_dir + "/nb_classifier.pickle",
            path_to_vec=self.model_dir + "/vectoriser.pickle"
        )

        curr_ds.extract_electrical_props(patterns=elec_patterns)
        elec_extracted = curr_ds.extracted_elec

        curr_ds.extract_temp_props(patterns=therm_patterns)
        therm_extracted = curr_ds.extracted_temp

        curr_ds.extract_mech_props(patterns=mech_patterns)
        mech_extracted = curr_ds.extracted_mech

        curr_ds.extract_misc_props()
        misc_extracted = curr_ds.extracted_misc

        if elec_extracted is not None:
            for key, item in elec_extracted.items():
                vals = item.get("vals")

                if vals is not None:
                    vals = [str(x) for x in vals]

                elec_extracted[key] = vals

        if therm_extracted is not None:
            for key, item in therm_extracted.items():
                vals = item.get("vals")

                if vals is not None:
                    vals = [str(x) for x in vals]

                therm_extracted[key] = vals

        return {
            "electrical": elec_extracted,
            "thermal": therm_extracted,
            "mech": mech_extracted,
            "misc": misc_extracted
        }

def output_result(key, val, args, img, img_file):
    if key == 'objects':
        if args.verbose:
            print(val)
        out_file = img_file.replace(".jpg", "_objects.json")
        with open(os.path.join(args.out_dir, out_file), 'w') as f:
            json.dump(val, f)
        if args.visualize:
            out_file = img_file.replace(".jpg", "_fig_tables.jpg")
            out_path = os.path.join(args.out_dir, out_file)
            visualize_detected_tables(img, val, out_path)
    elif not key == 'image' and not key == 'tokens':
        for idx, elem in enumerate(val):
            if key == 'crops':
                for idx, cropped_table in enumerate(val):
                    out_img_file = img_file.replace(".jpg", "_table_{}.jpg".format(idx))

                    # Ensuring that the output directory exists
                    os.makedirs(args.out_dir, exist_ok=True)

                    cropped_table['image'].save(os.path.join(args.out_dir,
                                                             out_img_file), 'JPEG', quality=95, optimize=True)
                    out_words_file = out_img_file.replace(".jpg", "_words.json")
                    with open(os.path.join(args.out_dir, out_words_file), 'w') as f:
                        json.dump(cropped_table['tokens'], f)
            elif key == 'cells':
                out_file = img_file.replace(".jpg", "_{}_objects.json".format(idx))
                with open(os.path.join(args.out_dir, out_file), 'w') as f:
                    json.dump(elem, f)
                if args.verbose:
                    print(elem)
                if args.visualize:
                    out_file = img_file.replace(".jpg", "_fig_cells.jpg")
                    out_path = os.path.join(args.out_dir, out_file)
                    visualize_cells(img, elem, out_path)
            elif key == "csv":
                out_file = img_file.replace(".jpg", "_{}.csv".format(idx))
                with open(os.path.join(args.out_dir, out_file), 'w') as f:
                    if elem is not None:
                        f.write(elem)
                if args.verbose:
                    print(elem)
            elif key == "xlsx":
                # Parse the CSV content
                parsed_data = list(csv.reader(StringIO(elem), delimiter=',', quotechar='"'))

                # Convert the parsed data into a DataFrame
                df = pd.DataFrame(parsed_data)

                out_file =  os.path.join(args.out_dir ,img_file.replace(".jpg", "_{}.xlsx".format(idx)))

                df.to_excel(out_file, index=False, header=False)
                '''out_file = img_file.replace(".jpg", "_{}.xlsx".format(idx))
                wb = openpyxl.Workbook()
                ws = wb.active

                if elem is not None:
                    # Split the string into rows based on newline characters
                    rows = elem.split('\r\n')
                    for row in rows:
                        # Split each row into cells based on comma separation
                        cells = row.split(',')
                        ws.append(cells)
                    wb.save(os.path.join(args.out_dir, out_file))'''




def get_list_of_files_with_ext(
    path_to_folder: str,
    ext: str,
    randomise: bool = False,
    verbose: bool = True
    ) -> list:
    """
    This function will go through all the files in the given
    folder and make a list of files with the provided extension.
    This can be used, for example, to filter out the required
    files in the folder.

    Parameters:
        path_to_folder:
            This is the path to folder that will be scanned
            for the required files.

        ext:
            This is the extension of the files that will be
            selected from the folder.

        randomise:
            If this flag is set to True, then the list of files
            will be shuffled before being returned.

        verbose:
            If this flag is set to True, then this function will
            display the information from the folder.

    Returns:
        list_of_files:
            This is the list of files in the provided
            directory (folder) that matches the extension
            provided. It contains the full path to the files
            not just the name of the files.
    """

    list_of_files = []

    # Evaluate all files in the directory
    for file in listdir(path_to_folder):

        # Skip the hidden files
        # In linux and macOS, the hidden files start
        # with '.'
        if not file.startswith('.'):

            # Get the files with the specified extension
            if file.endswith(ext):
                full_path = join(path_to_folder, file)
                list_of_files.append(full_path)

    if verbose:
        print()
        print("Looking for " + ext + " files in folder: " + path_to_folder)
        print()
        print("Total " + ext + " files found: " + str(len(list_of_files)))

    # Shuffle the list of files captured
    if randomise:
        random.shuffle(list_of_files)

    return list_of_files


def merge_tables(excel_files_folder, output_folder, pdf_files):
    for pdf_file in pdf_files:
        # Get the base name of the PDF file
        base_name = os.path.splitext(pdf_file)[0]

        # Initialize a list to store DataFrames
        dfs = []

        # Iterate through Excel files in the same folder
        for filename in os.listdir(excel_files_folder):
            if filename.startswith(base_name) and filename.endswith('.xlsx'):
                # Read the Excel file
                df = pd.read_excel(os.path.join(excel_files_folder, filename))

                # Append the DataFrame to combined_df
                dfs.append(df)

        # Concatenate all DataFrames in dfs list
        if dfs:
            combined_file_path = os.path.join(output_folder, base_name + '.xlsx')
            with pd.ExcelWriter(combined_file_path) as writer:
                for idx, df in enumerate(dfs):
                    # Write each DataFrame to a separate worksheet
                    sheet_name = f'{base_name}_{idx}'
                    df.to_excel(writer, index=False, header=False, sheet_name=sheet_name)

            '''combined_df = pd.concat(dfs, ignore_index=True)

            # Write combined_df to a new Excel file
            combined_file_path = os.path.join(output_folder, base_name + '.xlsx')
            with pd.ExcelWriter(combined_file_path) as writer:
                combined_df.to_excel(writer, index=False, sheet_name=base_name)'''
        else:
            print("No matching files found for:", pdf_file)

'''def csv_to_excel(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            csv_file = os.path.join(folder_path, filename)

            # Read the CSV file
            df = pd.read_csv(csv_file)
            # Extract the filename without the .csv extension for the sheet name
            sheet_name = os.path.splitext(filename)[0]
            # Create the Excel file with the same name as the CSV (minus .csv)
            excel_file = os.path.splitext(csv_file)[0] + ".xlsx"

            # Write the DataFrame to the Excel file with the extracted sheet name
            df.to_excel(excel_file, sheet_name=sheet_name, index=False)

            print(f"Converted '{filename}' to '{excel_file}'.")'''


def main():
    #setting the tesseract path for OCR
    #pytesseract.pytesseract.tesseract_cmd = r'D:\Users\swa86085\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    global pdf_files_dir, image_dir, pdf_files
    pytesseract.pytesseract.tesseract_cmd = r'S:\23502\2\216_PVM\Aktuell\01_Orga\21631_MAS_TeamModulbewertung\03_Arbeitsordner\Swathi_Thiruvengadam\Tesseract-OCR\tesseract.exe'
    #pytesseract.pytesseract.tesseract_cmd = r'/net/s/23502/2/280_PVM/Aktuell/01_Orga/23131_MAS_TeamModulbewertung/03_Arbeitsordner/Swathi_Thiruvengadam/Tesseract-OCR/tesseract.exe'
    start_time = time.time()

    args = get_args()
    print(args.__dict__)
    print('-' * 100)

    # create an output directory if it does not exists
    if not args.out_dir is None and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Create inference pipeline
    print("Creating inference pipeline")
    pipe = TableExtractionPipeline(det_device=args.detection_device,
                                   str_device=args.structure_device,
                                   det_config_path=args.detection_config_path,
                                   det_model_path=args.detection_model_path,
                                   str_config_path=args.structure_config_path,
                                   str_model_path=args.structure_model_path,
                                   model_dir = args.model_dir,
                                   in_dir = args.in_dir,
                                   out_dir = args.out_dir)

    # Load images
    # convert the pdf file to a images
    if args.mode == 'detect' or args.mode == 'extract':
        pdf_files = [f for f in os.listdir(args.in_dir) if f.lower().endswith(('.pdf'))]

        # Directory for storing intermediate images
        image_dir = os.path.join(args.in_dir, '..', 'images')
        os.makedirs(image_dir, exist_ok=True)

        for pdf in pdf_files:
            pdf_path = os.path.join(args.in_dir, pdf)
            pdf_image = convert_from_path(pdf_path, dpi=300)
            image_base_path = os.path.join(image_dir, os.path.splitext(pdf)[0])
            #for idx in range(len(pdf_image)):
                #pdf_image[idx].save(image_path + '_' + str(idx + 1) + '.jpg', 'JPEG')
            for idx, image in enumerate(pdf_image):
                image_path = f"{image_base_path}_{idx + 1}.jpg"
                image.save(image_path, 'JPEG')

    elif args.mode == 'recognize':
        pdf_files_dir = os.path.join(args.in_dir, '..', 'pdf')
        pdf_files = [f for f in os.listdir(pdf_files_dir) if f.lower().endswith(('.pdf'))]
        image_dir = os.path.join(args.in_dir, '..', 'images')
        os.makedirs(image_dir, exist_ok=True)

        for pdf in pdf_files:
            pdf_path = os.path.join(pdf_files_dir, pdf)
            pdf_image = convert_from_path(pdf_path, dpi=300)
            image_path = os.path.join(image_dir, os.path.splitext(pdf)[0])
            #for idx in range(len(pdf_image)):
                #pdf_image[idx].save(image_path + '_' + str(idx + 1) + '.jpg', 'JPEG')
            for idx, image in enumerate(pdf_image):
                image_path = f"{image_path}_{idx + 1}.jpg"
                image.save(image_path, 'JPEG')


    # create the OCR words
    if args.mode == 'detect' or args.mode == 'extract':
        words_dir = "../inferences/detectionwords"
        # OCR program to extract the words for all the image files present in our folder
        extract_words_from_images(image_dir, words_dir)
    elif args.mode == 'recognize':
        words_dir = "../inferences/words"
        image_dir = args.in_dir


    # List only image files in the input folder
    img_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    num_files = len(img_files)
    random.shuffle(img_files)

    # note : change directory paths to make more sense
    img_input_folder = os.path.join(image_dir, '..', 'detectionOutput')
    words_output_folder = os.path.join(image_dir, '..', 'words')

    for count, img_file in enumerate(img_files):
        print("({}/{})".format(count + 1, num_files))
        #img_path = os.path.join(args.image_dir, img_file)
        img_path = image_dir +'/' +  img_file
        img = Image.open(img_path).convert('RGB')
        print("Image loaded.")

        if not words_dir is None:
            tokens_path = words_dir + "/" + img_file.replace(".jpg", "_words.json")
            # tokens_path = os.path.join(args.words_dir, img_file.replace(".jpg", ".json"))
            with open(tokens_path, 'r') as f:
                tokens = json.load(f)

                # Handle dictionary format
                if type(tokens) is dict and 'words' in tokens:
                    tokens = tokens['words']

                # 'tokens' is a list of tokens
                # Need to be in a relative reading order
                # If no order is provided, use current order
                for idx, token in enumerate(tokens):
                    if not 'span_num' in token:
                        token['span_num'] = idx
                    if not 'line_num' in token:
                        token['line_num'] = 0
                    if not 'block_num' in token:
                        token['block_num'] = 0
        else:
            tokens = []

        if args.mode == 'recognize':
            extracted_table = pipe.recognize(img, tokens, out_objects=args.objects, out_cells=args.csv,
                                             out_html=args.html, out_csv=args.csv)
            print("Table(s) recognized.")

            for key, val in extracted_table.items():
                output_result(key, val, args, img, img_file)

            # merge into same file
            merge_tables(args.out_dir, pdf_files_dir, pdf_files)


        if args.mode == 'detect':
            detected_tables = pipe.detect(img, tokens, out_objects=args.objects, out_crops=args.crops)
            print("Table(s) detected.")

            for key, val in detected_tables.items():
                output_result(key, val, args, img, img_file)

        if args.mode == 'extract':
            extracted_tables = pipe.extract(img, tokens, out_objects=args.objects, out_cells=args.csv,
                                            out_html=args.html, out_csv=args.csv,
                                            crop_padding=args.crop_padding)
            print("Table(s) extracted.")

            for table_idx, extracted_table in enumerate(extracted_tables):
                for key, val in extracted_table.items():
                    output_result(key, val, args, extracted_table['image'],
                                  img_file.replace('.jpg', '_{}.jpg'.format(table_idx)))

            # merge into same file
            merge_tables(args.out_dir, args.in_dir, pdf_files)

    if args.mode == 'detect':
        extract_words_from_images(img_input_folder, words_output_folder)

    if args.mode == 'recognize':
        pipe.perform_final_step(pdf_files_dir, pdf_files_dir)
        pipe.save_to_excel(path=os.path.join(pdf_files_dir, "extracted_tables.xlsx"))

    if args.mode == 'extract':
        pipe.perform_final_step(args.in_dir, args.in_dir)
        pipe.save_to_excel(path=os.path.join(args.in_dir, "extracted_tables.xlsx"))

    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total time taken by Table-Transformer to extract tabular data is : {total_time} seconds")

if __name__ == "__main__":
    main()