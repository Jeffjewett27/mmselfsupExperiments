import json
import sys
import numpy as np
import pandas as pd

def read_annotation_file(file):
    with open(file, 'r') as openfile:
        json_object = json.load(openfile)
    return json_object

def write_annotation_file(file, annotations):
    with open(file, 'w') as openfile:
        json.dump(annotations, openfile, indent = 2)

def write_img_list(file, imgs):
    lines = '\n'.join(imgs.tolist())
    with open(file, 'w') as openfile:
        openfile.write(lines)

def split_annotations(annotations, train, test, valid):
    trainNorm = train / (train + test + valid)
    testNorm = test / (train + test + valid)

    images = annotations['images']
    files = np.asarray([img['file_name'] for img in images])
    n = len(files)
    print(f'{n}->{n*train}, {n*valid}, {n*test}')
    np.random.shuffle(files)
    trainX, testX, validX = np.split(files, [int(trainNorm * n), int((testNorm + trainNorm) * n)])
    return trainX, testX, validX

def sample_n_annotations(annotations, n):
    images = annotations['images']
    files = np.asarray([img['file_name'] for img in images])
    np.random.shuffle(files)
    keep, _, _ = np.split(files, [n, len(files)-n])
    return keep

def sample_annotations_multiple(annotations, arr):
    arr = [sum(arr[:i+1]) for i, _ in enumerate(arr)]
    images = annotations['images']
    files = np.asarray([img['file_name'] for img in images])
    np.random.shuffle(files)
    return np.split(files, arr)[:-1]

def filter_annotation_images(annotations, filterImages):
    filterDf = pd.DataFrame({'file_name': filterImages})
    imageDf = pd.DataFrame(annotations['images']) \
        .merge(filterDf, on ='file_name', how ="inner")
    annDf = pd.DataFrame(annotations['annotations']) \
        .merge(imageDf['id'], left_on='image_id', right_on='id', how='inner', suffixes=(None, '_filter')) \
        .drop(columns='id_filter')
    
    newImgs = imageDf.to_dict(orient='records')
    newAnns = annDf.to_dict(orient='records')
    return {
        'images': newImgs,
        'categories': annotations['categories'],
        'annotations': newAnns
    }

if __name__ == "__main__":
    annFile = sys.argv[1]
    annotations = read_annotation_file(annFile)

    # trainX, testX, validX = split_annotations(annotations, 0.941, 0.009, 0.05)
    # trainX
    # trainAnns = filter_annotation_images(annotations, trainX)
    # testAnns = filter_annotation_images(annotations, testX)
    # validAnns = filter_annotation_images(annotations, validX)

    # sample = sample_n_annotations(annotations, 5000)
    # write_img_list('data/coco5kImages.txt', sample)
    # sampledAnns = filter_annotation_images(annotations, sample)
    # write_annotation_file('data/coco5kAnnotations.json', sampledAnns)

    train, test, valid = sample_annotations_multiple(annotations, [5000, 1000, 1000])
    trainAnn = filter_annotation_images(annotations, train)
    testAnn = filter_annotation_images(annotations, test)
    validAnn = filter_annotation_images(annotations, valid)
    write_img_list('data/cocotrain5k.txt', train)
    write_img_list('data/cocotest1k.txt', test)
    write_img_list('data/cocovalid1k.txt', valid)
    write_annotation_file('data/cocotrain5k.json', trainAnn)
    write_annotation_file('data/cocotest1k.json', testAnn)
    write_annotation_file('data/cocovalid1k.json', validAnn)