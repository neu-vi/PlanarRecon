# This file is derived from [PlaneRCNN](https://github.com/NVlabs/planercnn).
# Originating Author: Chen Liu
# Modified for [PlanarRecon](https://github.com/neu-vi/PlanarRecon) by Yiming Xie.

# Original header:
# Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license
# (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

import numpy as np
import sys
import os
from plyfile import PlyData, PlyElement
import json
import glob
import cv2
import matplotlib.pyplot as plt
import torch
from tools.random_color import random_color

numPlanes = 200
numPlanesPerSegment = 2
planeAreaThreshold = 100
numIterations = 100
numIterationsPair = 1000
planeDiffThreshold = 0.05
fittingErrorThreshold = planeDiffThreshold
orthogonalThreshold = np.cos(np.deg2rad(60))
parallelThreshold = np.cos(np.deg2rad(30))


## Fit a 3D plane from points
def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]))[0]
    return


class ColorPalette:
    def __init__(self, numColors):
        np.random.seed(2)
        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [255, 230, 180],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],
                                  [255, 255, 0],
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],
                                  [0, 100, 100],
                                  [0, 255, 128],
                                  [0, 128, 255],
                                  [255, 0, 128],
                                  [128, 0, 255],
                                  [255, 128, 0],
                                  [128, 255, 0],
                                  ])

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.concatenate(
                [self.colorMap, np.random.randint(255, size=(numColors - self.colorMap.shape[0], 3))], axis=0)
            pass

        return

    def getColorMap(self):
        return self.colorMap

    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size=(3))
        else:
            return self.colorMap[index]
            pass


def loadClassMap(args):
    classMap = {}
    classLabelMap = {}
    with open(args.data_raw_path[:-10] + '/scannetv2-labels.combined.tsv') as info_file:
        line_index = 0
        for line in info_file:
            if line_index > 0:
                line = line.split('\t')

                key = line[1].strip()
                classMap[key] = line[7].strip()
                classMap[key + 's'] = line[7].strip()
                classMap[key + 'es'] = line[7].strip()
                classMap[key[:-1] + 'ves'] = line[7].strip()

                if line[4].strip() != '':
                    nyuLabel = int(line[4].strip())
                else:
                    nyuLabel = -1
                    pass
                classLabelMap[key] = [nyuLabel, line_index - 1]
                classLabelMap[key + 's'] = [nyuLabel, line_index - 1]
                classLabelMap[key[:-1] + 'ves'] = [nyuLabel, line_index - 1]
                pass
            line_index += 1
            continue
        pass
    return classMap, classLabelMap


def writePointCloudFace(filename, points, faces):
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0
element vertex """
        header += str(len(points))
        header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_index
end_header
"""
        f.write(header)
        for point in points:
            for value in point[:3]:
                f.write(str(value) + ' ')
                continue
            for value in point[3:]:
                f.write(str(int(value)) + ' ')
                continue
            f.write('\n')
            continue
        for face in faces:
            f.write('3 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')
            continue
        f.close()
        pass
    return


def mergePlanes(points, planes, planePointIndices, planeSegments, segmentNeighbors, numPlanes, debug=False):
    planeFittingErrors = []
    for plane, pointIndices in zip(planes, planePointIndices):
        XYZ = points[pointIndices]
        planeNorm = np.linalg.norm(plane)
        if planeNorm == 0:
            planeFittingErrors.append(fittingErrorThreshold * 2)
            continue
        diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / planeNorm
        planeFittingErrors.append(diff.mean())
        continue

    planeList = list(zip(planes, planePointIndices, planeSegments, planeFittingErrors))
    planeList = sorted(planeList, key=lambda x: x[3])

    while len(planeList) > 0:
        hasChange = False
        planeIndex = 0

        if debug:
            for index, planeInfo in enumerate(sorted(planeList, key=lambda x: -len(x[1]))):
                print(index, planeInfo[0] / np.linalg.norm(planeInfo[0]), planeInfo[2], planeInfo[3])
                continue
            pass

        while planeIndex < len(planeList):
            plane, pointIndices, segments, fittingError = planeList[planeIndex]
            if fittingError > fittingErrorThreshold:
                break
            neighborSegments = []
            for segment in segments:
                if segment in segmentNeighbors:
                    neighborSegments += segmentNeighbors[segment]
                    pass
                continue
            neighborSegments += list(segments)
            neighborSegments = set(neighborSegments)
            bestNeighborPlane = (fittingErrorThreshold, -1, None)
            for neighborPlaneIndex, neighborPlane in enumerate(planeList):
                if neighborPlaneIndex <= planeIndex:
                    continue
                if not bool(neighborSegments & neighborPlane[2]):
                    continue
                neighborPlaneNorm = np.linalg.norm(neighborPlane[0])
                if neighborPlaneNorm < 1e-4:
                    continue
                dotProduct = np.abs(
                    np.dot(neighborPlane[0], plane) / np.maximum(neighborPlaneNorm * np.linalg.norm(plane), 1e-4))
                if dotProduct < orthogonalThreshold:
                    continue
                newPointIndices = np.concatenate([neighborPlane[1], pointIndices], axis=0)
                XYZ = points[newPointIndices]
                if dotProduct > parallelThreshold and len(neighborPlane[1]) > len(pointIndices) * 0.5:
                    newPlane = fitPlane(XYZ)
                else:
                    newPlane = plane
                    pass
                diff = np.abs(np.matmul(XYZ, newPlane) - np.ones(XYZ.shape[0])) / np.linalg.norm(newPlane)
                newFittingError = diff.mean()
                if debug:
                    print(
                        len(planeList), planeIndex, neighborPlaneIndex, newFittingError, plane / np.linalg.norm(plane),
                                                                                         neighborPlane[
                                                                                             0] / np.linalg.norm(
                                                                                             neighborPlane[0]),
                        dotProduct, orthogonalThreshold)
                    pass
                if newFittingError < bestNeighborPlane[0]:
                    newPlaneInfo = [newPlane, newPointIndices, segments.union(neighborPlane[2]), newFittingError]
                    bestNeighborPlane = (newFittingError, neighborPlaneIndex, newPlaneInfo)
                    pass
                continue
            if bestNeighborPlane[1] != -1:
                newPlaneList = planeList[:planeIndex] + planeList[planeIndex + 1:bestNeighborPlane[1]] + planeList[
                                                                                                         bestNeighborPlane[
                                                                                                             1] + 1:]
                newFittingError, newPlaneIndex, newPlane = bestNeighborPlane
                for newPlaneIndex in range(len(newPlaneList)):
                    if (newPlaneIndex == 0 and newPlaneList[newPlaneIndex][3] > newFittingError) \
                            or newPlaneIndex == len(newPlaneList) - 1 \
                            or (newPlaneList[newPlaneIndex][3] < newFittingError and newPlaneList[newPlaneIndex + 1][
                        3] > newFittingError):
                        newPlaneList.insert(newPlaneIndex, newPlane)
                        break
                    continue
                if len(newPlaneList) == 0:
                    newPlaneList = [newPlane]
                    pass
                planeList = newPlaneList
                hasChange = True
            else:
                planeIndex += 1
                pass
            continue
        if not hasChange:
            break
        continue

    planeList = sorted(planeList, key=lambda x: -len(x[1]))

    minNumPlanes, maxNumPlanes = numPlanes
    if minNumPlanes == 1 and len(planeList) == 0:
        if debug:
            print('at least one plane')
            pass
    elif len(planeList) > maxNumPlanes:
        if debug:
            print('too many planes', len(planeList), maxNumPlanes)
            pass
        planeList = planeList[:maxNumPlanes] + [(np.zeros(3), planeInfo[1], planeInfo[2], fittingErrorThreshold) for
                                                planeInfo in planeList[maxNumPlanes:]]
        pass

    groupedPlanes, groupedPlanePointIndices, groupedPlaneSegments, groupedPlaneFittingErrors = zip(*planeList)
    return groupedPlanes, groupedPlanePointIndices, groupedPlaneSegments


def furthest_point_sampling(points, N=100):
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(points, metric='euclidean')
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)


def writePointFacePlane(filename, points, faces):
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0
element vertex """
        header += str(len(points))
        header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_index
end_header
"""
        f.write(header)
        for point in points:
            for value in point[:3]:
                f.write(str(value) + ' ')
                continue
            for value in point[3:]:
                f.write(str(int(value)) + ' ')
                continue
            f.write('\n')
            continue
        for face in faces:
            for value in face:
                f.write(str(value) + ' ')
                continue
            f.write('\n')
            continue
        f.close()
        pass
    return


def project2plane(plane, plane_points):
    A, B = plane_points[0], plane_points[1]
    AB = B - A
    N = plane / np.linalg.norm(plane, ord=2)
    U = AB / np.linalg.norm(AB, ord=2)
    V = np.cross(U, N)
    u = A + U
    v = A + V
    n = A + N
    S = [[A[0], u[0], v[0], n[0]], [A[1], u[1], v[1], n[1]], [A[2], u[2], v[2], n[2]], [1, 1, 1, 1]]
    D = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]]
    M = np.matmul(D, np.linalg.inv(S))
    return M


def points2contours(points, size=0.1):
    points = points / size
    points = points.astype(np.int)
    min_point = points.min(axis=0) - 5
    max_point = points.max(axis=0) + 5
    image_size = max_point - min_point + 1
    points = points - min_point
    image = np.zeros(image_size).astype(np.uint8)
    image[points[:, 0], points[:, 1]] = 255
    # plt.imshow(image, cmap='gray')
    # plt.show()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_valid = []
    for i, con in enumerate(contours):
        if con.shape[0] > 10:
            con = (con + min_point) * size
            contours_valid.append(con)
    return contours_valid


def generate_planes(args, scene_id, high_res=False, save_mesh=False, debug=False):
    if not os.path.exists(args.save_path + '/' + scene_id + '/annotation'):
        os.system('mkdir -p ' + args.save_path + '/' + scene_id + '/annotation')

    filename = args.data_raw_path + scene_id + '/' + scene_id + '.aggregation.json'
    data = json.load(open(filename, 'r'))
    aggregation = np.array(data['segGroups'])

    if high_res:
        filename = args.data_raw_path + scene_id + '/' + scene_id + '_vh_clean.labels.ply'
    else:
        filename = args.data_raw_path + scene_id + '/' + scene_id + '_vh_clean_2.labels.ply'

    plydata = PlyData.read(filename)
    vertices = plydata['vertex']
    points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
    faces = np.array(plydata['face']['vertex_indices'])

    # semanticSegmentation = vertices['label']

    if high_res:
        filename = args.data_raw_path + scene_id + '/' + scene_id + '_vh_clean.segs.json'
    else:
        filename = args.data_raw_path + scene_id + '/' + scene_id + '_vh_clean_2.0.010000.segs.json'

    data = json.load(open(filename, 'r'))
    segmentation = np.array(data['segIndices'])

    groupSegments = []
    groupLabels = []
    for segmentIndex in range(len(aggregation)):
        groupSegments.append(aggregation[segmentIndex]['segments'])
        groupLabels.append(aggregation[segmentIndex]['label'])

    segmentation = segmentation.astype(np.int32)

    uniqueSegments = np.unique(segmentation).tolist()
    numSegments = 0
    for segments in groupSegments:
        for segmentIndex in segments:
            if segmentIndex in uniqueSegments:
                uniqueSegments.remove(segmentIndex)
        numSegments += len(segments)

    for segment in uniqueSegments:
        groupSegments.append([segment, ])
        groupLabels.append('unannotated')

    segmentEdges = []
    for faceIndex in range(faces.shape[0]):
        face = faces[faceIndex]
        segment_1 = segmentation[face[0]]
        segment_2 = segmentation[face[1]]
        segment_3 = segmentation[face[2]]
        if segment_1 != segment_2 or segment_1 != segment_3:
            if segment_1 != segment_2 and segment_1 != -1 and segment_2 != -1:
                segmentEdges.append((min(segment_1, segment_2), max(segment_1, segment_2)))
            if segment_1 != segment_3 and segment_1 != -1 and segment_3 != -1:
                segmentEdges.append((min(segment_1, segment_3), max(segment_1, segment_3)))
            if segment_2 != segment_3 and segment_2 != -1 and segment_3 != -1:
                segmentEdges.append((min(segment_2, segment_3), max(segment_2, segment_3)))
    segmentEdges = list(set(segmentEdges))

    labelNumPlanes = {'wall': [1, 3],
                      'floor': [1, 1],
                      'cabinet': [0, 5],
                      'bed': [0, 5],
                      'chair': [0, 5],
                      'sofa': [0, 10],
                      'table': [0, 5],
                      'door': [1, 2],
                      'window': [0, 2],
                      'bookshelf': [0, 5],
                      'picture': [1, 1],
                      'counter': [0, 10],
                      'blinds': [0, 0],
                      'desk': [0, 10],
                      'shelf': [0, 5],
                      'shelves': [0, 5],
                      'curtain': [0, 0],
                      'dresser': [0, 5],
                      'pillow': [0, 0],
                      'mirror': [0, 0],
                      'entrance': [1, 1],
                      'floor mat': [1, 1],
                      'clothes': [0, 0],
                      'ceiling': [0, 5],
                      'book': [0, 1],
                      'books': [0, 1],
                      'refridgerator': [0, 5],
                      'television': [1, 1],
                      'paper': [0, 1],
                      'towel': [0, 1],
                      'shower curtain': [0, 1],
                      'box': [0, 5],
                      'whiteboard': [1, 5],
                      'person': [0, 0],
                      'night stand': [1, 5],
                      'toilet': [0, 5],
                      'sink': [0, 5],
                      'lamp': [0, 1],
                      'bathtub': [0, 5],
                      'bag': [0, 1],
                      'otherprop': [0, 5],
                      'otherstructure': [0, 5],
                      'otherfurniture': [0, 5],
                      'unannotated': [0, 5],
                      '': [0, 0],
                      }
    nonPlanarGroupLabels = ['bicycle', 'bottle', 'water bottle']
    nonPlanarGroupLabels = {label: True for label in nonPlanarGroupLabels}

    # verticalLabels = ['wall', 'door', 'cabinet']
    classMap, classLabelMap = loadClassMap(args)
    classMap['unannotated'] = 'unannotated'
    classLabelMap['unannotated'] = [max([index for index, label in classLabelMap.values()]) + 1, 41]
    allXYZ = points.reshape(-1, 3)

    segmentNeighbors = {}
    for segmentEdge in segmentEdges:
        if segmentEdge[0] not in segmentNeighbors:
            segmentNeighbors[segmentEdge[0]] = []
        segmentNeighbors[segmentEdge[0]].append(segmentEdge[1])

        if segmentEdge[1] not in segmentNeighbors:
            segmentNeighbors[segmentEdge[1]] = []
        segmentNeighbors[segmentEdge[1]].append(segmentEdge[0])

    planeGroups = []
    print('num groups', len(groupSegments))

    debugIndex = -1

    for groupIndex, group in enumerate(groupSegments):
        if debugIndex != -1 and groupIndex != debugIndex:
            continue
        if groupLabels[groupIndex] in nonPlanarGroupLabels:
            groupLabel = groupLabels[groupIndex]
            minNumPlanes, maxNumPlanes = 0, 0
        elif groupLabels[groupIndex] in classMap:
            groupLabel = classMap[groupLabels[groupIndex]]
            minNumPlanes, maxNumPlanes = labelNumPlanes[groupLabel]
        else:
            minNumPlanes, maxNumPlanes = 0, 0
            groupLabel = ''

        if maxNumPlanes == 0:
            pointMasks = []
            for segmentIndex in group:
                pointMasks.append(segmentation == segmentIndex)
            pointIndices = np.any(np.stack(pointMasks, 0), 0).nonzero()[0]
            groupPlanes = [[np.zeros(3), pointIndices, []]]
            planeGroups.append(groupPlanes)
            continue
        groupPlanes = []
        groupPlanePointIndices = []
        groupPlaneSegments = []
        for segmentIndex in group:
            segmentMask = segmentation == segmentIndex
            allSegmentIndices = segmentMask.nonzero()[0]
            segmentIndices = allSegmentIndices.copy()

            XYZ = allXYZ[segmentMask.reshape(-1)]
            numPoints = XYZ.shape[0]

            for c in range(2):
                if c == 0:
                    ## First try to fit one plane
                    plane = fitPlane(XYZ)
                    diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
                    if diff.mean() < fittingErrorThreshold:
                        groupPlanes.append(plane)
                        groupPlanePointIndices.append(segmentIndices)
                        groupPlaneSegments.append(set([segmentIndex]))
                        break
                else:
                    ## Run ransac
                    segmentPlanes = []
                    segmentPlanePointIndices = []

                    for planeIndex in range(numPlanesPerSegment):
                        if len(XYZ) < planeAreaThreshold:
                            continue
                        bestPlaneInfo = [None, 0, None]
                        for iteration in range(min(XYZ.shape[0], numIterations)):
                            sampledPoints = XYZ[np.random.choice(np.arange(XYZ.shape[0]), size=(3), replace=False)]
                            try:
                                plane = fitPlane(sampledPoints)
                            except:
                                continue
                            diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
                            inlierMask = diff < planeDiffThreshold
                            numInliers = inlierMask.sum()
                            if numInliers > bestPlaneInfo[1]:
                                bestPlaneInfo = [plane, numInliers, inlierMask]

                        if bestPlaneInfo[1] < planeAreaThreshold:
                            break

                        pointIndices = segmentIndices[bestPlaneInfo[2]]
                        bestPlane = fitPlane(XYZ[bestPlaneInfo[2]])

                        segmentPlanes.append(bestPlane)
                        segmentPlanePointIndices.append(pointIndices)

                        outlierMask = np.logical_not(bestPlaneInfo[2])
                        segmentIndices = segmentIndices[outlierMask]
                        XYZ = XYZ[outlierMask]

                    if sum([len(indices) for indices in segmentPlanePointIndices]) < numPoints * 0.5:
                        groupPlanes.append(np.zeros(3))
                        groupPlanePointIndices.append(allSegmentIndices)
                        groupPlaneSegments.append(set([segmentIndex]))
                    else:
                        if len(segmentIndices) > 0:
                            ## Add remaining non-planar regions
                            segmentPlanes.append(np.zeros(3))
                            segmentPlanePointIndices.append(segmentIndices)
                        groupPlanes += segmentPlanes
                        groupPlanePointIndices += segmentPlanePointIndices

                        for _ in range(len(segmentPlanes)):
                            groupPlaneSegments.append(set([segmentIndex]))

        numRealPlanes = len([plane for plane in groupPlanes if np.linalg.norm(plane) > 1e-4])
        if minNumPlanes == 1 and numRealPlanes == 0:
            ## Some instances always contain at least one planes (e.g, the floor)
            maxArea = (planeAreaThreshold, -1)
            for index, indices in enumerate(groupPlanePointIndices):
                if len(indices) > maxArea[0]:
                    maxArea = (len(indices), index)
            maxArea, planeIndex = maxArea
            if planeIndex >= 0:
                groupPlanes[planeIndex] = fitPlane(allXYZ[groupPlanePointIndices[planeIndex]])
                numRealPlanes = 1
        if minNumPlanes == 1 and maxNumPlanes == 1 and numRealPlanes > 1:
            ## Some instances always contain at most one planes (e.g, the floor)

            pointIndices = np.concatenate(
                [indices for plane, indices in list(zip(groupPlanes, groupPlanePointIndices))],
                axis=0)
            XYZ = allXYZ[pointIndices]
            plane = fitPlane(XYZ)
            diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)

            if groupLabel == 'floor':
                ## Relax the constraint for the floor due to the misalignment issue in ScanNet
                fittingErrorScale = 3
            else:
                fittingErrorScale = 1

            if diff.mean() < fittingErrorThreshold * fittingErrorScale:
                groupPlanes = [plane]
                groupPlanePointIndices = [pointIndices]
                planeSegments = []
                for segments in groupPlaneSegments:
                    planeSegments += list(segments)
                groupPlaneSegments = [set(planeSegments)]
                numRealPlanes = 1

        if numRealPlanes > 1:
            groupPlanes, groupPlanePointIndices, groupPlaneSegments = mergePlanes(points, groupPlanes,
                                                                                  groupPlanePointIndices,
                                                                                  groupPlaneSegments, segmentNeighbors,
                                                                                  numPlanes=(
                                                                                      minNumPlanes, maxNumPlanes),
                                                                                  debug=debugIndex != -1)

        groupNeighbors = []
        for planeIndex, planeSegments in enumerate(groupPlaneSegments):
            neighborSegments = []
            for segment in planeSegments:
                if segment in segmentNeighbors:
                    neighborSegments += segmentNeighbors[segment]
            neighborSegments += list(planeSegments)
            neighborSegments = set(neighborSegments)
            neighborPlaneIndices = []
            for neighborPlaneIndex, neighborPlaneSegments in enumerate(groupPlaneSegments):
                if neighborPlaneIndex == planeIndex:
                    continue
                if bool(neighborSegments & neighborPlaneSegments):
                    plane = groupPlanes[planeIndex]
                    neighborPlane = groupPlanes[neighborPlaneIndex]
                    if np.linalg.norm(plane) * np.linalg.norm(neighborPlane) < 1e-4:
                        continue
                    dotProduct = np.abs(
                        np.dot(plane, neighborPlane) / np.maximum(np.linalg.norm(plane) * np.linalg.norm(neighborPlane),
                                                                  1e-4))
                    if dotProduct < orthogonalThreshold:
                        neighborPlaneIndices.append(neighborPlaneIndex)
            groupNeighbors.append(neighborPlaneIndices)
        groupPlanes = list(zip(groupPlanes, groupPlanePointIndices, groupNeighbors))
        # groupPlanes = zip(groupPlanes, groupPlanePointIndices, groupNeighbors)
        planeGroups.append(groupPlanes)

    if debug:
        colorMap = ColorPalette(segmentation.max() + 2).getColorMap()
        colorMap[-1] = 0
        colorMap[-2] = 255
        annotationFolder = 'test/'
    else:
        numPlanes = sum([len(group) for group in planeGroups])
        segmentationColor = (np.arange(numPlanes + 1) + 1) * 100
        colorMap = np.stack([segmentationColor / (256 * 256), segmentationColor / 256 % 256, segmentationColor % 256],
                            axis=1)
        colorMap[-1] = 0
        annotationFolder = args.save_path + scene_id + '/annotation'

    if debug:
        colors = colorMap[segmentation]
        writePointCloudFace(annotationFolder + '/segments.ply', np.concatenate([points, colors], axis=-1), faces)

        groupedSegmentation = np.full(segmentation.shape, fill_value=-1)
        for segmentIndex in range(len(aggregation)):
            indices = aggregation[segmentIndex]['segments']
            for index in indices:
                groupedSegmentation[segmentation == index] = segmentIndex
        groupedSegmentation = groupedSegmentation.astype(np.int32)
        colors = colorMap[groupedSegmentation]
        writePointCloudFace(annotationFolder + '/groups.ply', np.concatenate([points, colors], axis=-1), faces)

    planes = []
    planePointIndices = []
    planeInfo = []
    structureIndex = 0
    for index, group in enumerate(planeGroups):
        groupPlanes, groupPlanePointIndices, groupNeighbors = zip(*group)

        diag = np.diag(np.ones(len(groupNeighbors)))
        adjacencyMatrix = diag.copy()
        for groupIndex, neighbors in enumerate(groupNeighbors):
            for neighbor in neighbors:
                adjacencyMatrix[groupIndex][neighbor] = 1
        if groupLabels[index] in classLabelMap:
            label = classLabelMap[groupLabels[index]]
        else:
            print('label not valid', groupLabels[index])
            exit(1)
            label = -1
        groupInfo = [[(index, label[0], label[1])] for _ in range(len(groupPlanes))]
        groupPlaneIndices = (adjacencyMatrix.sum(-1) >= 2).nonzero()[0]
        usedMask = {}
        for groupPlaneIndex in groupPlaneIndices:
            if groupPlaneIndex in usedMask:
                continue
            groupStructure = adjacencyMatrix[groupPlaneIndex].copy()
            for neighbor in groupStructure.nonzero()[0]:
                if np.any(adjacencyMatrix[neighbor] < groupStructure):
                    groupStructure[neighbor] = 0
            groupStructure = groupStructure.nonzero()[0]

            if len(groupStructure) < 2:
                print('invalid structure')
                print(groupPlaneIndex, groupPlaneIndices)
                print(groupNeighbors)
                print(groupPlaneIndex)
                print(adjacencyMatrix.sum(-1) >= 2)
                print((adjacencyMatrix.sum(-1) >= 2).nonzero()[0])
                print(adjacencyMatrix[groupPlaneIndex])
                print(adjacencyMatrix)
                print(groupStructure)
                exit(1)
            if len(groupStructure) >= 4:
                print('complex structure')
                print('group index', index)
                print(adjacencyMatrix)
                print(groupStructure)
                groupStructure = groupStructure[:3]
            if len(groupStructure) in [2, 3]:
                for planeIndex in groupStructure:
                    groupInfo[planeIndex].append((structureIndex, len(groupStructure)))
                structureIndex += 1
            for planeIndex in groupStructure:
                usedMask[planeIndex] = True
        planes += groupPlanes
        planePointIndices += groupPlanePointIndices
        planeInfo += groupInfo

    planeSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
    for planeIndex, planePoints in enumerate(planePointIndices):
        planeSegmentation[planePoints] = planeIndex

    # generate planar
    if save_mesh:
        import copy
        color_vis = random_color()
        points_plane = []
        points_tensor = torch.Tensor(points).cuda()
        faces_copy = copy.deepcopy(faces)
        faces_copy = torch.Tensor(np.stack(faces_copy)).cuda().int()
        planes_tensor = torch.zeros_like(points_tensor)
        indices_tensor = torch.zeros_like(points_tensor[:, 0]).int()
        for i in range(len(planes)):
            planes_tensor[planePointIndices[i]] = torch.Tensor(planes[i]).cuda()
            indices_tensor[planePointIndices[i]] = i
        
        valid = (planes_tensor != 0).any(-1)
        invalid_ind = torch.nonzero(valid == 0, as_tuple=False).squeeze(1)
        planes_tensor_valid = planes_tensor[valid]
        points_tensor_valid = points_tensor[valid]
        t = ((points_tensor_valid.unsqueeze(1) @ planes_tensor_valid.unsqueeze(-1)).squeeze() - 1) / (planes_tensor_valid[:, 0] ** 2 + planes_tensor_valid[:, 1] ** 2 + planes_tensor_valid[:, 2] ** 2)
        plane_points = points_tensor_valid - planes_tensor_valid[:, :3] * t.unsqueeze(-1)
        points_tensor[valid] = plane_points
        points_tensor[invalid_ind] = plane_points[0]
        
        n = 100
        part_num = faces_copy.shape[0] // n
        match_list = []
        for i in range(n):
            if i == n-1:
                faces_part = faces_copy[i * part_num:]
            else:
                faces_part = faces_copy[i * part_num: (i + 1) * part_num]
            match = (faces_part.unsqueeze(0) != invalid_ind.unsqueeze(-1).unsqueeze(-1))
            match = match.all(-1)
            match = match.all(0)
            match_list.append(match)
        match = torch.cat(match_list)
        faces_copy = faces_copy[match]
        points_plane = points_tensor.data.cpu().numpy()

        n_ins = indices_tensor.data.cpu().numpy().max() + 1
        indices_tensor = indices_tensor.data.cpu().numpy()
        segmentationColor = (np.arange(n_ins + 1) + 1) * 100
        colorMap = np.stack([segmentationColor / (256 * 256), segmentationColor / 256 % 256, segmentationColor % 256],
                            axis=1)
        colorMap[-1] = 0
        plane_colors = colorMap[indices_tensor]

        # for vis
        colorMap_vis = color_vis(n_ins)
        plane_colors_vis = colorMap_vis[indices_tensor]

        writePointCloudFace(annotationFolder + '/planes_mesh.ply',
                            np.concatenate([points_plane, plane_colors], axis=-1), faces_copy.data.cpu().numpy())

        writePointCloudFace(annotationFolder + '/planes_mesh_vis.ply',
                            np.concatenate([points_plane, plane_colors_vis], axis=-1), faces_copy.data.cpu().numpy())

    if debug:
        groupSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
        structureSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
        typeSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
        for planeIndex, planePoints in enumerate(planePointIndices):
            if len(planeInfo[planeIndex]) > 1:
                structureSegmentation[planePoints] = planeInfo[planeIndex][1][0]
                typeSegmentation[planePoints] = np.maximum(typeSegmentation[planePoints],
                                                           planeInfo[planeIndex][1][1] - 2)
            groupSegmentation[planePoints] = planeInfo[planeIndex][0][0]

        colors = colorMap[groupSegmentation]
        writePointCloudFace(annotationFolder + '/group.ply', np.concatenate([points, colors], axis=-1), faces)

        colors = colorMap[structureSegmentation]
        writePointCloudFace(annotationFolder + '/structure.ply', np.concatenate([points, colors], axis=-1), faces)

        colors = colorMap[typeSegmentation]
        writePointCloudFace(annotationFolder + '/type.ply', np.concatenate([points, colors], axis=-1), faces)

    planes = np.array(planes)
    print('number of planes: ', planes.shape[0])
    # planesD = 1.0 / np.maximum(np.linalg.norm(planes, axis=-1, keepdims=True), 1e-4)
    # planes *= pow(planesD, 2)

    if debug:
        print(len(planes), len(planeInfo))
        exit(1)

    plane_points = []
    for i in range(len(planePointIndices)):
        plane_points.append(points[planePointIndices[i]])

    planes_valid = []
    planeInfo_valid = []
    plane_points_valid = []
    for i in range(len(plane_points)):
        if (planes[i] != 0).any():
            planes_valid.append(planes[i])
            planeInfo_valid.append(planeInfo[i])
            plane_points_valid.append(plane_points[i])

    planes_valid = np.stack(planes_valid)
    np.save(annotationFolder + '/planes.npy', planes_valid)
    np.save(annotationFolder + '/plane_info.npy', planeInfo_valid)
    np.save(annotationFolder + '/plane_points', plane_points_valid)

    return planes_valid, plane_points_valid


if __name__ == '__main__':

    scene_ids = os.listdir(ROOT_FOLDER)
    scene_ids = sorted(scene_ids)

    for index, scene_id in enumerate(scene_ids):
        if scene_id[:5] != 'scene':
            continue

        if not os.path.exists(args.save_path + '/' + scene_id + '/annotation'):
            os.system('mkdir -p ' + args.save_path + '/' + scene_id + '/annotation')

        if not os.path.exists(args.save_path + '/' + scene_id + '/annotation/planes.ply'):
            print('plane fitting', scene_id)
            readMesh(scene_id)
