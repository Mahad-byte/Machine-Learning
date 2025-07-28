import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import seaborn as sns

def getProxyImg(img):
    proxy_img = np.zeros(shape=(646, 469), dtype=np.uint8)
    land = np.array([184, 236, 168])
    water = np.array([105, 169, 255])
    unknown = np.array([255, 255, 255])

    is_land = (img == land).all(axis=2)
    is_water = (img == water).all(axis=2)
    is_unknown = (img == unknown).all(axis=2)

    proxy_img[is_land] = 1
    proxy_img[is_water] = 2
    proxy_img[is_unknown] = 0
    # count = (proxy_img == 0).sum()
    return proxy_img


def knn(img, neighbors, dist_type):
    proxy_img = getProxyImg(img)
    KNN = KNeighborsClassifier(
        n_neighbors=neighbors,
        metric=dist_type
    )
    known_pixels = np.argwhere(proxy_img != 0)
    known_labels = proxy_img[proxy_img != 0]
    KNN.fit(known_pixels, known_labels)

    unknown_pixels = np.argwhere(proxy_img == 0)
    predicted_labels = KNN.predict(unknown_pixels)
    # Fill predictions into output array
    output_arrayyy = np.copy(proxy_img)
    output_arrayyy[proxy_img == 0] = predicted_labels

    return output_arrayyy


def showConfusionMatrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Accent')
    plt.show()



if __name__ == '__main__':
    img = skio.imread('Italy10.png')
    img2 = skio.imread('Italy20.png')
    img3 = skio.imread('Italy30.png')
    img4 = skio.imread('Italy40.png')
    img5 = skio.imread('Italy50.png')
    img_true = skio.imread('Italy.png')

    output_array = knn(img, 1, 'manhattan')
    output_array2 = knn(img2, 1, 'manhattan')
    output_array3 = knn(img3, 1, 'manhattan')
    output_array4 = knn(img4, 1, 'manhattan')
    output_array5 = knn(img5, 1, 'manhattan')

    plt.figure(figsize=(8, 8))
    plt.suptitle("Resuts for K = 1 and Manhattan")
    plt.subplot(2, 5, 1)
    plt.title("Original Image 1")
    plt.imshow(img, cmap='jet')

    plt.subplot(2, 5, 2)
    plt.title("KNN Classified 1")
    plt.imshow(output_array, cmap='jet')

    plt.subplot(2, 5, 3)
    plt.title("Original Image 2")
    plt.imshow(img2, cmap='jet')

    plt.subplot(2, 5, 4)
    plt.title("KNN Classified 2")
    plt.imshow(output_array2, cmap='jet')
    plt.colorbar()
    # plt.show()

    plt.subplot(2, 5, 5)
    plt.title("Original Image 3")
    plt.imshow(img3, cmap='jet')

    plt.subplot(2, 5, 6)
    plt.title("KNN Classified 3")
    plt.imshow(output_array3, cmap='jet')
    plt.colorbar()
    # plt.show()

    plt.subplot(2, 5, 7)
    plt.title("Original Image 4")
    plt.imshow(img4, cmap='jet')

    plt.subplot(2, 5, 8)
    plt.title("KNN Classified 4")
    plt.imshow(output_array4, cmap='jet')
    plt.colorbar()
    # plt.show()

    plt.subplot(2, 5, 9)
    plt.title("Original Image 5")
    plt.imshow(img5, cmap='jet')

    plt.subplot(2, 5, 10)
    plt.title("KNN Classified 5")
    plt.imshow(output_array5, cmap='jet')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    proxy_img = getProxyImg(img_true)
    y_true = proxy_img.flatten()
    y_pred = output_array.flatten()
    y_pred2 = output_array2.flatten()
    y_pred3 = output_array3.flatten()
    y_pred4 = output_array4.flatten()
    y_pred5 = output_array5.flatten()

    print("Unique in y_true:", np.unique(y_true))
    print("Unique in y_pred:", np.unique(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    cm2 = confusion_matrix(y_true, y_pred2, labels=[0, 1, 2])
    cm3 = confusion_matrix(y_true, y_pred3, labels=[0, 1, 2])
    cm4 = confusion_matrix(y_true, y_pred4, labels=[0, 1, 2])
    cm5 = confusion_matrix(y_true, y_pred5, labels=[0, 1, 2])

    # print(cm)
    plt.title('Confusion Matrix 1')
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Pred 0', 'Pred 1', 'Pred 2'],
               yticklabels=['True 0', 'True 1', 'True 2'])
    plt.tight_layout()
    plt.show()
    plt.title('Confusion Matrix 2')
    sns.heatmap(cm2, annot=True, fmt='d', xticklabels=['Pred 0', 'Pred 1', 'Pred 2'],
                yticklabels=['True 0', 'True 1', 'True 2'])
    plt.tight_layout()
    plt.show()
    plt.title('Confusion Matrix 3')
    sns.heatmap(cm3, annot=True, fmt='d', xticklabels=['Pred 0', 'Pred 1', 'Pred 2'],
                yticklabels=['True 0', 'True 1', 'True 2'])
    plt.tight_layout()
    plt.show()
    plt.title('Confusion Matrix 4')
    sns.heatmap(cm4, annot=True, fmt='d', xticklabels=['Pred 0', 'Pred 1', 'Pred 2'],
                yticklabels=['True 0', 'True 1', 'True 2'])
    plt.tight_layout()
    plt.show()
    plt.title('Confusion Matrix 5')
    sns.heatmap(cm5, annot=True, fmt='d', xticklabels=['Pred 0', 'Pred 1', 'Pred 2'],
                yticklabels=['True 0', 'True 1', 'True 2'])
    plt.tight_layout()
    plt.show()

    intersection = np.logical_and(output_array, proxy_img).sum()
    union = np.logical_or(output_array, proxy_img).sum()
    jaccard = float(intersection) / float(union)
    print(f" Jaccard for 10% Img: {jaccard}")
    intersection2 = np.logical_and(output_array2, proxy_img).sum()
    union2 = np.logical_or(output_array2, proxy_img).sum()
    jaccard2 = float(intersection2) / float(union2)
    print(f" Jaccard for 20% Img: {jaccard2}")
    intersection3 = np.logical_and(output_array3, proxy_img).sum()
    union3 = np.logical_or(output_array3, proxy_img).sum()
    jaccard3 = float(intersection3) / float(union3)
    print(f" Jaccard for 30% Img: {jaccard3}")
    intersection4 = np.logical_and(output_array4, proxy_img).sum()
    union4 = np.logical_or(output_array4, proxy_img).sum()
    jaccard4 = float(intersection4) / float(union4)
    print(f" Jaccard for 40% Img: {jaccard4}")
    intersection5 = np.logical_and(output_array5, proxy_img).sum()
    union5 = np.logical_or(output_array5, proxy_img).sum()
    jaccard5 = float(intersection5) / float(union5)
    print(f" Jaccard for 50% Img: {jaccard5}")

    acc = accuracy_score(y_true, y_pred)
    print(f" Accuracy for 10% Img: {acc}")
    prec = precision_score(y_true, y_pred, average=None)
    print(f" Precision for 10% Img: {prec}")
    recall = recall_score(y_true, y_pred, average=None)
    print(f" Recall for 10% Img: {recall}")
    f1 = f1_score(y_true, y_pred, average=None)
    print(f" f1 for 10% Img: {f1}")

    acc2 = accuracy_score(y_true, y_pred2)
    print(f" Accuracy for 20% Img: {acc2}")
    prec2 = precision_score(y_true, y_pred2, average=None)
    print(f" Precision for 20% Img: {prec2}")
    recall2 = recall_score(y_true, y_pred2, average=None)
    print(f" Recall for 20% Img: {recall2}")
    f12 = f1_score(y_true, y_pred2, average=None)
    print(f" f1 for 20% Img: {f12}")

    acc3 = accuracy_score(y_true, y_pred3)
    print(f" Accuracy for 30% Img: {acc3}")
    prec3 = precision_score(y_true, y_pred3, average=None)
    print(f" Precision for 30% Img: {prec3}")
    recall3 = recall_score(y_true, y_pred3, average=None)
    print(f" Recall for 30% Img: {recall3}")
    f13 = f1_score(y_true, y_pred3, average=None)
    print(f" f1 for 30% Img: {f13}")

    acc4 = accuracy_score(y_true, y_pred4)
    print(f" Accuracy for 40% Img: {acc4}")
    prec4 = precision_score(y_true, y_pred4, average=None)
    print(f" Precision for 40% Img: {prec4}")
    recall4 = recall_score(y_true, y_pred4, average=None)
    print(f" Recall for 40% Img: {recall4}")
    f14 = f1_score(y_true, y_pred4, average=None)
    print(f" f1 for 40% Img: {f14}")

    acc5 = accuracy_score(y_true, y_pred5)
    print(f" Accuracy for 50% Img: {acc5}")
    prec5 = precision_score(y_true, y_pred5, average=None)
    print(f" Precision for 50% Img: {prec5}")
    recall5 = recall_score(y_true, y_pred5, average=None)
    print(f" Recall for 50% Img: {recall5}")
    f15 = f1_score(y_true, y_pred5, average=None)
    print(f" f1 for 50% Img: {f15}")



# def get_limited_unknowns(proxy_im, max_pixels=10000):
#     unknown_cords = np.argwhere(proxy_im == 0)
#     if len(unknown_cords) > max_pixels:
#         step = len(unknown_cords) // max_pixels
#         return unknown_cords[::step]
#     return unknown_cords
#
# limited_unknowns = get_limited_unknowns(proxy_img)
#
# known_pixels = {(i,j): proxy_img[i,j]
#                for i,j in np.argwhere(proxy_img != 0)}
#
# def manhattan_distance(x1, x2):
#     return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])
#
#
# def knn(n, l, known_pixel, dist_type, k=5):  # Added missing parameters
#     distances = []
#
#     for (x1, y), value in known_pixel.items():
#         if dist_type == 'manhattan':
#             dist = manhattan_distance((n, l), (x1, y))
#             distances.append((dist, value))
#
#     # Sort by distance and get k nearest neighbors
#     distances.sort()
#     neighbors = distances[:k]
#
#     # Count votes
#     votes = {1: 0, 2: 0}
#     for _, value in neighbors:
#         votes[value] += 1
#
#     return max(votes.items(), key=lambda x: x[1])[0]
#
#
# output_array = np.copy(proxy_img)
#
# # Classify unknown pixels
# # for i in range(proxy_img.shape[0]):
# #     for j in range(proxy_img.shape[1]):
# #         if proxy_img[i, j] == 0:
# #             output_array[i, j] = knn(i, j, known_pixels, 'manhattan', k=5)  # Fixed call
#
# for (i,j) in limited_unknowns:
#     output_array[i,j] = knn(i, j, known_pixels, 'manhattan', k=5)

# unknown_coords = np.argwhere(proxy_img == 0)
# known_pixels = {(i,j): proxy_img[i,j] for i,j in np.argwhere(proxy_img != 0)}

# def manhattan_distance(x1, x2):
#     return abs(x1[0]-x2[0]) + abs(x1[1]-x2[1])
#
# def knn(i, j, known_pixel, k=5):
#     distances = []
#     for (x, y), value in known_pixel.items():
#         dist = manhattan_distance((i, j), (x, y))
#         distances.append((dist, value))
#     distances.sort()
#     neighbors = distances[:k]
#     votes = {1: 0, 2: 0}
#     for _, value in neighbors:
#         votes[value] += 1
#     return max(votes.items(), key=lambda x: x[1])[0]
#
# # Initialize output
# output_array = np.copy(proxy_img)
#
# # OPTIMIZATION: Process unknown pixels in batches with progress feedback
# batch_size = 1000
# for batch_start in range(0, len(unknown_coords), batch_size):
#     batch = unknown_coords[batch_start:batch_start + batch_size]
#     for (i,j) in batch:
#         output_array[i,j] = knn(i, j, known_pixels)
#     print(f"Processed {min(batch_start + batch_size, len(unknown_coords))}/{len(unknown_coords)} pixels")
#
# # Fill remaining unknowns by nearest classified pixel
# from scipy.ndimage import distance_transform_edt
# unknown_mask = (output_array == 0)
# if unknown_mask.any():
#     nearest_labels = distance_transform_edt(unknown_mask, return_indices=True)[1]
#     output_array[unknown_mask] = output_array[tuple(nearest_labels)]
#

# Visualization
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Original Proxy Image")
# plt.imshow(proxy_img, cmap='jet')
#
# plt.subplot(1, 2, 2)
# plt.title("Classified Output (KNN)")
# plt.imshow(output_array, cmap='jet')
# plt.colorbar()
# plt.show()
