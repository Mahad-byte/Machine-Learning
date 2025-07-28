# import csv
# import math
#
# import numpy as np
# from PIL import Image
# import os
# import matplotlib.pyplot as plt
# from tf_keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
# from tf_keras.models import Sequential, load_model
# from  tf_keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from sklearn.model_selection import train_test_split
# import  pandas as pd
# from tf_keras.src.regularizers import l2

# Dataset from https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset


# # --- Code to read images using pillow, making a tuple of file paths and labels (cat/dog) and saving it to file classify_data ---
# train_list = []
#
# directory = 'PetImages/Cat'
# directory2 = 'PetImages/Dog'
#
# # 1. Fetching Images from Directory
# for filename in os.listdir(directory):
#     filepath = os.path.join(directory, filename)
#     with Image.open(filepath) as img:
#         train_list.append((filepath, "cat"))
#
#
# print("total images with Cat: ", len(train_list))
#
# for filename in os.listdir(directory2):
#     filepath = os.path.join(directory2, filename)
#     with Image.open(filepath) as img:
#         train_list.append((filepath, "dog"))
#
#
# print("total images with Cat and dog: ", len(train_list))
# # 2. Separating Filepaths and labels
# file_paths, labels = zip(*train_list)
#
# df = pd.DataFrame({"filepath": file_paths, "labels": labels})
# df.to_csv("classify_data.csv", index=False)

# file_path, label = train_list[0]
# img = Image.open(file_path)
# img.show()

# # 3. Writing it in csv
# df = pd.read_csv('classify_data.csv')
# filepath = df['filepath']
# labels = df['labels']

# print(filepath.head())
# print(labels.head())
#
# # 4. 80/20 Ratio for training and testing
# x_train, x_test, y_train, y_test = train_test_split(filepath, labels, test_size=0.2, random_state=42)
#
# # 5. Create generators
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest',
#     validation_split=0.2
# )
# train_generator = train_datagen.flow_from_dataframe(
#     dataframe=pd.DataFrame({"filename": x_train, "class": y_train}),
#     x_col="filename",
#     y_col="class",
#     target_size=(100, 100), Resizing to 100 x 100
#     batch_size=32,
#     class_mode="binary"
# )
#
# val_generator = train_datagen.flow_from_dataframe(
#     dataframe=pd.DataFrame({"filename": x_test, "class": y_test}),
#     x_col="filename",
#     y_col="class",
#     target_size=(100, 100),
#     batch_size=32,
#     class_mode="binary"
# )
# # 6. Train  CNN model
# model = Sequential(
#     [
#         Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
#         MaxPool2D(2, 2),
#         # Dropout(0.2),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPool2D(2, 2),
#         # Dropout(0.2),
#         Conv2D(128, (3, 3), activation='relu'),
#         MaxPool2D(2, 2),
#         Flatten(),
#         Dense(512, activation='relu', name='layer1'),
#         # Dropout(0.2),
#         Dense(1, activation='sigmoid', name='layer2'),
#     ]
# )
#
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# trained_model = model.fit(
#     train_generator,
#     epochs=10,
#     validation_data=val_generator
# )
# print(model.summary())
#
# loss, accuracy = model.evaluate(val_generator)
# print("Accuracy: ", accuracy) # 0.83
# # 7. Save Model
# model.save("classifier.keras")

# Load Model
# classifier = load_model("classifier.keras")

# predictions = []
# images_array = []
# fileeee = []
# # 8. Fetching first and last 500 Images path from the saved csv file, loading and converting it into np array,
# # if ran on all 25000, takes too much time :/
# for img_name in filepath[:500]:
#     fileeee.append(img_name)
#     imgs = load_img(img_name, target_size=(100, 100))
#     imgs_arr = img_to_array(imgs)
#     np_img = np.expand_dims(imgs_arr, axis=0)
#     images_array.append(np_img)
#
# for img_name in filepath[-500:]:
#     fileeee.append(img_name)
#     imgs = load_img(img_name, target_size=(100, 100))
#     imgs_arr = img_to_array(imgs)
#     np_img = np.expand_dims(imgs_arr, axis=0)
#     images_array.append(np_img)
#
# # Check this line
# # dataframe = pd.DataFrame({"filepath": fileeee, "Pridictions": images_array})
# # dataframe.to_csv("classify_data.csv", index=False)
#
# print("Length of images array: ", len(images_array))
# # 9. Prediction the first and last 500 images
# for img in images_array:
#     pred = classifier.predict(img)[0][0]
#     predictions.append(pred)
#


# 10. Fetching the predictions plus path into csv
# new_df = pd.read_csv('predictions_result.csv')
# file_path_pred = new_df['filepaths']
# list_of_pred_cat = new_df['Predictions'][:500]
# list_of_pred_dog = new_df['Predictions'][-500:]

# print(list_of_pred_cat)
# print("\nblahhhhhh\n")
# print(list_of_pred_dog)


# def calc_distance(x: float, y: float):
#     return math.sqrt((x-y)**2)
#
#
# # 11. Giving a new picture to predict
# imgs = load_img("dog_predict_3.jpg", target_size=(100, 100))
# imgs_arr = img_to_array(imgs)
# np_img = np.expand_dims(imgs_arr, axis=0)
#
# result_ = classifier.predict(np_img)[0][0]
# result = result_
# print("Result: ", result, type(result))
#
# distance_cat = []
# distance_dog = []

# 12. Calculating Distances, also adding filepaths to keep track when sorting, and displaying first 5 images from sorted list.
# if result > 0.5:
#     for _, row in new_df.iloc[-500:].iterrows():  # Last 500 rows (dogs)
#         distance = calc_distance(row['Predictions'], result)
#         distance_dog.append((distance, row['filepaths']))
#
#     print("Length")
#     print(len(distance_dog))
#     print("After filepaths and pred")
#     print(distance_dog[0])
#     print("Sorting Arrays...")
#     distance_dog_sorted = sorted(distance_dog, key=lambda x: x[0])
#     print("First 5 Images: ")
#     for dist, path in distance_dog_sorted[:5]:
#         print(f"Distance: {dist}, Image: {path}")
#         with Image.open(path) as img:
#             img.show()
#
# else:
#     for _, row in new_df.iloc[:500].iterrows():  # First 500 rows (cats)
#         distance = calc_distance(row['Predictions'], result)
#         distance_cat.append((distance, row['filepaths']))  # Store (distance, path)
#
#     print("Length")
#     print(len(distance_cat))
#     print("After filepaths and pred")
#     print(distance_cat[0])
#     print("Sorting Arrays...")
#     distance_cat_sorted = sorted(distance_cat, key=lambda x: x[0])
#     print("First 5 Images: ")
#     for dist, path in distance_cat_sorted[:5]:
#         print(f"Distance: {dist}, Image: {path}")
#         with Image.open(path) as img:
#             img.show()













