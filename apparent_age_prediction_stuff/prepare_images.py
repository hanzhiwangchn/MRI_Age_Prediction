from PIL import Image
import face_recognition
import os
import pandas as pd
import numpy as np


def detect_and_resize_images(parent_dir):
    # detect faces from three folders and save to new folders
    os.makedirs(parent_dir + 'train_face_images', exist_ok=True)
    os.makedirs(parent_dir + 'validation_face_images', exist_ok=True)
    os.makedirs(parent_dir + 'test_face_images', exist_ok=True)

    for data_set in ['Train/', 'Validation/', 'Test/']:
        for image_name in os.listdir(path=parent_dir + data_set):
            image_id = image_name.split('.')[0].split('_')[-1]
            # Load the jpg file into a numpy array
            image = face_recognition.load_image_file(parent_dir + data_set + image_name)

            face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2)

            # if at least one face is detected
            if len(face_locations) > 0:
                top, right, bottom, left = face_locations[0]
                # You can access the actual face itself like this:
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                # pil_image.show()
                pil_image = pil_image.resize((100, 100), Image.ANTIALIAS)
                if data_set == 'Validation/':
                    pil_image.save(parent_dir + 'validation_face_images/' + f'image_{image_id}.jpg')
                elif data_set == 'Test/':
                    pil_image.save(parent_dir + 'test_face_images/' + f'image_{image_id}.jpg')


def load_data_into_array(parent_dir, data_set):
    # load images from new folders and save to array
    image_list = []
    label_list = []

    # load and format label file
    df = pd.read_csv(parent_dir + f'{data_set.split("_")[0]}_label.csv', header=None, names=['image_id', 'age', 'std'])
    df['age'] = df['image_id'].apply(lambda x: x.split(' ')[1].strip())
    df['std'] = df['image_id'].apply(lambda x: x.split(' ')[2].strip())
    df['image_id'] = df['image_id'].apply(lambda x: x.split(' ')[0].strip())
    df['id'] = df['image_id'].apply(lambda x: x.split('.')[0].split('_')[-1])

    print(df.isna().sum())
    df.set_index(df['id'], inplace=True)

    # save images and labels
    for image_name in sorted(os.listdir(path=parent_dir + data_set)):
        image_id = image_name.split('.')[0].split('_')[-1]

        image = Image.open(parent_dir + data_set + image_name)
        image_array = np.array(image.getdata()).reshape((3, 100, 100))
        image_list.append(image_array)

        image_label = df.loc[image_id, 'age']
        label_list.append(int(image_label))

    image_list = np.reshape(image_list, (len(image_list), 3, 100, 100)).astype(np.float32)
    np.save(f'{data_set[:-1]}', image_list)
    label_list = np.reshape(label_list, (len(label_list), )).astype(np.float32)
    np.save(f'{data_set[:-2]}' + '_labels.npy', label_list)


def combine_all_images():
    image_train = np.load('train_face_images.npy')
    image_val = np.load('validation_face_images.npy')
    image_test = np.load('test_face_images.npy')

    label_train = np.load('train_face_image_labels.npy')
    label_val = np.load('validation_face_image_labels.npy')
    label_test = np.load('test_face_image_labels.npy')

    total_images = np.concatenate([image_train, image_val, image_test], axis=0)
    total_labels = np.concatenate([label_train, label_val, label_test], axis=0)

    np.save('../total_face_images.npy', total_images)
    np.save('../total_face_labels.npy', total_labels)


if __name__ == '__main__':
    # detect_and_resize_images(parent_dir='../../Desktop/cha_learn/')
    # load_data_into_array(parent_dir='../../Desktop/cha_learn/', data_set='test_face_images/')
    combine_all_images()
