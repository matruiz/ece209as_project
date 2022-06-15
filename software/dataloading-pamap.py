# Subject processing for PAMAP

import numpy as np
subject1 = np.loadtxt('gdrive/MyDrive/209 Final Proj/pamap2/optional/subject109.dat', unpack = True)

subject = subject1[:,(subject1[1,:] != 0)] #remove where activity is 0
subject.shape, subject1.shape

for i in range(0,subject.shape[1],412): #increment such that there is 1 second overlap
  #reach end, not 5.12 sec interval
  if i + 512 > subject.shape[1]:
    break

  #instance goes from now (i) to 5.12 sec (i+512)
  #calc mean

  #activity = np.bincount(subject[1,i:i+512]).argmax()

  u, c = np.unique(subject[1,i:i+512], return_counts=True)
  activity = u[c.argmax()]
  
  #hand
  hand_accX_mean = np.nanmean(subject[4,i:i+512])
  hand_accX_std = np.nanstd(subject[4,i:i+512])
  hand_accY_mean = np.nanmean(subject[5,i:i+512])
  hand_accY_std = np.nanstd(subject[5,i:i+512])
  hand_accZ_mean = np.nanmean(subject[6,i:i+512])
  hand_accZ_std = np.nanstd(subject[6,i:i+512])
  hand_gyroX_mean = np.nanmean(subject[10,i:i+512])
  hand_gyroX_std = np.nanstd(subject[10,i:i+512])
  hand_gyroY_mean = np.nanmean(subject[11,i:i+512])
  hand_gyroY_std = np.nanstd(subject[11,i:i+512])
  hand_gyroZ_mean = np.nanmean(subject[12,i:i+512])
  hand_gyroZ_std = np.nanstd(subject[12,i:i+512])

  #chest
  chest_accX_mean = np.nanmean(subject[21,i:i+512])
  chest_accX_std = np.nanstd(subject[21,i:i+512])
  chest_accY_mean = np.nanmean(subject[22,i:i+512])
  chest_accY_std = np.nanstd(subject[22,i:i+512])
  chest_accZ_mean = np.nanmean(subject[23,i:i+512])
  chest_accZ_std = np.nanstd(subject[23,i:i+512])
  chest_gyroX_mean = np.nanmean(subject[27,i:i+512])
  chest_gyroX_std = np.nanstd(subject[27,i:i+512])
  chest_gyroY_mean = np.nanmean(subject[28,i:i+512])
  chest_gyroY_std = np.nanstd(subject[28,i:i+512])
  chest_gyroZ_mean = np.nanmean(subject[29,i:i+512])
  chest_gyroZ_std = np.nanstd(subject[29,i:i+512])

  #ankle
  ankle_accX_mean = np.nanmean(subject[38,i:i+512])
  ankle_accX_std = np.nanstd(subject[38,i:i+512])
  ankle_accY_mean = np.nanmean(subject[39,i:i+512])
  ankle_accY_std = np.nanstd(subject[39,i:i+512])
  ankle_accZ_mean = np.nanmean(subject[40,i:i+512])
  ankle_accZ_std = np.nanstd(subject[40,i:i+512])
  ankle_gyroX_mean = np.nanmean(subject[44,i:i+512])
  ankle_gyroX_std = np.nanstd(subject[44,i:i+512])
  ankle_gyroY_mean = np.nanmean(subject[45,i:i+512])
  ankle_gyroY_std = np.nanstd(subject[45,i:i+512])
  ankle_gyroZ_mean = np.nanmean(subject[46,i:i+512])
  ankle_gyroZ_std = np.nanstd(subject[46,i:i+512])
  
  features = [hand_accX_mean, hand_accX_std, hand_accY_mean, hand_accY_std, hand_accZ_mean, hand_accZ_std,
              hand_gyroX_mean, hand_gyroX_std, hand_gyroY_mean, hand_gyroY_std, hand_gyroZ_mean, hand_gyroZ_std,
              
              chest_accX_mean, chest_accX_std, chest_accY_mean, chest_accY_std, chest_accZ_mean, chest_accZ_std,
              chest_gyroX_mean, chest_gyroX_std, chest_gyroY_mean, chest_gyroY_std, chest_gyroZ_mean, chest_gyroZ_std,
              
              ankle_accX_mean, ankle_accX_std, ankle_accY_mean, ankle_accY_std, ankle_accZ_mean, ankle_accZ_std,
              ankle_gyroX_mean, ankle_gyroX_std, ankle_gyroY_mean, ankle_gyroY_std, ankle_gyroZ_mean, ankle_gyroZ_std]
  features_np = np.asarray(features)

  instance = [features_np, activity] #X, label

  data.append(instance)
  
len(data)




''' Class labels '''
lying = 0
sitting = 1
standing = 2
walking = 3
running = 4
cycling = 5
Nordic_walking = 6
watching_TV = 7
computer_work = 8
car_driving = 9
ascending_stairs = 10
descending_stairs = 11
vacuum_cleaning = 12
ironing = 13
folding_laundry = 14
house_cleaning = 15
playing_soccer = 16
rope_jumping = 17
''''''

''' FOLDS '''
fold_1 = [watching_TV, house_cleaning, standing, ascending_stairs]
fold_2 = [walking, rope_jumping, sitting, descending_stairs]
fold_3 = [playing_soccer, lying, vacuum_cleaning, computer_work]
fold_4 = [cycling, running, Nordic_walking]
fold_5 = [ironing, car_driving, folding_laundry]









# Find protypes of i3d videos
import os
import numpy as np
directory_i3d = 'gdrive/MyDrive/209 Final Proj/i3drgb/'

activity_i3d_dict = {}

for activity_i3d in os.listdir(directory_i3d):

  save_np_file = activity_i3d + '_prototype.npy'

  directory = directory_i3d + activity_i3d

  sum_tensors = []

  for file in os.listdir(directory):
    f = os.path.join(directory, file)
    video_attributes = np.load(f, allow_pickle=True)
    sum_tensors.append(video_attributes)

  activity_mean = np.mean(sum_tensors, axis=0).T
  activity_i3d_dict[activity_i3d] = activity_mean.T[0]

#same order as class labels [lying, ..., rope jumping]
#fold_1 = [watching_TV, house_cleaning, standing, ascending_stairs]
fold_1_i3d = [activity_i3d_dict['watching tv'], activity_i3d_dict['house cleaning'], activity_i3d_dict['standing'], activity_i3d_dict['ascending stairs']]

#fold_2 = [walking, rope_jumping, sitting, descending_stairs]
fold_2_i3d = [activity_i3d_dict['walking'], activity_i3d_dict['rope jumping'], activity_i3d_dict['sitting'], activity_i3d_dict['descending stairs']]

#fold_3 = [playing_soccer, lying, vacuum_cleaning, computer_work]
fold_3_i3d = [activity_i3d_dict['playing soccer'], activity_i3d_dict['lying'], activity_i3d_dict['vacuum cleaning'], activity_i3d_dict['computer work']]

#fold_4 = [cycling, running, Nordic_walking]
fold_4_i3d = [activity_i3d_dict['cycling'], activity_i3d_dict['running'], activity_i3d_dict['nordic walking']]

#fold_5 = [ironing, car_driving, folding_laundry]
fold_5_i3d = [activity_i3d_dict['ironing'], activity_i3d_dict['car driving'], activity_i3d_dict['folding laundry']]