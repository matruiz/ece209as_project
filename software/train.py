import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import neighbors, datasets
device = 'cuda:0'

""" HYPERPARAMTERS """
lr = 1e-5
epochs = 500
batch_size = 64
z_dim = 10
# USE k-fold for seen/unseen classes
unseen_classes = fold_1
unseen_classes_i3d = fold_1_i3d
seen_classes = fold_2 + fold_3 + fold_4 + fold_5
seen_classes_i3d = fold_2_i3d + fold_3_i3d + fold_4_i3d + fold_5_i3d 


def train():

  #Initialize models
  generator_net = Generator(400, z_dim, 1000, 36).to(device)
  discriminator_net = Discriminator(36, 1000, 14).to(device)
  generator_net.apply(weights_init)
  discriminator_net.apply(weights_init)
  nets = [generator_net, discriminator_net]

  #Optimizers
  optimizer_G = optim.Adam(generator_net.parameters(), lr=lr, betas=(0.5,0.9))
  optimizer_D = optim.Adam(discriminator_net.parameters(), lr=lr, betas=(0.5,0.9))


  #CHOOSE SEEN and UNSEEN classes based on folds
  unseen_data = data
  unseen_prototypes = []
  seen_data = data
  seen_prototypes = []
  for activity in seen_classes:
    unseen_data = unseen_data[(unseen_data[:,1] != activity)]
  for activity in unseen_classes:
    seen_data = seen_data[(seen_data[:,1] != activity)]
  seen_data[:,1] = np.unique(seen_data[:,1], return_inverse=True)[1] #make labels (0, classesSeen-1)
  vis_feat = np.array([seen_classes_i3d[i] for i in seen_data[:,1]])
  imu_feats = np.array([i for i in seen_data[:,0]])
  seen_data = np.append(seen_data, vis_feat, 1)
  seen_data = np.append(seen_data, imu_feats, 1)


  best_acc = 0

  for it in range(epochs):

    #shuffle seen class data
    #np.random.shuffle(seen_data)
    curr_GC_loss = 0
    curr_DC_loss = 0
    batches = 0

    for batch_start_index in range(0, seen_data.shape[0]-batch_size, batch_size):

      batch = seen_data[batch_start_index:batch_start_index+batch_size]
      real_imu_features = batch[:,2+400:]
      real_imu_labels = batch[:,1]

      v = Variable(torch.from_numpy(batch[:,2:400+2].astype('float32'))).cuda()

      """ DISCRIMINATOR """
      X = Variable(torch.from_numpy(real_imu_features.astype('float32'))).cuda()
      y_true = Variable(torch.from_numpy(real_imu_labels.astype('int'))).cuda()
      z = Variable(torch.randn(batch_size, z_dim)).cuda()

      # Discriminator update on real data
      D_real, C_real = discriminator_net(X)
      D_loss_real = torch.mean(D_real)
      C_loss_real = F.cross_entropy(C_real, y_true)
      DC_loss = -D_loss_real + C_loss_real
      DC_loss.backward()

      #Discriminator update on fake/generated data
      G_sample = generator_net(z, v).detach()
      D_fake, C_fake = discriminator_net(G_sample)
      D_loss_fake = torch.mean(D_fake)
      C_loss_fake = F.cross_entropy(C_fake, y_true)
      DC_loss = D_loss_fake + C_loss_fake
      curr_DC_loss = curr_DC_loss + DC_loss
      DC_loss.backward()

      optimizer_D.step()
      reset_grad(nets)



      """ GENERATOR """
      
      X = Variable(torch.from_numpy(real_imu_features.astype('float32'))).cuda()
      y_true = Variable(torch.from_numpy(real_imu_labels.astype('int'))).cuda()
      z = Variable(torch.randn(batch_size, z_dim)).cuda()
      v = Variable(torch.from_numpy(batch[:,2:400+2].astype('float32'))).cuda()

      # Generator update on
      G_sample = generator_net(z,v)
      D_fake, C_fake = discriminator_net(G_sample)
      _, C_real = discriminator_net(X)

      G_loss = torch.mean(D_fake)
      C_loss= (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake, y_true)) / 2

      GC_loss = -G_loss + C_loss

      # ||W||_2 regularization
      reg_loss = Variable(torch.Tensor([0.0])).cuda()
      if True:
          for name, p in generator_net.named_parameters():
              if 'weight' in name:
                  reg_loss += p.pow(2).sum()
          reg_loss.mul_(0.001)

      # ||W_z||21 regularization, make W_z sparse
      '''
      reg_Wz_loss = Variable(torch.Tensor([0.0])).cuda()
      if True != 0:
          Wz = generator_net.rdc_text.weight
          reg_Wz_loss = Wz.pow(2).sum(dim=0).sqrt().sum().mul(opt.REG_Wz_LAMBDA)
      '''

      GC_loss = GC_loss + reg_loss
      curr_GC_loss = curr_GC_loss + GC_loss
      GC_loss.backward()
      optimizer_G.step()
      reset_grad(nets)

      batches = batches + 1


    """ EVALUATE """
    if True:
      #Generate synthesized samples from all classes (using i3d data as input for Generator)
      #(36 rep data, 1 label)
      X_knn = []
      labels_knn = []
      n_samples_per_class = 100
      n_neighbors = 10

      for activity in os.listdir(directory_i3d):
        activity_semantic = Variable(torch.from_numpy(activity_i3d_dict[activity].astype('float32'))).cuda()

        for samp in range(0, n_samples_per_class):
          z = Variable(torch.randn(z_dim)).cuda()

          #print(activity_i3d_dict[activity].shape)
          #print(torch.randn(z_dim).shape)


          sample = generator_net(z, activity_semantic, test=True)
          X_knn.append(sample.cpu().detach().numpy())
          labels_knn.append(acivity_to_label[activity])


      #print(np.unique(labels_knn))
      #print(np.unique(unseen_data[:,1]))
      #fit KNN model
      clf = neighbors.KNeighborsClassifier(n_neighbors)
      clf.fit(X_knn, labels_knn)

      imu_test_feats_unseen = np.array([i for i in unseen_data[:,0]])
      score_test = clf.score(imu_test_feats_unseen, unseen_data[:,1].astype('int'))

      imu_test_feats_seen = np.array([i for i in seen_data[:,0]])
      training_score = clf.score(imu_test_feats_seen, seen_data[:,1].astype('int'))

      if score_test > best_acc:
        best_acc = score_test

      print('Epoch', it)
      print('Cur acc:', score_test, '  Best acc:', best_acc, '     Train score:', training_score)
      print('DC LOSS:', (curr_DC_loss/batches).cpu().detach().numpy(), '  GC LOSS :', (curr_GC_loss/batches).cpu().detach().numpy(), '\n')



train()