import torch

import numpy as np

class AdversarySampler:
    def __init__(self, budget):
        self.budget = budget


    def sample(self, vae, discriminator, data, cuda):
        all_preds = []
        all_indices = []


        #print(data)
        for images, _, indices in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)
            #print(len(all_preds))

        all_preds = torch.stack(all_preds)
        #print(all_preds.shape)
	# last column is unlableled prediction
        all_preds = all_preds[:, -1].view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        #print(all_preds.shape, all_preds)
        #print(querry_indices.shape, querry_indices)
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices
        
