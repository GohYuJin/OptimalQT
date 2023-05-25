import torch
import numpy as np



def create_default_qtables():
    y_table = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61], 
         [12, 12, 14, 19, 26, 58, 60, 55], 
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62], 
         [18, 22, 37, 56, 68, 109, 103, 77], 
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101], 
         [72, 92, 95, 98, 112, 100, 103, 99]],
        dtype=np.float32).T

    y_table = torch.from_numpy(y_table)

    c_table = np.empty((8, 8), dtype=np.float32)
    c_table.fill(99)
    c_table[:4, :4] = np.array([[17, 18, 24, 47], 
                                [18, 21, 26, 66],
                                [24, 26, 56, 99], 
                                [47, 66, 99, 99]]).T
        
    c_table = torch.from_numpy(c_table)
    return y_table, c_table


def return_class_name(id_classname_json, predictions):
    max_dim = predictions.argmax(dim=1).item()
    class_name = id_classname_json[str(max_dim)][1]
    return class_name , max_dim


def return_class_accuracy(predictions, class_id):
    prob = torch.nn.functional.softmax(predictions, dim=1)
    accuracy = prob[0, class_id] * 100
    return torch.round(accuracy).item()


# https://github.com/nisharaichur/Fast-Gradient-Signed-Method-FGSM

def visualize(image, adv_image, epsilon, gradients,  target_class, adversarial_class, target_acc, adversarial_acc, acc_of_original):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    image = image.squeeze(0) 
    image = image.detach() 
    image = image.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).detach().numpy()
    image = np.transpose( image , (1,2,0))   
    image = np.clip(image, 0, 1)

    adv_image = adv_image.squeeze(0)
    adv_image = adv_image.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).detach().numpy()
    adv_image = np.transpose( adv_image , (1,2,0))  
    adv_image = np.clip(adv_image, 0, 1)

    gradients = gradients.squeeze(0).detach().numpy()
    gradients = np.transpose(gradients, (1,2,0))
    gradients = np.clip(gradients, 0, 1)

    figure, ax = plt.subplots(1,3, figsize=(18,8))
    ax[0].imshow(image)
    ax[0].set_title('Original Image', fontsize=20)
    ax[0].axis("off")

    ax[1].imshow(gradients)
    ax[1].set_title('Perturbation epsilon: {}'.format(epsilon), fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].imshow(adv_image)
    ax[2].set_title('Adversarial Example', fontsize=20)
    ax[2].axis("off")

    ax[0].text(0.5,-0.13, "Prediction: {}\n Accuracy: {}%".format(target_class, target_acc), size=15, ha="center", transform=ax[0].transAxes)
    ax[2].text(0.5,-0.13, "Prediction of {} is {}%\n Prediction of {} is {}%".format(adversarial_class, adversarial_acc, target_class, acc_of_original), size=15, ha="center", transform=ax[2].transAxes)
    plt.show()