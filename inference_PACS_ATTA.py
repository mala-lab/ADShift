import torch
from torchvision.datasets import ImageFolder
from resnet_TTA import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
import torchvision.transforms as transforms
from test import evaluation_ATTA


def test_PACS(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    labels_dict = {
        0: 'dog',
        1: 'elephant',
        2: 'giraffe',
        3: 'guitar',
        4: 'horse',
        5: 'house',
        6: 'person'
    }
    name_dataset = labels_dict[_class_]
    print('Class: ', name_dataset)

    #load data
    size = 256
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    img_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])

    test_path_ID = './PACS/val/photo/' #update here
    test_path_OOD_art_painting = './PACS/val/art_painting/' #update here
    test_path_OOD_cartoon = './PACS/val/cartoon/' #update here
    test_path_OOD_sketch = './PACS/val/sketch/' #update here

    test_data_ID = ImageFolder(root=test_path_ID, transform=img_transforms)
    test_data_OOD_art_painting = ImageFolder(root=test_path_OOD_art_painting, transform=img_transforms)
    test_data_OOD_cartoon = ImageFolder(root=test_path_OOD_cartoon, transform=img_transforms)
    test_data_OOD_sketch = ImageFolder(root=test_path_OOD_sketch, transform=img_transforms)

    data_ID_loader = torch.utils.data.DataLoader(test_data_ID, batch_size=1, shuffle=False)
    data_OOD_art_painting_loader = torch.utils.data.DataLoader(test_data_OOD_art_painting, batch_size=1, shuffle=False)
    data_OOD_cartoon_loader = torch.utils.data.DataLoader(test_data_OOD_cartoon, batch_size=1, shuffle=False)
    data_OOD_sketch_loader = torch.utils.data.DataLoader(test_data_OOD_sketch, batch_size=1, shuffle=False)

    ckp_path_decoder = './checkpoints/' + 'PACS_DINL_' + str(_class_) + '_19.pth'

    #load model
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    #load checkpoint
    ckp = torch.load(ckp_path_decoder)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'], strict=False)
    bn.load_state_dict(ckp['bn'], strict=False)
    decoder.eval()
    bn.eval()

    lamda = 0.5

    list_results = []
    auroc_sp = evaluation_ATTA(encoder, bn, decoder, data_ID_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    print('Sample Auroc_ID {:.4f}'.format(auroc_sp))
    list_results.append(auroc_sp)

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, data_OOD_art_painting_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    print('Sample Auroc_art {:.4f}'.format(auroc_sp))
    list_results.append(auroc_sp)

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, data_OOD_cartoon_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results.append(auroc_sp)
    print('Sample Auroc_cartoon {:.4f}'.format(auroc_sp))

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, data_OOD_sketch_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results.append(auroc_sp)
    print('Sample Auroc_sketch {:.4f}'.format(auroc_sp))
    print(list_results)


    return


for i in range(0,7):
    test_PACS(i)
    print('===============================================')
    print('')
    print('')

