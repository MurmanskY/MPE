import torch


def getPthKeys(paraPath):
    """
    返回layers
    :param paraPath: 待获得的参数pth
    :return:
    """
    return torch.load(paraPath, map_location=torch.device("mps")).keys()



def modify_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # 移除'module.'，长度为7
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


if __name__ == '__main__':


    # 加载原始的.pth文件
    original_path = './convnext_large/ImageNetRe/convnext_large_40layers_4inter_7corr_ep_1_error.pth'
    model_state_dict = torch.load(original_path, map_location=torch.device("mps"))

    # 修改key
    modified_state_dict = modify_keys(model_state_dict)

    # 保存修改后的参数到新的文件
    new_path = './convnext_large/ImageNetRe/convnext_large_40layers_4inter_7corr_ep_1.pth'
    torch.save(modified_state_dict, new_path)

    print(f"Modified model saved to {new_path}")



    # 对比修改后的pth文件和原始pth文件之间的差异
    change_keys = torch.load(new_path, map_location=torch.device("mps"))
    init_keys = torch.load(original_path, map_location=torch.device("mps"))

    num = 0
    for change_key, init_key in zip(change_keys, init_keys):
        if change_key != init_key:
            print(change_key, init_key)
            num += 1

    print(num)