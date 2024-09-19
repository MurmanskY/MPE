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
    for key in state_dict.keys():
        new_key = key.replace('mlp.0.', 'mlp.linear_1.').replace('mlp.3.', 'mlp.linear_2.')
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


if __name__ == '__main__':


    # 加载原始的.pth文件
    original_path = './vith14/2CIFAR100/frac_23_ep_5.pth'
    model_state_dict = torch.load(original_path, map_location=torch.device("mps"))

    # 修改key
    modified_state_dict = modify_keys(model_state_dict)

    # 保存修改后的参数到新的文件
    new_path = './vith14/2CIFAR100_right_key/frac_23_ep_5.pth'
    torch.save(modified_state_dict, new_path)

    print(f"Modified model saved to {new_path}")



    # 对比修改后的pth文件和原始pth文件之间的差异
    change_keys = torch.load(new_path, map_location=torch.device("mps"))
    init_keys = torch.load('./vith14/bitFlip/frac_23.pth', map_location=torch.device("mps"))

    num = 0
    for change_key, init_key in zip(change_keys, init_keys):
        if change_key != init_key:
            print(change_key, init_key)
            num += 1

    print(num)