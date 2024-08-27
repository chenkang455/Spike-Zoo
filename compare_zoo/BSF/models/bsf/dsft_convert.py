import torch


def convert_dsft4(dsft, spike):
    '''
    input: Pytorch Tensor
        dsft: dsft(1,1)  b x T x h x w
        spike: 01 spike  b x T x h x w
    output: Pytorch Tensor
        dsft_dict: {dsft(1,1), dsft(1,2), dsft(2,1), dsft(2,2)}
    '''

    b, T, h, w = spike.shape

    ## dsft_mask_left_shift  -- abbr. -->  dmls1,  (right-shift: dmrs1)
    dmls1 = -1 * torch.ones(spike.shape, device=spike.device, dtype=torch.float32)
    dmrs1 = -1 * torch.ones(spike.shape, device=spike.device, dtype=torch.float32)

    ## for dmls1
    # flag的用途是为了边界的copy-padding
    flag = -2 * torch.ones([b, h, w], device=spike.device, dtype=torch.float32)
    for ii in range(T-1, 0-1, -1):
        flag += (spike[:,ii]==1)

        copy_pad_coord = (flag < 0)
        dmls1[:,ii][copy_pad_coord] = dsft[:,ii][copy_pad_coord]

        if ii < T-1:
            ## dmls1的数据该更新的情况
            update_coord = (spike[:,ii+1]==1) * (~copy_pad_coord)
            dmls1[:,ii][update_coord] = dsft[:,ii+1][update_coord]

            ## dmls1的数据不该更新，该继承之前的数的情况
            non_update_coord = (spike[:,ii+1]!=1) * (~copy_pad_coord)
            dmls1[:,ii][non_update_coord] = dmls1[:, ii+1][non_update_coord]
    
    
    ## for dmrs1
    # flag的用途是为了边界的copy-padding
    flag = -2 * torch.ones([b, h, w], device=spike.device, dtype=torch.float32)
    for ii in range(0, T, 1):
        flag += (spike[:,ii]==1)

        ## for 边界的 copy-padding
        copy_pad_coord = (flag < 0)
        dmrs1[:,ii][copy_pad_coord] = dsft[:,ii][copy_pad_coord]

        if ii > 0:
            ## dmrs1的数据该更新的情况
            update_coord = (spike[:,ii]==1) * (~copy_pad_coord)
            dmrs1[:,ii][update_coord] = dsft[:,ii-1][update_coord]

            ## dmrs1的数据不该更新，该继承之前的数的情况
            non_update_coord = (spike[:,ii]!=1) * (~copy_pad_coord)
            dmrs1[:,ii][non_update_coord] = dmrs1[:, ii-1][non_update_coord]
    

    dsft12 = dsft + dmls1
    dsft21 = dsft + dmrs1
    dsft22 = dsft + dmls1 + dmrs1


    dsft_dict = {
        'dsft11': dsft,
        'dsft12': dsft12,
        'dsft21': dsft21,
        'dsft22': dsft22,
    }

    return dsft_dict



if __name__ == '__main__':
    # spike = [0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0]
    # dsft  = [2,2,2,2,2,5,5,5,5,5,3,3,3,2,2,4,4,4,4,4,4]

    spike = [0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0]
    dsft  = [3,3,3,3,3,4,4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,4,4,4,4,3,3,3,3,3,3,3,3]

    spike = torch.tensor(spike, device='cpu', dtype=torch.float32)[None,:,None,None]
    dsft  = torch.tensor(dsft , device='cpu', dtype=torch.float32)[None,:,None,None]

    dsft_dict = convert_dsft4(dsft=dsft, spike=spike)
    dsft_11 = dsft_dict['dsft11']
    dsft_12 = dsft_dict['dsft12']
    dsft_21 = dsft_dict['dsft21']
    dsft_22 = dsft_dict['dsft22']

    print(dsft_11[0,:,0,0])
    print()
    print(dsft_12[0,:,0,0])
    print()
    print(dsft_21[0,:,0,0])
    print()
    print(dsft_22[0,:,0,0])