
import torch
import torch.nn as nn
import torch.nn.functional as F



def upsampling(x):
    b_z = x.shape[0]
    c = x.shape[1]
    l = x.shape[3]


    #---for batch---
    for batch in range(b_z):
        each_batch = x[batch,...]
        each_batch = each_batch.reshape(1,c,l,l)
        # print('batch',each_batch.shape,each_batch)
    #---for channel----
        for channel in range(c):
            each_channel = each_batch[0,channel,...]
            each_channel = each_channel.reshape(1,1,l,l)
            print('channel',each_channel.shape,each_channel)
    
    #---for shape---
            s = each_channel.reshape(1,1,-1,1)
            s = torch.cat([s,s],axis=3)
            print(s.shape,s)
    
            s = s.reshape(1,1,l,-1)

            temp = s[0,0,0,:]
            temp = temp.reshape(1,1,1,-1)
            upsample = temp
            upsample = torch.cat([upsample,temp],axis=2)
            # print(upsample.shape,upsample)
            # for i in range(l-1):
            #     for _ in range(2):
            #         temp = x[0,0,i+1,:]
            #         temp = temp.reshape(1,1,1,-1)
            #         print(temp.shape,temp)
            #         upsample = torch.cat([upsample,temp],axis=2)
            
 
   
    # return 

if __name__ == '__main__':
    # x = [1,2,3,4,5,6,7,8,9]
    x = torch.randn(2,3,2,2)
    x = torch.tensor(x)
    print(x.shape)
    # x = x.reshape(3,3)
    print(x)
    x = upsampling(x)
    # print(x.shape,x)
    # print(x[0,0,0,:])
    

    