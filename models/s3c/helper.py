# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    treg = Averager()
    model = model.train()
    
    for i, batch in enumerate(trainloader):
        data, train_label = [_.cuda() for _ in batch]
        
        ####### Self supervision #########
        data = torch.stack([torch.rot90(data, k, (2, 3)) for k in range(4)], 1)
        data = data.view(-1, 3, 32, 32)
        train_label = torch.stack([train_label * 4 + k for k in range(4)], 1).view(-1)
        
        #

        logits, _, _ = model(data, stochastic = True)
        logits = logits[:, :args.base_class*4]
        loss = F.cross_entropy(logits, train_label)
        #print(c)
        acc = count_acc(logits, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    #treg = treg.item()
    treg = 0
    return tl, ta, treg


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding, _ = model(data, stochastic = False)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.mu.data[:args.base_class] = proto_list

    return model
def replace_fc(trainset, transform, model, args, session):
    present_class = (args.base_class + session * args.way) * 4
    previous_class = (args.base_class + (session-1) * args.way) * 4
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            data = torch.stack([torch.rot90(data, k, (2, 3)) for k in range(4)], 1)
            data = data.view(-1, 3, 32, 32)
            label = torch.stack([label * 4 + k for k in range(4)], 1).view(-1)
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(previous_class, present_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.mu[previous_class:present_class] = proto_list

    return model

def update_sigma_protos_feature_output(trainloader, trainset, transform, model, args, session):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    
    
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            #print(data.shape)
            #model.module.mode = 'encoder'
            _,embedding, _ = model(data, stochastic=False)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    radius = []
    if session == 0:
        
        for class_index in range(args.base_class):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            #embedding_this = F.normalize(embedding_this, p=2, dim=-1)
            #print('dim of emd', embedding_this.shape)
            #print(c)
            feature_class_wise = embedding_this.numpy()
            cov = np.cov(feature_class_wise.T)
            radius.append(np.trace(cov)/64)
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        
        args.radius = np.sqrt(np.mean(radius)) 
        args.proto_list = torch.stack(proto_list, dim=0)
    else:
        for class_index in  np.unique(trainset.targets):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            #embedding_this = F.normalize(embedding_this, p=2, dim=-1)
            #print('dim of emd', embedding_this.shape)
            #print(c)
            feature_class_wise = embedding_this.numpy()
            cov = np.cov(feature_class_wise.T)
            radius.append(np.trace(cov)/64)
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        args.proto_list = torch.cat((args.proto_list, torch.stack(proto_list, dim=0)), dim =0)


        
    


def test_agg(model, testloader, epoch, args, session, print_numbers=False):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va_0 = Averager()
    va_90 = Averager()
    va_180 = Averager()
    va_270 = Averager()
    va_agg = Averager()
    va_0_stochastic_agg = Averager()
    va_90_stochastic_agg = Averager()
    va_180_stochastic_agg = Averager()
    va_270_stochastic_agg = Averager()
    va_agg_stochastic_agg = Averager()
    num_stoch_samples = 10
    with torch.no_grad():
        #tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(testloader):
            data, test_label = [_.cuda() for _ in batch]
            data = torch.stack([torch.rot90(data, k, (2, 3)) for k in range(4)], 1)# teja
            
            
            data = data.view(-1, 3, 32, 32)# teja
            
            logits, features, _ = model(data, stochastic = False)
            
            
            logits_0 = logits[0::4, 0:test_class*4:4]
            logits_90 = logits[1::4, 1:test_class*4:4]
            logits_180 = logits[2::4, 2:test_class*4:4]
            logits_270 = logits[3::4, 3:test_class*4:4]
            logits_agg = (logits_0 + logits_90 + logits_180 + logits_270)/4
         

            logits_original = logits[0::4, :test_class*4:4]
            
            loss = F.cross_entropy(logits_original, test_label)
            acc = count_acc(logits_original, test_label)
            acc_0 = count_acc(logits_0, test_label)
            acc_90 = count_acc(logits_90, test_label)
            acc_180 = count_acc(logits_180, test_label)
            acc_270 = count_acc(logits_270, test_label)
            acc_agg = count_acc(logits_agg, test_label)

 
            vl.add(loss.item())
            va.add(acc)
            va_0.add(acc_0)
            va_90.add(acc_90)
            va_180.add(acc_180)
            va_270.add(acc_270)
            va_agg.add(acc_agg)

            

        vl = vl.item()
        va = va.item()
        va_agg = va_agg.item()
        va_0 = va_0.item()
        va_90 = va_90.item()
        va_180 = va_180.item()
        va_270 = va_270.item()
        
    return vl, va, va_agg, 0


def save_features(model, testloader, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va_0 = Averager()
    va_90 = Averager()
    va_180 = Averager()
    va_270 = Averager()
    va_agg = Averager()

    class_means = model.module.fc.mu.data[:test_class*4:4, :].detach()
    class_means = F.normalize(class_means, p=2, dim=-1)
    data_features = []
    labels = []
    predictions = []
    
    
    with torch.no_grad():
        #tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(testloader):
            data, test_label = [_.cuda() for _ in batch]
            data = torch.stack([torch.rot90(data, k, (2, 3)) for k in range(4)], 1)# teja
            
            
            data = data.view(-1, 3, 32, 32)# teja
            #test_label = torch.stack([test_label * 4 + k for k in range(4)], 1).view(-1)
            #print(test_label)
            #print(c)
            logits, features, _ = model(data, stochastic = False)
            #print('1',logits[4])
            #print('1',logits[5])
            
            logits_0 = logits[0::4, 0:test_class*4:4]
            logits_90 = logits[1::4, 1:test_class*4:4]
            logits_180 = logits[2::4, 2:test_class*4:4]
            logits_270 = logits[3::4, 3:test_class*4:4]
            logits_agg = (logits_0 + logits_90 + logits_180 + logits_270)/4
         

            logits_original = logits[0::4, :test_class*4:4]
            
            loss = F.cross_entropy(logits_original, test_label)
            acc = count_acc(logits_original, test_label)
            acc_0 = count_acc(logits_0, test_label)
            acc_90 = count_acc(logits_90, test_label)
            acc_180 = count_acc(logits_180, test_label)
            acc_270 = count_acc(logits_270, test_label)
            acc_agg = count_acc(logits_agg, test_label)

 
            vl.add(loss.item())
            va.add(acc)
            va_0.add(acc_0)
            va_90.add(acc_90)
            va_180.add(acc_180)
            va_270.add(acc_270)
            va_agg.add(acc_agg)

            data_features.append(features[::4])
            labels.append(test_label)
            predictions.append(torch.argmax(logits_agg, dim=1))


            

        vl = vl.item()
        va = va.item()
        va_agg = va_agg.item()
        va_0 = va_0.item()
        va_90 = va_90.item()
        va_180 = va_180.item()
        va_270 = va_270.item()

    data_features = torch.stack(data_features).view(-1, 64)
    data_features = F.normalize(data_features, p=2, dim=-1)
    labels = torch.stack(labels).view(-1,1)
    predictions = torch.stack(predictions).view(-1,1)     
    
    with open(args.save_path+'/S3C_features_session_'+ str(session)+'.npy', 'wb' ) as f:
        np.save(f, class_means.cpu().detach().numpy())
        #np.save(f, Glove_inputs.cpu().detach().numpy())
        np.save(f, data_features.cpu().detach().numpy())
        np.save(f, labels.cpu().detach().numpy())
        np.save(f, predictions.cpu().detach().numpy())