from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
import torch.distributions.normal as normal

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        self.old_model = MYNET(self.args, mode=self.args.base_mode)
        self.old_model = nn.DataParallel(self.old_model, list(range(self.args.num_gpu)))
        self.old_model = self.old_model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def get_novel_dataloader(self, session):
        if session == 0:
            _, _, testloader = get_base_dataloader(self.args)
        else:
            testloader = get_novel_test_dataloader(self.args, session)
        return testloader

    def get_task_specific_test_dataloader(self, session):
        if session == 0:
            _, _, testloader = get_base_dataloader(self.args)
        else:
            testloader = get_task_specific_new_dataloader(self.args, session)
        return testloader


    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)



            self.model.load_state_dict(self.best_model_dict)
            best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                #self.best_model_dict = torch.load(best_model_dir)['params']
                #self.model.load_state_dict(self.best_model_dict)
                
                for epoch in range(args.epochs_base):
                    start_time = time.time()
                   
                
                    # train base sess
                    tl, ta, treg = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    # test model with all seen class
                    tsl, tsa, _,_ = test_agg(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('epoch:%03d,lr:%.4f,training_ce_loss:%.5f, training_reg_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, treg,ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
                
                if not args.not_data_init:
                    #print("Went inside #################")
                    print("Updating old class with class means ")
                    self.model.load_state_dict(self.best_model_dict)
                    #self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa, _ ,_= test_agg(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))
                
                
               
                

            else:  # incremental learning sessions
                
                print("training session: [%d]" % session)
                previous_class = (args.base_class + (session-1) * args.way) 
                present_class = (args.base_class + session * args.way) 
                

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
               
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                ### We can replace with semanctic closed variances here ####

                #############################################################
                
                
                
                
                self.model.train()
                for parameter in self.model.module.parameters():
                    parameter.requires_grad = False

                for m in self.model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

                for parameter in self.model.module.fc.parameters():
                    parameter.requires_grad = True
                

                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name)

                

                optimizer = torch.optim.SGD(self.model.parameters(),lr=self.args.lr_new, momentum=0.9, dampening=0.9 , weight_decay=0)
                
                
                print('Started fine tuning')
                T = 2
                beta = 0.25
                with torch.enable_grad():
                    for epoch in range(self.args.epochs_new):
                        for batch in trainloader:
                            inputs, label = [_.cuda() for _ in batch]


                            inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
                            inputs = inputs.view(-1, 3, 32, 32)
                            label = torch.stack([label * 4 + k for k in range(4)], 1).view(-1)
                           
                            logits, feature, attention_op = self.model(inputs, stochastic = False)
                            
                        
                            protos = args.proto_list
                            indexes = torch.randperm(protos.shape[0])
                            protos = protos[indexes]
                            temp_protos = protos.cuda()
                            
                           
                            
                            num_protos = temp_protos.shape[0] 
                             
                            
                            label_proto = torch.arange(previous_class).cuda()
                            label_proto = label_proto[indexes] * 4
                            
                            
                            temp_protos = torch.cat((temp_protos,feature))
                            label_proto = torch.cat((label_proto,label))
                            logits_protos = self.model.module.fc(temp_protos, stochastic=True)
                            ############################

                            
                            loss_proto = nn.CrossEntropyLoss()(logits_protos[:num_protos,:present_class*4], label_proto[:num_protos]) * args.lamda_proto 
                            loss_ce = nn.CrossEntropyLoss()(logits_protos[num_protos:, :present_class*4], label_proto[num_protos:] )
                            
                            optimizer.zero_grad()
                            
                            
                            loss = loss_proto + loss_ce
                            loss.backward()
                            optimizer.step()
                            #self.model.module.fc.mu.data[:previous_class] = old_fc
                        print('Epoch: {}, Loss_CE: {}, Loss proto:{}, Loss: {}'.format(epoch, loss_ce, loss_proto, loss))
                        #print(c)
                
                
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                #########################################################################            
                
            ################  Printing performance metrics ##################
            self.model.module.mode = self.args.new_mode
            self.model.eval()
            
            _, _, testloader = self.get_dataloader(session)
            tsl, tsa, tsa_agg, _ = test_agg(self.model, testloader, 0, args, session, print_numbers=True)
            print('Overall cumulative accuracy: {}, after agg: {}'.format(tsa*100, tsa_agg*100))
            save_features(self.model, testloader, args, session)
            
            testloader = self.get_novel_dataloader(session)
            tsl_novel, tsa_novel, tsa_agg_novel, _= test_agg(self.model, testloader, 0, args, session)
            print('Novel classes cumulative accuracy: {}, after agg: {}'.format(tsa_novel*100, tsa_agg_novel*100))
            hm = 0
            hm_agg = 0
            hm_agg_stoc_agg = 0
            for j in range(0,session+1):
                testloader = self.get_task_specific_test_dataloader(j)
                tsl, tsa,tsa_agg, _= test_agg(self.model, testloader, 0, args, session)
                if session ==0:
                    tsa_base = tsa
                    tsa_agg_base = tsa_agg
                print('session: {} test accuracy: {}, after agg: {}'.format(j, tsa * 100, tsa_agg*100))
                hm += 1/((tsa+0.0000000001)*100)
                hm_agg += 1/((tsa_agg+0.00000001)*100)
            if session>0:    
                print('Task wise Harmonic mean is : {}, agg: {}'.format((session+1)/hm, (session+1)/hm_agg))
                hm = (2*tsa_base*tsa_novel)/(tsa_base+0.00000001+tsa_novel)
                hm_agg = ((2*tsa_agg_base*tsa_agg_novel)/(tsa_agg_base+tsa_agg_novel+0.00000001))
                print('Harmonic mean between old and new classes : {}, agg: {}'.format(hm*100,hm_agg*100))
            ###################################################################
            
            ############ Update protos and save features #####################
            update_sigma_protos_feature_output(trainloader, train_set, testloader.dataset.transform, self.model, args, session)
            
            
            
            print('protos, radius', args.proto_list.shape, args.radius)

        
            self.model.module.mode = self.args.new_mode
            ##################################################################


    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        self.args.save_path = self.args.save_path + '%s/' % self.args.Method
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
