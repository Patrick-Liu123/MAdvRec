import torch
from torch import nn
from torchfm.layer import FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron, FactorizationMachine


class DeepCrossNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.

    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims1, field_dims2, embed_dim, num_layers, mlp_dims, dropout, add_text=False):
        super().__init__()
        self.embed_output_dim1 = len(field_dims1) * embed_dim
        self.embed_output_dim2 = len(field_dims2) * embed_dim

        self.cn = CrossNetwork(self.embed_output_dim1+self.embed_output_dim2+1, num_layers)
        self.cn_ = CrossNetwork(self.embed_output_dim1+self.embed_output_dim2, num_layers)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim1+self.embed_output_dim2+1, mlp_dims, dropout, output_layer=False)
        self.mlp_ = MultiLayerPerceptron(self.embed_output_dim1+self.embed_output_dim2, mlp_dims, dropout, output_layer=False)
        self.embedding1 = FeaturesEmbedding(field_dims1, embed_dim)
        self.linear_text = nn.Linear(12,1)
        self.embedding2 = FeaturesEmbedding(field_dims2, embed_dim)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim1+self.embed_output_dim2+1, 1)
        self.linear_ = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim1+self.embed_output_dim2, 1)

        self.backup1 = {}
        self.backup2 = {}

    def forward(self, x_b, x_u, text, train, epoch, loss_fct, label, adv=False, domain_similar=False, add_text=False, dssm=False):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x_b = self.embedding1(x_b).view(-1, self.embed_output_dim1)
        embed_x_u = self.embedding2(x_u).view(-1, self.embed_output_dim2)
        text_embed = self.linear_text(text)
        # print(embed_x_b.shape)
        # print(embed_x_u.shape)
        b = self.fm(self.embedding1(x_b))
        u = self.fm(self.embedding2(x_u))

        if add_text:
            embed_x_cat = torch.cat([embed_x_b, embed_x_u, text_embed],dim=1)

            x_l1 = self.cn(embed_x_cat)
            h_l2 = self.mlp(embed_x_cat)
        else:
            embed_x_cat = torch.cat([embed_x_b, embed_x_u],dim=1)
            # print(embed_x_cat.shape)
            x_l1 = self.cn_(embed_x_cat)
            h_l2 = self.mlp_(embed_x_cat)


        x_out = torch.cat([x_l1, h_l2], dim=1)

        if add_text:
            x_out = self.linear(x_out)
        else:
            x_out = self.linear_(x_out)
        out = torch.sigmoid(x_out.squeeze(1))


        if train:
            if epoch > 0 and adv:
                delta_dict = {}
                param_lst1 = ['embedding1.embedding.weight']
                param_lst2 = ['embedding2.embedding.weight']


                # loss_fct = nn.BCELoss()
                if domain_similar:
                    sim_loss_fct = torch.nn.CosineEmbeddingLoss()
                    device = torch.device('cuda:0')
                    y = torch.ones(len(b)).to(device)
                    loss_sim = sim_loss_fct(b, u, y)
                else:
                    loss_sim = 0

                loss = loss_fct(out.view(-1), label.float())
                # loss_fct.zero_grad()
                loss.backward(retain_graph=True)

                adv_loss_fct = nn.BCELoss(reduction='mean')

                num_feature1 = x_b.shape[1]
                num_feature2 = x_u.shape[1]
                perturbed_fea_id1 = []
                perturbed_fea_id2 = []

                for fea_id in range(num_feature1 - 1):
                    perturbed_fea_id1.append(fea_id)
                for fea_id in range(num_feature2 - 1):
                    perturbed_fea_id2.append(fea_id)

                # print(fix)
                for name, param in self.named_parameters():
                    if param.requires_grad and name in param_lst1:
                        # if name in param_lst:
                        self.backup1[name] = param.data.clone()
                        delta = nn.functional.normalize(param.grad, p=2, dim=0)
                        # print(param)
                        # print(param.requires_grad)
                        # print(param.grad)
                        # param.cuda()
                        # delta = nn.functional.normalize(param.grad, p=2, dim=0)
                        sub_delta_dict = {}
                        for i in range(param.shape[0]):
                            delta_i = delta[i, :]
                            sub_delta_dict[i] = delta_i
                        delta_dict[name] = sub_delta_dict
                        # print(name)
                # print(x)
                # embed_x = self.embedding(x)
                embed_b = self.embedding1(x_b)
                # embed_u = emb2(x_u)
                # embed_x_ = self.embedding(x)
                for fea_id in perturbed_fea_id1:
                    for i in range(embed_b[fea_id].shape[0]):
                        value = int(x_b[i, fea_id])
                        delta_i = delta_dict["embedding1.embedding.weight"][value]
                        # eps_delta_i = torch.clamp(delta_i * 0.5, -perturb_interval, perturb_interval)
                        eps_delta_i = torch.clamp(delta_i*0.5, -0.5, 0.5)

                        # eps_delta_i = torch.clamp(delta_i*0.5, -perturb_interval, perturb_interval)
                        # eps_delta_i = torch.clamp(delta_i, -0.3, 0.3)
                        # H = 100
                        # eps_delta_b = H * eps_delta_i
                        embed_b[fea_id][i, :] = embed_b[fea_id][i, :] +  0.5*eps_delta_i.unsqueeze(0)
                        # embed_x_[fea_id][i, :] = embed_x_[fea_id][i, :] + eps_delta_b.unsqueeze(0)



                for name, param in self.named_parameters():
                    if param.requires_grad and name in param_lst2:
                        # if name in param_lst:
                        self.backup2[name] = param.data.clone()
                        delta = nn.functional.normalize(param.grad, p=2, dim=0)
                        # print(param)
                        # print(param.requires_grad)
                        # print(param.grad)
                        # param.cuda()
                        # delta = nn.functional.normalize(param.grad, p=2, dim=0)
                        sub_delta_dict = {}
                        for i in range(param.shape[0]):
                            delta_i = delta[i, :]
                            sub_delta_dict[i] = delta_i
                        delta_dict[name] = sub_delta_dict
                        # print(name)
                # print(x)
                # embed_x = self.embedding(x)
                embed_u = self.embedding2(x_u)
                # embed_u = emb2(x_u)
                # embed_x_ = self.embedding(x)
                for fea_id in perturbed_fea_id2:
                    for i in range(embed_u[fea_id].shape[0]):
                        value = int(x_u[i, fea_id])
                        delta_i = delta_dict["embedding2.embedding.weight"][value]
                        # eps_delta_i = torch.clamp(delta_i * 0.5, -perturb_interval, perturb_interval)
                        eps_delta_i = torch.clamp(delta_i * 0.5, -0.5, 0.5)

                        # eps_delta_i = torch.clamp(delta_i*0.5, -perturb_interval, perturb_interval)
                        # eps_delta_i = torch.clamp(delta_i, -0.3, 0.3)
                        # H = 100
                        # eps_delta_b = H * eps_delta_i
                        embed_u[fea_id][i, :] = embed_u[fea_id][i, :] + 0.5 * eps_delta_i.unsqueeze(0)
                        # embed_x_[fea_id][i, :] = embed_x_[fea_id][i, :] + eps_delta_b.unsqueeze(0)

                embed_b = embed_b.view(-1, self.embed_output_dim1)
                embed_u = embed_u.view(-1, self.embed_output_dim2)
                if add_text:
                    embed_x_cat = torch.cat([embed_b, embed_u, text_embed], dim=1)
                    x_l1 = self.cn(embed_x_cat)
                    h_l2 = self.mlp(embed_x_cat)
                else:
                    embed_x_cat = torch.cat([embed_b, embed_u], dim=1)
                    x_l1 = self.cn_(embed_x_cat)
                    h_l2 = self.mlp_(embed_x_cat)

                x_out = torch.cat([x_l1, h_l2], dim=1)

                if add_text:
                    x_out = self.linear(x_out)
                else:
                    x_out = self.linear_(x_out)
                out = torch.sigmoid(x_out.squeeze(1))


                loss_adv = adv_loss_fct(out.view(-1), label.float())
                loss = loss_adv

                for name, param in self.named_parameters():
                    for emb_name in param_lst1:
                        if param.requires_grad and emb_name in name:
                            assert name in self.backup1
                            # print(self.backup)
                            param.data = self.backup1[name]

                for name, param in self.named_parameters():
                    for emb_name in param_lst2:
                        if param.requires_grad and emb_name in name:
                            assert name in self.backup2
                            # print(self.backup)
                            param.data = self.backup2[name]

                self.backup1 = {}
                self.backup2 = {}
            else:
                loss = loss_fct(out.view(-1), label.float())
                if domain_similar:
                    sim_loss_fct = torch.nn.CosineEmbeddingLoss()
                    device = torch.device('cuda:0')
                    y = torch.ones(len(b)).to(device)
                    loss_sim = sim_loss_fct(b, u, y)
                else:
                    loss_sim = 0

            return out, loss, loss_sim
        else:
            # return torch.sigmoid(x.squeeze(1))
            return out, 0, 0
