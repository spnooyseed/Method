import torch
import torch.utils.data as dataloader
import numpy as np
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Model import init, uniformInit, zeroinit
from Params import args
from DataHandler import DataHandler
from DataHandler import TrnData
import numpy as np
import pickle
from Utils.Utils import *
from Utils.Utils import contrast
import os
import random
import setproctitle
from datetime import datetime
import time
from torch import nn

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdaGCL(nn.Module):
    def __init__(self):
        super(AdaGCL, self).__init__()

        self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gcn_layer0)])

    def forward_gcn(self, adj):
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)

        return mainEmbeds[: args.user], mainEmbeds[args.user :]

    def forward_graphcl(self, adj):
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    def forward_graphcl_(self, generator):
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        count = 0
        for gcn in self.gcnLayers:
            with torch.no_grad():
                adj = generator.generate(x=embedsLst[-1], layer=count)
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
            count += 1
        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    def loss_graphcl(self, x1, x2, users, items):
        T = args.temp
        user_embeddings1, item_embeddings1 = torch.split(
            x1, [args.user, args.item], dim=0
        )
        user_embeddings2, item_embeddings2 = torch.split(
            x2, [args.user, args.item], dim=0
        )

        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
        all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

        all_embs1_abs = all_embs1.norm(dim=1)
        all_embs2_abs = all_embs2.norm(dim=1)

        sim_matrix = torch.einsum("ik,jk->ij", all_embs1, all_embs2) / torch.einsum(
            "i,j->ij", all_embs1_abs, all_embs2_abs
        )
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[
            np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])
        ]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss)

        return loss

    def getEmbeds(self):
        self.unfreeze(self.gcnLayers)
        return torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

    def unfreeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = True

    def getGCN(self):
        return self.gcnLayers


class vgae_encoder(AdaGCL):
    def __init__(self):
        super(vgae_encoder, self).__init__()
        hidden = args.latdim
        self.encoder_mean = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden)
        )
        self.encoder_std = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
        )

    def forward(self, adj):
        x = self.forward_graphcl(adj)

        x_mean = self.encoder_mean(x)
        x_std = self.encoder_std(x)
        gaussian_noise = torch.randn(x_mean.shape).cuda()
        x = gaussian_noise * x_std + x_mean
        return x, x_mean, x_std


class vgae_decoder(nn.Module):
    def __init__(self, hidden=args.latdim):
        super(vgae_decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction="none")

    def forward(self, x, x_mean, x_std, users, items, neg_items, encoder):
        x_user, x_item = torch.split(x, [args.user, args.item], dim=0)

        edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
        edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))
        print(
            f"edge_pos_pred 范围: min={edge_pos_pred.min().item()}, max={edge_pos_pred.max().item()}"
        )
        print(
            f"edge_neg_pred 范围: min={edge_neg_pred.min().item()}, max={edge_neg_pred.max().item()}"
        )
        print(f"edge_pos_pred 是否有 nan: {torch.isnan(edge_pos_pred).any().item()}")
        print(f"edge_pos_pred 是否有 inf: {torch.isinf(edge_pos_pred).any().item()}")

        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones_like(edge_pos_pred))
        loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros_like(edge_neg_pred))
        loss_rec = loss_edge_pos + loss_edge_neg

        kl_divergence = -0.5 * (
            1 + 2 * torch.log(x_std) - x_mean**2 - x_std**2
        ).sum(dim=1)

        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = -(scoreDiff).sigmoid().log().sum() / args.batch
        regLoss = calcRegLoss(encoder) * args.reg

        beta = 0.1
        loss = (loss_rec + beta * kl_divergence.mean() + bprLoss + regLoss).mean()

        return loss


class vgae(nn.Module):
    def __init__(self, encoder, decoder):
        super(vgae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, users, items, neg_items):
        x, x_mean, x_std = self.encoder(data)
        loss = self.decoder(x, x_mean, x_std, users, items, neg_items, self.encoder)
        return loss

    def generate(self, data, edge_index, adj):
        x, _, _ = self.encoder(data)

        edge_pred = self.decoder.sigmoid(
            self.decoder.decoder(x[edge_index[0]] * x[edge_index[1]])
        )

        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        edge_pred = edge_pred[:, 0]
        mask = ((edge_pred + 0.5).floor()).type(torch.bool)

        newVals = vals[mask]

        newVals = newVals / (newVals.shape[0] / edgeNum[0])
        newIdxs = idxs[:, mask]

        return torch.torch.sparse_coo_tensor(newIdxs, newVals, adj.shape)


class DenoisingNet(nn.Module):
    def __init__(self, gcnLayers, features):
        super(DenoisingNet, self).__init__()

        self.features = features

        self.gcnLayers = gcnLayers

        self.edge_weights = []
        self.nblayers = []
        self.selflayers = []

        self.attentions = []
        self.attentions.append([])
        self.attentions.append([])

        hidden = args.latdim

        self.nblayers_0 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )
        self.nblayers_1 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )

        self.selflayers_0 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )
        self.selflayers_1 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )

        self.attentions_0 = nn.Sequential(nn.Linear(2 * hidden, 1))
        self.attentions_1 = nn.Sequential(nn.Linear(2 * hidden, 1))

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    def get_attention(self, input1, input2, layer=0):
        if layer == 0:
            nb_layer = self.nblayers_0
            selflayer = self.selflayers_0
        if layer == 1:
            nb_layer = self.nblayers_1
            selflayer = self.selflayers_1

        input1 = nb_layer(input1)
        input2 = selflayer(input2)

        input10 = torch.concat([input1, input2], axis=1)

        if layer == 0:
            weight10 = self.attentions_0(input10)
        if layer == 1:
            weight10 = self.attentions_1(input10)

        return weight10

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        gamma = args.gamma
        zeta = args.zeta

        if training:
            debug_var = 1e-7
            bias = 0.0
            np_random = np.random.uniform(
                low=debug_var,
                high=1.0 - debug_var,
                size=np.shape(log_alpha.cpu().detach().numpy()),
            )
            random_noise = bias + torch.tensor(np_random)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.cuda() + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta - gamma) + gamma
        cliped = torch.clamp(stretched_values, 0.0, 1.0)
        return cliped.float()

    def generate(self, x, layer=0):
        f1_features = x[self.row, :]
        f2_features = x[self.col, :]

        weight = self.get_attention(f1_features, f2_features, layer)

        mask = self.hard_concrete_sample(weight, training=False)

        mask = torch.squeeze(mask)
        adj = torch.sparse_coo_tensor(self.adj_mat._indices(), mask, self.adj_mat.shape)

        ind = adj._indices()
        row = ind[0, :]
        col = ind[1, :]

        rowsum = torch.sparse.sum(adj, dim=-1).to_dense()
        d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
        d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
        row_inv_sqrt = d_inv_sqrt[row]
        col_inv_sqrt = d_inv_sqrt[col]
        values = torch.mul(adj._values(), row_inv_sqrt)
        values = torch.mul(values, col_inv_sqrt)

        support = torch.sparse_coo_tensor(adj._indices(), values, adj.shape)

        return support

    def l0_norm(self, log_alpha, beta):
        gamma = args.gamma
        zeta = args.zeta
        gamma = torch.tensor(gamma)
        zeta = torch.tensor(zeta)
        reg_per_weight = torch.sigmoid(log_alpha - beta * torch.log(-gamma / zeta))

        return torch.mean(reg_per_weight)

    def set_fea_adj(self, nodes, adj):
        self.node_size = nodes
        self.adj_mat = adj

        ind = adj._indices()

        self.row = ind[0, :]
        self.col = ind[1, :]

    def call(self, inputs, training=None):
        if training:
            temperature = inputs
        else:
            temperature = 1.0

        self.maskes = []

        x = self.features.detach()
        layer_index = 0
        embedsLst = [self.features.detach()]

        for layer in self.gcnLayers:
            xs = []
            f1_features = x[self.row, :]
            f2_features = x[self.col, :]

            weight = self.get_attention(f1_features, f2_features, layer=layer_index)
            mask = self.hard_concrete_sample(weight, temperature, training)

            self.edge_weights.append(weight)
            self.maskes.append(mask)
            mask = torch.squeeze(mask)

            adj = torch.sparse_coo_tensor(
                self.adj_mat._indices(), mask, self.adj_mat.shape
            ).coalesce()
            ind = adj._indices()
            row = ind[0, :]
            col = ind[1, :]

            rowsum = torch.sparse.sum(adj, dim=-1).to_dense() + 1e-6
            d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
            d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
            row_inv_sqrt = d_inv_sqrt[row]
            col_inv_sqrt = d_inv_sqrt[col]
            values = torch.mul(adj.values(), row_inv_sqrt)
            values = torch.mul(values, col_inv_sqrt)
            support = torch.sparse_coo_tensor(
                adj._indices(), values, adj.shape
            ).coalesce()

            nextx = layer(support, x, False)
            xs.append(nextx)
            x = xs[0]
            embedsLst.append(x)
            layer_index += 1
        return sum(embedsLst)

    def lossl0(self, temperature):
        l0_loss = torch.zeros([]).cuda()
        for weight in self.edge_weights:
            l0_loss += self.l0_norm(weight, temperature)
        self.edge_weights = []
        return l0_loss

    def forward(self, users, items, neg_items, temperature):
        self.freeze(self.gcnLayers)
        x = self.call(temperature, True)
        x_user, x_item = torch.split(x, [args.user, args.item], dim=0)
        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = -(scoreDiff).sigmoid().log().sum() / args.batch
        regLoss = calcRegLoss(self) * args.reg

        lossl0 = self.lossl0(temperature) * args.lambda0
        return bprLoss + regLoss + lossl0


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds, flag=True):
        # if (flag):
        return torch.spmm(adj, embeds)
        # else:
        #     return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)


class Coach:
    def __init__(self, handler):
        self.handler = handler

        print("USER", args.user, "ITEM", args.item)
        print("NUM OF INTERACTIONS", self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ["Loss", "preLoss", "Recall", "NDCG"]
        for met in mets:
            self.metrics["Train" + met] = list()
            self.metrics["Test" + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = "Epoch %d/%d, %s: " % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += "%s = %.4f, " % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + "  "
        return f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} ' + ret

    def makePrintRes(self, name, ep, reses):
        ret = "Epoch %d/%d, %s: " % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += "%s = %.4f, " % (metric, val)
        ret = ret[:-2] + "  "
        return f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} ' + ret

    def run(self):
        self.prepareModel()
        log("AdaGCL Model Prepared")
        if args.load_model:
            self.loadModel()
            stloc = len(self.metrics["TrainLoss"]) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log("AdaGCL Model Initialized")
        curTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fileName = f"AdaGCL_{args.data}-{curTime}_pid_{os.getppid()}"
        with open("./Result/" + fileName + ".txt", "w") as f:
            hypeParameters = vars(args)
            with open(__file__, "r", encoding="utf-8") as ff:
                content = ff.read()
                f.write(content)
            with open("./Model.py", "r", encoding="utf-8") as ff:
                content = ff.read()
                f.write(content)
            with open("./DataHandler.py", "r", encoding="utf-8") as ff:
                content = ff.read()
                f.write(content)
            f.write("HyperParameters:\n")
            for k, v in hypeParameters.items():
                f.write(f"{k}: {v}\n")
        bestRes = None
        for ep in range(stloc, args.epoch):
            tstFlag = ep % args.tstEpoch == 0
            temperature = max(
                0.05, args.init_temperature * pow(args.temperature_decay, ep)
            )
            reses = self.trainEpoch(ep, temperature)
            log(self.makePrint("Train", ep, reses, tstFlag))
            with open("./Result/" + fileName + ".txt", "a") as f:
                f.write(f"{self.makePrintRes('Train', ep, reses)}\n")
            if tstFlag:
                with torch.no_grad():
                    reses = self.testEpoch()
                    with open("./Result/" + fileName + ".txt", "a") as f:
                        f.write(f"{self.makePrintRes('Test', ep, reses)}\n")
                log(self.makePrint("Test", ep, reses, tstFlag))
                bestRes = (
                    reses
                    if bestRes is None or reses["Recall10"] > bestRes["Recall10"]
                    else bestRes
                )
            print()
        with torch.no_grad():
            reses = self.testEpoch()
            self.saveRecord(reses, fileName)
            bestRes = (
                reses
                if bestRes is None or reses["Recall10"] > bestRes["Recall10"]
                else bestRes
            )

        log(self.makePrint("Test", args.epoch, reses, True))
        log(self.makePrint("Best Result", args.epoch, bestRes, True))
        with open("./Result/" + fileName + ".txt", "a") as f:
            f.write(f"{self.makePrintRes('Test', ep, reses)}\n")
            f.write(f"{self.makePrintRes('Best Result', args.epoch, bestRes)}")

    def prepareModel(self):
        self.model = AdaGCL().cuda()
        encoder = vgae_encoder().cuda()
        decoder = vgae_decoder().cuda()
        self.generator_1 = vgae(encoder, decoder).cuda()
        self.generator_2 = DenoisingNet(
            self.model.getGCN(), self.model.getEmbeds()
        ).cuda()
        self.generator_2.set_fea_adj(args.user + args.item, self.handler.torchBiAdj)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.opt_gen_1 = torch.optim.Adam(
            self.generator_1.parameters(), lr=args.lr, weight_decay=0
        )
        self.opt_gen_2 = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.generator_2.parameters()),
            lr=args.lr,
            weight_decay=0,
            eps=args.eps,
        )

    def trainEpoch(self, ep, temperature):
        generate_loss_1, generate_loss_2, bpr_loss, im_loss, ib_loss, reg_loss = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        steps = trnLoader.dataset.__len__() // args.batch
        self.model.train()

        for i, tem in enumerate(trnLoader):
            data = self.handler.torchBiAdj

            data1 = self.generator_generate(self.generator_1)

            self.opt.zero_grad()
            self.opt_gen_1.zero_grad()
            self.opt_gen_2.zero_grad()

            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()

            out1 = self.model.forward_graphcl(data1)
            out2 = self.model.forward_graphcl_(self.generator_2)

            loss = self.model.loss_graphcl(out1, out2, ancs, poss).mean() * args.ssl_reg
            im_loss += float(loss)
            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            # info bottleneck
            _out1 = self.model.forward_graphcl(data1)
            _out2 = self.model.forward_graphcl_(self.generator_2)

            loss_ib = self.model.loss_graphcl(
                _out1, out1.detach(), ancs, poss
            ) + self.model.loss_graphcl(_out2, out2.detach(), ancs, poss)
            loss = loss_ib.mean() * args.ib_reg
            ib_loss += float(loss)
            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            # BPR
            usrEmbeds, itmEmbeds = self.model.forward_gcn(data)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]
            scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
            bprLoss = -(scoreDiff).sigmoid().log().sum() / args.batch
            regLoss = calcRegLoss(self.model) * args.reg
            loss = bprLoss + regLoss
            bpr_loss += float(bprLoss)
            reg_loss += float(regLoss)
            loss.backward()

            loss_1 = self.generator_1(self.handler.torchBiAdj, ancs, poss, negs)
            loss_2 = self.generator_2(ancs, poss, negs, temperature)

            loss = loss_1 + loss_2
            generate_loss_1 += float(loss_1)
            generate_loss_2 += float(loss_2)
            loss.backward()

            self.opt.step()
            self.opt_gen_1.step()
            self.opt_gen_2.step()

            log(
                "Step %d/%d: gen 1 : %.3f ; gen 2 : %.3f ; bpr : %.3f ; im : %.3f ; ib : %.3f ; reg : %.3f  "
                % (
                    i,
                    steps,
                    generate_loss_1,
                    generate_loss_2,
                    bpr_loss,
                    im_loss,
                    ib_loss,
                    reg_loss,
                ),
                save=False,
                oneline=True,
            )
            torch.cuda.empty_cache()
        ret = dict()
        ret["Gen_1 Loss"] = generate_loss_1 / steps
        ret["Gen_2 Loss"] = generate_loss_2 / steps
        ret["BPR Loss"] = bpr_loss / steps
        ret["IM Loss"] = im_loss / steps
        ret["IB Loss"] = ib_loss / steps
        ret["Reg Loss"] = reg_loss / steps

        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epLoss, epRecall10, epNdcg10, epRecall20, epNdcg20 = [0] * 5
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        self.model.eval()
        usrEmbeds, itmEmbeds = self.model.forward_gcn(self.handler.torchBiAdj)

        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().to(device)
            trnMask = trnMask.to(device)
            allPreds = (
                torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0))
                * (1 - trnMask)
                - trnMask * 1e8
            )
            _, topLocs10 = torch.topk(allPreds, 10)
            _, topLocs20 = torch.topk(allPreds, 20)
            recall10, ndcg10 = self.calcRes(
                topLocs10.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr, 10
            )
            recall20, ndcg20 = self.calcRes(
                topLocs20.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr, 20
            )
            epRecall10 += recall10
            epNdcg10 += ndcg10
            epRecall20 += recall20
            epNdcg20 += ndcg20
            log(
                "Steps %d/%d: recall = %.1f, ndcg = %.1f          "
                % (i, steps, recall10, ndcg10),
                save=False,
                oneline=True,
            )
            log(
                "Steps %d/%d: recall = %.1f, ndcg = %.1f          "
                % (i, steps, recall20, ndcg20),
                save=False,
                oneline=True,
            )
        ret = dict()
        ret["Recall10"] = epRecall10 / num
        ret["NDCG10"] = epNdcg10 / num
        ret["Recall20"] = epRecall20 / num
        ret["NDCG20"] = epNdcg20 / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds, topk):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum(
                [np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, topk))]
            )
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def loadModel(self):
        ckp = torch.load("./Models/" + args.load_model + ".mod")
        self.model = ckp["model"]
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open("./History/" + args.load_model + ".his", "rb") as fs:
            self.metrics = pickle.load(fs)
        log("Model Loaded")

    def saveRecord(self, reses, fileName):
        pass

    def generator_generate(self, generator):
        edge_index = []
        edge_index.append([])
        edge_index.append([])
        adj = self.handler.torchBiAdj
        idxs = adj._indices()

        with torch.no_grad():
            view = generator.generate(self.handler.torchBiAdj, idxs, adj)

        return view


if __name__ == "__main__":
    logger.saveDefault = True
    log("Start")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    handler = DataHandler()
    handler.LoadData()
    log("Load Data")
    coach = Coach(handler)
    coach.run()
