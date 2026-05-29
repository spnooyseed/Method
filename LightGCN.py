import torch
import torch.utils.data as dataloader
import numpy as np
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import LightGCN
from DataHandler import DataHandler
from DataHandler import TrnData
import numpy as np
import pickle
from Utils.Utils import *
from Utils.Utils import contrast
import os
import random
import setproctitle
from copy import deepcopy
from datetime import datetime
import time


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        log("LightGCN Model Prepared")
        if args.load_model:
            self.loadModel()
            stloc = len(self.metrics["TrainLoss"]) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log("LightGCN Model Initialized")
        curTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fileName = f"LightGCN_{args.data}-{curTime}_pid_{os.getppid()}"
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
            reses = self.trainEpoch(ep)
            log(self.makePrint("Train", ep, reses, tstFlag))
            with open("./Result/" + fileName + ".txt", "a") as f:
                f.write(f"{self.makePrintRes('Train', ep, reses)}\n")
            if tstFlag:
                with torch.no_grad():
                    reses = self.testEpoch()
                    with open("./Result/" + fileName + ".txt", "a") as f:
                        f.write(f"{self.makePrintRes('Test', ep, reses)}\n")
                log(self.makePrint("Test", ep, reses, tstFlag))
                # bestRes = reses if bestRes is None or reses['Recall10'] > bestRes['Recall10'] else bestRes
                if bestRes is None or reses["Recall10"] > bestRes["Recall10"]:
                    bestRes = reses
                    torch.save(self.model.state_dict(), f"LightGCN_{args.data}.ckpt")
            print()
        with torch.no_grad():
            reses = self.testEpoch()
            self.saveRecord(reses, fileName)
            if bestRes is None or reses["Recall10"] > bestRes["Recall10"]:
                bestRes = reses
                torch.save(self.model.state_dict(), f"LightGCN_{args.data}.ckpt")
        log(self.makePrint("Test", args.epoch, reses, True))
        log(self.makePrint("Best Result", args.epoch, bestRes, True))
        with open("./Result/" + fileName + ".txt", "a") as f:
            f.write(f"{self.makePrintRes('Test', ep, reses)}\n")
            f.write(f"{self.makePrintRes('Best Result', args.epoch, bestRes)}")

    def prepareModel(self):
        self.model = LightGCN().to(device)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.decay
        )

    def trainEpoch(self, ep):
        epLoss, epPreLoss, epreg_loss, epunformity_loss = 0, 0, 0, 0
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        steps = trnLoader.dataset.__len__() // args.batch

        self.model.train()

        for i, tem in enumerate(trnLoader):
            ancs, poss, neg = tem
            ancs = ancs.long().to(device)
            poss = poss.long().to(device)
            neg = neg.long().to(device)
            batch_idx = torch.cat([poss, neg], dim=0)
            batch_idx += args.user
            batch_idx = torch.cat([ancs, batch_idx], dim=0)
            batch_idx = torch.unique(batch_idx)
            usrEmbeds, itmEmbeds = self.model(
                self.handler.torchBiAdj, self.handler.torchD_1Aadj
            )
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[neg]
            ancs_unique = torch.unique(ancs)
            pos_unique = torch.unique(poss)
            BPRloss = torch.mean(
                -torch.log(
                    10e-6 + torch.sigmoid(pairPredict(ancEmbeds, posEmbeds, negEmbeds))
                )
            )
            regLoss = calcRegLoss_normal(ancEmbeds, posEmbeds, negEmbeds) * args.reg
            loss = BPRloss + regLoss
            epLoss += loss.item()
            epPreLoss += BPRloss.item()
            epreg_loss += regLoss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            log(
                "Step %d/%d: loss = %.1f, bprloss = %.1f, regloss =  %.1f "
                % (i, steps, loss.item(), BPRloss.item(), regLoss.item()),
                save=False,
                oneline=True,
            )
        ret = dict()
        ret["Loss"] = epLoss / steps
        ret["bprLoss"] = epPreLoss / steps
        ret["epreg_loss"] = epreg_loss / steps
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epLoss, epRecall10, epNdcg10, epRecall20, epNdcg20 = [0] * 5
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        self.model.eval()
        usrEmbeds, itmEmbeds = self.model(
            self.handler.torchBiAdj, self.handler.torchD_1Aadj
        )

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
