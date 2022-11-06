"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2022/4/13
@description: null
"""
import math
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from toolbox.data.DataSchema import RelationalTripletData, RelationalTripletDatasetCachePath
from toolbox.data.DatasetSchema import get_dataset
from toolbox.data.LinkPredictDataset import LinkPredictDataset, LinkPredictTypeConstraintDataset
from toolbox.data.ScoringAllDataset import ScoringAllDataset
from toolbox.data.functional import with_inverse_relations, build_map_hr_t
from toolbox.evaluate.Evaluate import get_score
from toolbox.evaluate.LinkPredict import batch_link_predict2, as_result_dict2, batch_link_predict_type_constraint
from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.optim.lr_scheduler import get_scheduler
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds


class LorentzE(nn.Module):
    def __init__(self, embedding_dim, entities_num, relations_num, drop_out, c=1.0):
        super(LorentzE, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num

        self.E_ct = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_x = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_y = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_z = nn.Embedding(self.entities_num, self.embedding_dim)

        # self.R_r = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_x = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_y = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_z = nn.Embedding(self.relations_num, self.embedding_dim)

        self.E_bn = nn.BatchNorm1d(self.embedding_dim)
        self.R_bn = nn.BatchNorm1d(self.embedding_dim)
        self.b_a = nn.Parameter(torch.zeros(entities_num))
        self.b_x = nn.Parameter(torch.zeros(entities_num))
        self.b_y = nn.Parameter(torch.zeros(entities_num))
        self.b_z = nn.Parameter(torch.zeros(entities_num))

        # self.c_embedding = nn.Embedding(1, self.embedding_dim)
        self.drop_out = nn.Dropout(drop_out)
        self.c = c
        self.bce = nn.BCELoss()
        # self.bias = 0.01

    def init(self):
        nn.init.xavier_normal_(self.E_ct.weight.data)
        nn.init.xavier_normal_(self.E_x.weight.data)
        nn.init.xavier_normal_(self.E_y.weight.data)
        nn.init.xavier_normal_(self.E_z.weight.data)
        nn.init.xavier_normal_(self.R_x.weight.data)
        nn.init.xavier_normal_(self.R_y.weight.data)
        nn.init.xavier_normal_(self.R_z.weight.data)
        # nn.init.xavier_normal_(self.R_r.weight.data)
        # nn.init.xavier_normal_(self.R_theta.weight.data)
        # nn.init.xavier_normal_(self.R_phi.weight.data)

    def forward(self, head, rel):
        head = head.view(-1)
        rel = rel.view(-1)
        h_ct = self.E_ct(head)
        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)

        h_ct = self.E_bn(h_ct)
        h_x = self.E_bn(h_x)
        h_y = self.E_bn(h_y)
        h_z = self.E_bn(h_z)
        # print("h:", h_ct.shape, h_x.shape, h_y.shape, h_z.shape)

        # r_r = self.R_r(rel)
        r_x = self.R_x(rel)
        r_y = self.R_y(rel)
        r_z = self.R_z(rel)
        length = torch.sqrt(r_x **2 + r_y**2 + r_z**2).detach()
        # r_theta = self.R_theta(rel)
        # r_phi = self.R_phi(rel)
        # print("r:", r_r.shape, r_theta.shape, r_phi.shape)

        r_v_rate = torch.sigmoid(length)
        # r_v = r_v_rate * self.c

        r_x = r_x / length
        r_y = r_y / length
        r_z = r_z / length

        r_v_rate_2 = r_v_rate * r_v_rate
        gamma = 1.0 / (torch.sqrt(1.0 - r_v_rate_2))

        t_ct = gamma * h_ct + gamma * r_x * r_v_rate * h_x + gamma * r_y * r_v_rate * h_y + gamma * r_z * r_v_rate * h_z

        t_x = r_x * r_v_rate * gamma * h_ct + \
              (1 + (r_x * r_x * (gamma - 1))) * h_x + \
              (r_x * r_y * (gamma - 1)) * h_y + \
              (r_x * r_z * (gamma - 1)) * h_z

        t_y = r_y * r_v_rate * gamma * h_ct + \
              (r_x * r_y * (gamma - 1)) * h_x + \
              (1 + (r_y * r_y * (gamma - 1))) * h_y + \
              (r_z * r_y * (gamma - 1)) * h_z

        t_z = r_z * r_v_rate * gamma * h_ct + \
              (r_x * r_z * (gamma - 1)) * h_x + \
              (r_y * r_z * (gamma - 1)) * h_y + \
              (1 + (r_z * r_z * (gamma - 1))) * h_z
        # print("t:", t_ct.shape, t_x.shape, t_y.shape, t_z.shape)

        t_ct = self.drop_out(t_ct)
        t_x = self.drop_out(t_x)
        t_y = self.drop_out(t_y)
        t_z = self.drop_out(t_z)

        E_ct = self.drop_out(self.E_bn(self.E_ct.weight))
        E_x = self.drop_out(self.E_bn(self.E_x.weight))
        E_y = self.drop_out(self.E_bn(self.E_y.weight))
        E_z = self.drop_out(self.E_bn(self.E_z.weight))

        score_ct = torch.mm(t_ct, E_ct.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x, E_x.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y, E_y.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z, E_z.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_ct = torch.sigmoid(score_ct + self.b_a.expand_as(score_ct))
        score_x = torch.sigmoid(score_x + self.b_x.expand_as(score_x))
        score_y = torch.sigmoid(score_y + self.b_y.expand_as(score_y))
        score_z = torch.sigmoid(score_z + self.b_z.expand_as(score_z))
        # score = (score_ct + score_x + score_y + score_z) / 4
        return score_ct, score_x, score_y, score_z

    def loss(self, target, y):
        y_a, y_ai, y_b, y_bi = target
        return self.bce(y_a, y) + self.bce(y_ai, y) + self.bce(y_b, y) + self.bce(y_bi, y)

class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema, data: RelationalTripletData,
                 start_step, max_steps, every_test_step, every_valid_step,
                 batch_size, test_batch_size, sampling_window_size, label_smoothing,
                 train_device, test_device,
                 resume, resume_by_score,
                 lr, amsgrad, lr_decay, weight_decay,
                 edim, rdim, input_dropout, hidden_dropout,
                 ):
        super(MyExperiment, self).__init__(output)
        self.log(f"{locals()}")

        self.model_param_store.save_scripts([__file__])
        hyper = {
            'learning_rate': lr,
            'batch_size': batch_size,
            "edim": edim,
            "rdim": rdim,
        }
        self.metric_log_store.add_hyper(hyper)
        for k, v in hyper.items():
            self.log(f'{k} = {v}')
        self.metric_log_store.add_progress(max_steps)

        data.load_cache(["train_triples_ids", "test_triples_ids", "valid_triples_ids", "all_triples_ids"])
        data.load_cache(["hr_t_train"])
        data.print(self.log)
        max_relation_id = data.relation_count

        # 1. build train dataset
        train_triples, _, _ = with_inverse_relations(data.train_triples_ids, max_relation_id)
        train_data = ScoringAllDataset(train_triples, data.entity_count)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        # 2. build valid and test dataset
        all_triples, _, _ = with_inverse_relations(data.all_triples_ids, max_relation_id)
        tail_type_constraint = defaultdict(set)
        for h, r, t in all_triples:
            tail_type_constraint[r].add(t)
        hr_t = build_map_hr_t(all_triples)
        valid_data = LinkPredictDataset(data.valid_triples_ids, hr_t, max_relation_id, data.entity_count)
        test_data = LinkPredictDataset(data.test_triples_ids, hr_t, max_relation_id, data.entity_count)
        test_type_constraint_data = LinkPredictTypeConstraintDataset(data.test_triples_ids, tail_type_constraint, hr_t, max_relation_id, data.entity_count)
        valid_dataloader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_type_constraint_dataloader = DataLoader(test_type_constraint_data, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # 3. build model
        print('rel_num_out: ', data.relation_count)
        # model = QubitE(data.entity_count, 2 * data.relation_count, edim, input_dropout=input_dropout, hidden_dropout=hidden_dropout).to(train_device)
        model = LorentzE(edim, data.entity_count, data.relation_count * 2, input_dropout).to(train_device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
        scheduler = get_scheduler(opt, lr_policy="step")
        best_score = 0
        best_test_score = 0
        if resume:
            if resume_by_score > 0:
                start_step, _, best_score = self.model_param_store.load_by_score(model, opt, resume_by_score)
            else:
                start_step, _, best_score = self.model_param_store.load_best(model, opt)
            self.dump_model(model)
            model.eval()
            with torch.no_grad():
                self.debug("Resumed from score %.4f." % best_score)
                self.debug("Take a look at the performance after resumed.")
                self.debug("Validation (step: %d):" % start_step)
                result = self.evaluate(model, valid_data, valid_dataloader, test_batch_size, test_device)
                best_score = self.visual_result(start_step + 1, result, "Valid")
                self.debug("Test (step: %d):" % start_step)
                result = self.evaluate(model, test_data, test_dataloader, test_batch_size, test_device)
                self.evaluate_with_type_constraint(model, test_type_constraint_data, test_type_constraint_dataloader, test_batch_size, test_device)
                best_test_score = self.visual_result(start_step + 1, result, "Test")
        else:
            model.init()
            self.dump_model(model)

        # 4. training
        self.debug("training")
        progbar = Progbar(max_step=max_steps)
        for step in range(start_step, max_steps):
            model.train()
            losses = []
            for h, r, targets in train_dataloader:
                opt.zero_grad()

                h = h.to(train_device)
                r = r.to(train_device)
                targets = targets.to(train_device).float()
                if label_smoothing:
                    targets = ((1.0 - label_smoothing) * targets) + (1.0 / targets.size(1))

                predictions = model(h, r)
                loss = model.loss(predictions, targets)
                # print(loss)
                # loss = loss + model.regular_loss(h, r)
                losses.append(loss.item())
                loss.backward()
                opt.step()
            # scheduler.step(step + 1)
            scheduler.step()

            log = {
                "loss": torch.mean(torch.Tensor(losses)).item(),
                "lr": torch.mean(torch.Tensor(scheduler.get_last_lr())).item(),
            }

            progbar.update(step + 1, [("step", step + 1), ("loss", log["loss"]), ("lr", log["lr"])])
            for metric in log:
                self.vis.add_scalar(metric, log[metric], step)
            self.metric_log_store.add_loss(log, step + 1)

            if (step + 1) % every_valid_step == 0:
                model.eval()
                with torch.no_grad():
                    print("")
                    self.debug("Validation (step: %d):" % (step + 1))
                    result = self.evaluate(model, valid_data, valid_dataloader, test_batch_size, test_device)
                    score = self.visual_result(step + 1, result, "Valid")
                    if score >= best_score:
                        self.success("current score=%.4f > best score=%.4f" % (score, best_score))
                        best_score = score
                        self.debug("saving best score %.4f" % score)
                        self.metric_log_store.add_best_metric({"result": result}, "Valid")
                        self.model_param_store.save_best(model, opt, step, 0, score)
                    else:
                        self.model_param_store.save_by_score(model, opt, step, 0, score)
                        self.fail("current score=%.4f < best score=%.4f" % (score, best_score))
            if (step + 1) % every_test_step == 0:
                model.eval()
                with torch.no_grad():
                    print("")
                    self.debug("Test (step: %d):" % (step + 1))
                    result = self.evaluate(model, test_data, test_dataloader, test_batch_size, test_device)
                    self.evaluate_with_type_constraint(model, test_type_constraint_data, test_type_constraint_dataloader, test_batch_size, test_device)
                    score = self.visual_result(step + 1, result, "Test")
                    if score >= best_test_score:
                        best_test_score = score
                        self.metric_log_store.add_best_metric({"result": result}, "Test")
                    print("")
        # 5. report the best
        start_step, _, best_score = self.model_param_store.load_best(model, opt)
        model.eval()
        with torch.no_grad():
            self.debug("Reporting the best performance...")
            self.debug("Resumed from score %.4f." % best_score)
            self.debug("Validation (step: %d):" % start_step)
            self.evaluate(model, valid_data, valid_dataloader, test_batch_size, test_device)
            self.debug("Test (step: %d):" % start_step)
            self.evaluate(model, test_data, test_dataloader, test_batch_size, test_device)
            self.final_result = self.evaluate_with_type_constraint(model, test_type_constraint_data, test_type_constraint_dataloader, test_batch_size, test_device)
        self.metric_log_store.finish()

    def evaluate_with_type_constraint(self, model, test_data, test_dataloader, test_batch_size, device="cuda:0"):
        self.log("with type constraint")
        data = iter(test_dataloader)

        def predict(i):
            h, r, mask_for_hr, t, reverse_r, mask_for_tReverser = next(data)
            h = h.to(device)
            r = r.to(device)
            mask_for_hr = mask_for_hr.to(device)
            t = t.to(device)
            reverse_r = reverse_r.to(device)
            mask_for_tReverser = mask_for_tReverser.to(device)
            pred_tail = model(h, r)
            pred_head = model(t, reverse_r)
            pred_tail = (pred_tail[0] + pred_tail[1] + pred_tail[2] + pred_tail[3]) / 2
            pred_head = (pred_head[0] + pred_head[1] + pred_head[2] + pred_head[3]) / 2
            return t, h, pred_tail, pred_head, mask_for_hr, mask_for_tReverser

        progbar = Progbar(max_step=len(test_data) // (test_batch_size * 5))

        def log(i, hits, hits_left, hits_right, ranks, ranks_left, ranks_right):
            if i % (test_batch_size * 5) == 0:
                progbar.update(i // (test_batch_size * 5), [("Hits @10", np.mean(hits[9]))])

        hits, hits_left, hits_right, ranks, ranks_left, ranks_right = batch_link_predict_type_constraint(test_batch_size, len(test_data), predict, log)
        result = as_result_dict2((hits, hits_left, hits_right, ranks, ranks_left, ranks_right))
        for i in (0, 2, 9):
            self.log('Hits @{0:2d}: {1:2.2%}    left: {2:2.2%}    right: {3:2.2%}'.format(i + 1, np.mean(hits[i]), np.mean(hits_left[i]), np.mean(hits_right[i])))
        self.log('Mean rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(ranks), np.mean(ranks_left), np.mean(ranks_right)))
        self.log('Mean reciprocal rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(1. / np.array(ranks)), np.mean(1. / np.array(ranks_left)), np.mean(1. / np.array(ranks_right))))
        return result

    def evaluate(self, model, test_data, test_dataloader, test_batch_size, device="cuda:0"):
        self.log("without type constraint")
        data = iter(test_dataloader)

        def predict(i):
            h, r, mask_for_hr, t, reverse_r, mask_for_tReverser = next(data)
            h = h.to(device)
            r = r.to(device)
            mask_for_hr = mask_for_hr.to(device)
            t = t.to(device)
            reverse_r = reverse_r.to(device)
            mask_for_tReverser = mask_for_tReverser.to(device)
            pred_tail = model(h, r)
            pred_head = model(t, reverse_r)
            pred_tail = (pred_tail[0] + pred_tail[1] + pred_tail[2] + pred_tail[3]) / 2
            pred_head = (pred_head[0] + pred_head[1] + pred_head[2] + pred_head[3]) / 2
            return t, h, pred_tail, pred_head, mask_for_hr, mask_for_tReverser

        progbar = Progbar(max_step=len(test_data) // (test_batch_size * 5))

        def log(i, hits, hits_left, hits_right, ranks, ranks_left, ranks_right):
            if i % (test_batch_size * 5) == 0:
                progbar.update(i // (test_batch_size * 5), [("Hits @10", np.mean(hits[9]))])

        hits, hits_left, hits_right, ranks, ranks_left, ranks_right = batch_link_predict2(test_batch_size, len(test_data), predict, log)
        result = as_result_dict2((hits, hits_left, hits_right, ranks, ranks_left, ranks_right))
        for i in (0, 2, 9):
            self.log('Hits @{0:2d}: {1:2.2%}    left: {2:2.2%}    right: {3:2.2%}'.format(i + 1, np.mean(hits[i]), np.mean(hits_left[i]), np.mean(hits_right[i])))
        self.log('Mean rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(ranks), np.mean(ranks_left), np.mean(ranks_right)))
        self.log('Mean reciprocal rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(1. / np.array(ranks)), np.mean(1. / np.array(ranks_left)), np.mean(1. / np.array(ranks_right))))
        return result

    def visual_result(self, step_num: int, result, scope: str):
        average = result["average"]
        left2right = result["left2right"]
        right2left = result["right2left"]
        sorted(average)
        sorted(left2right)
        sorted(right2left)
        for i in average:
            self.vis.add_scalar(scope + i, average[i], step_num)
        for i in left2right:
            self.vis.add_scalar(scope + i, left2right[i], step_num)
        for i in right2left:
            self.vis.add_scalar(scope + i, right2left[i], step_num)
        score = get_score(result)
        return score


@click.command()
@click.option("--dataset", type=str, default="WN18RR", help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
@click.option("--name", type=str, default="LorentzE", help="Name of the experiment.")
@click.option("--start_step", type=int, default=0, help="start step.")
@click.option("--max_steps", type=int, default=200, help="Number of steps.")
@click.option("--every_test_step", type=int, default=10, help="Number of steps.")
@click.option("--every_valid_step", type=int, default=5, help="Number of steps.")
@click.option("--batch_size", type=int, default=128, help="Batch size.")
@click.option("--test_batch_size", type=int, default=64, help="Test batch size.")
@click.option("--sampling_window_size", type=int, default=1000, help="Sampling window size.")
@click.option("--label_smoothing", type=float, default=0.1, help="Amount of label smoothing.")
@click.option("--train_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--test_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--resume", type=bool, default=False, help="Resume from output directory.")
@click.option("--resume_by_score", type=float, default=0.0, help="Resume by score from output directory. Resume best if it is 0. Default: 0")
@click.option("--lr", type=float, default=0.001, help="Learning rate.")
@click.option("--amsgrad", type=bool, default=False, help="AMSGrad for Adam.")
@click.option("--lr_decay", type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
@click.option('--weight_decay', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
@click.option("--edim", type=int, default=200, help="Entity embedding dimensionality.")
@click.option("--rdim", type=int, default=200, help="Relation embedding dimensionality.")
@click.option("--input_dropout", type=float, default=0.1, help="Input layer dropout.")
@click.option("--hidden_dropout", type=float, default=0.1, help="Dropout after the first hidden layer.")
@click.option("--times", type=int, default=1, help="Run multi times to get error bars.")
def main(dataset, name,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, test_batch_size, sampling_window_size, label_smoothing,
         train_device, test_device,
         resume, resume_by_score,
         lr, amsgrad, lr_decay, weight_decay,
         edim, rdim, input_dropout, hidden_dropout, times
         ):
    output = OutputSchema(dataset + "-" + name)
    data_home = Path.home() / "data"
    if dataset == "all":
        datasets = [get_dataset(i, data_home) for i in ["FB15k", "FB15k-237", "WN18", "WN18RR"]]
    else:
        datasets = [get_dataset(dataset, data_home)]

    for i in datasets:
        dataset = i
        cache = RelationalTripletDatasetCachePath(dataset.cache_path)
        data = RelationalTripletData(dataset=dataset, cache_path=cache)
        data.preprocess_data_if_needed()
        data.load_cache(["meta"])

        result_bracket = []
        for idx in range(times):
            seed = 10 ** idx
            set_seeds(seed)
            print("seed = ", seed)
            exp = MyExperiment(
                output, data,
                start_step, max_steps, every_test_step, every_valid_step,
                batch_size, test_batch_size, sampling_window_size, label_smoothing,
                train_device, test_device,
                resume, resume_by_score,
                lr, amsgrad, lr_decay, weight_decay,
                edim, rdim, input_dropout, hidden_dropout,
            )
            result_bracket.append(exp.final_result["average"])

        keys = list(result_bracket[0].keys())
        matrix = [[avg[key] for key in keys] for avg in result_bracket]
        result_tensor = torch.Tensor(matrix)
        result_mean = torch.mean(result_tensor, dim=0)
        result_var = torch.var(result_tensor, dim=0)
        for idx, key in enumerate(keys):
            output.logger.info(key + "  mean=" + str(float(result_mean[idx])) + "  var=" + str(float(result_var[idx])))


if __name__ == '__main__':
    main()
