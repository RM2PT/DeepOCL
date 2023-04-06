from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
import pandas as pd
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
import numpy as np
from OCLMetric.key_match import  key_match_metric
from OCLMetric.Sacrebleu import Sacrebleu
from datetime import datetime
from IPython import embed

from tqdm import tqdm


class OCLgenerator():
    def __init__(self, checkpoint, datasource, random_seed=17, predictor=None):
        self.args = None

        self.trainer = None

        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


            

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.data = datasource

        self.seed = random_seed

        self.model = self.model.to(self.device)

        if (predictor):
            self.predictor = AutoModelForSeq2SeqLM.from_pretrained(predictor)
            
            self.predictor = self.predictor.to(self.device)
            
        self.dataset = load_dataset('csv', data_files=self.data, encoding='utf8')

        self.split_datasets = self.dataset["train"].train_test_split(train_size=0.9, seed=self.seed)

        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        self.bleu = Sacrebleu()

        self.tokenized_datasets = self.split_datasets.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.split_datasets["train"].column_names, )

    def preprocess_function(self, examples):
        # 预处理函数，为生成任务准备数据
        # max_nl_length=674
        # max_ocl_length=1742
        max_input_length = 1024
        max_target_length = 1024
        inputs = ['Generate OCL : ' + ex for ex in examples["NL"]]
        targets = [ex for ex in examples["OCL"]]
        model_inputs = self.tokenizer(inputs, max_length=max_input_length, truncation=True)
        # Set up the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_preds):
        #没啥用
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    def gen_pre_train(self, save_dict='./checkpoints_generate2'):
        
        self.args = Seq2SeqTrainingArguments(
            save_dict,
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=1000,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=10,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False,
            # report_to="wandb"
        )
        self.gen_trainer = Seq2SeqTrainer(
            self.model,
            self.args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

    def gen_train(self,save_dict):
        # 第一阶段生成任务训练
        self.gen_pre_train(save_dict)
        self.gen_trainer.train()

    def predict_score(self, nl, ocl) -> float:
        #输入NL,OCL，使用predictor构造形式并输出分数，对错误的输出给0分
        sep = self.tokenizer.sep_token
        cls = self.tokenizer.cls_token
        prefix = 'Evaluate OCL : '

        try:
            item = cls + prefix + nl + sep + ocl + sep
        except TypeError:
            embed()
        else:
            input_ids = self.tokenizer(item, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.device)

        generated_ids = self.predictor.generate(input_ids, max_length=2048)
        #embed()
        result = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        try:
            score=float(result)
        except ValueError:
            score=0.0
        return score

    def gen_score(self, predictions, references, metric):
        scores = []
        score = 0
        for i in range(len(predictions)):
            if (metric.name == 'sacrebleu'):
                score = metric.compute(predictions=[predictions[i]], references=[[references['OCL'][i]]])
            else:
                score = metric.compute(predictions=predictions[i], references=references['OCL'][i])
            scores.append(score['score'])
        return scores

    def gen_result(self, with_predict=False):
        #使用模型生成结果，添加生成前缀，k采样，有predictor则评分，没有就不管
        dataset = self.split_datasets['test']
        ret = []
        NL = dataset['NL']
        inputs = ['Generate OCL : ' + item for item in NL]

        for i in range(len(inputs)):
            item = inputs[i]
            input_ids = self.tokenizer(item, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.device)
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                max_length=2048,
                top_k=40,
                # top_p=0.9,
                num_return_sequences=5
            )
            result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            if (with_predict):
                code1 = result[0]
                score1 = self.predict_score(ocl=code1, nl=dataset['NL'][i])
                # this compuet a score withou further useage
                for code in result:
                    score = self.predict_score(ocl=code, nl=dataset['NL'][i])
                    if score1 < score:
                        score1 = score
                        code1 = code
                code = code1
            else:
                code = result[0]

            ret.append(code)

        return ret

    def gen_evaluate(self, with_predict=False, eva_predict=False):
        #生成评估结果，with_predict控制是否使用predictor生成，eva_predict控制是否显示predictor分数
        generated = self.gen_result(with_predict)
        result = None
        dataset = self.split_datasets['test']
        bleu_scores = {
            'score': self.gen_score(predictions=generated, references=dataset, metric=self.bleu),
            'name': 'bleu'}
        custom_scores = {
            'score': self.gen_score(predictions=generated, references=dataset, metric=key_match_metric()),
            'name': 'key_match'}
        if eva_predict:
            pred_scores = []
            for i in range(len(generated)):
                pred_scores.append(self.predict_score(nl=dataset['NL'][i], ocl=generated[i]))
            result = pd.DataFrame(
                {'NL': dataset['NL'], 'OCL': dataset['OCL'], 'Candidate': generated,
                 'BLEU': bleu_scores['score'], 'Key_metric': custom_scores['score'], 'Predict': pred_scores})

        else:
            result = pd.DataFrame(
                {'NL': dataset['NL'], 'OCL': dataset['OCL'], 'Candidate': generated,
                 'BLEU': bleu_scores['score'], 'Key_metric': custom_scores['score']})

        result.to_csv(
            './eval/seed=' + str(self.seed) + 'data=' + str(self.data).replace('./', '') + str(
                datetime.now()) + 'eval_result.csv')

    def pre2(self, exp, prefix=''):
        #评估数据的map函数， 构造输入形式，将score转换为str
        sep = self.tokenizer.sep_token
        cls = self.tokenizer.cls_token
        max_input_length = 2560
        max_target_length = 5
        inp11 = [a.replace('Generate OCL: ', '') for a in exp['NL']]
        inp1 = [prefix + a for a in inp11]
        inp2 = [a.replace('\n', '') for a in exp['OCL']]
        inputs = []
        # 此处len(exp)和len(exp['NL'])不一样
        for i in range(len(exp['NL'])):
            inputs.append(cls + inp1[i] + sep + inp2[i] + sep)
        targets = [str(s) for s in exp['SCORE']]
        # model_inputs = self.tokenizer(inputs, max_length=max_input_length,padding='max_length', truncation=True)
        model_inputs = self.tokenizer(inputs, max_length=max_input_length, truncation=True)
        # Set up the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=max_target_length, padding='max_length', truncation=True)
        model_inputs["labels"] = labels["input_ids"]


        return model_inputs

    def load_eva(self, eva_data):
        # 准备评估任务的数据
        # dataset = load_dataset('csv', data_files='./testscore3.csv', encoding='utf8')
        dataset = load_dataset('csv', data_files=eva_data, encoding='utf8')
        dataset = dataset.remove_columns('Unnamed: 0')
        dataset = dataset.remove_columns('TYPE')
        data = dataset['train']

        # embed()

        dataset2 = data.map(self.pre2, batched=True, remove_columns=data.column_names,
                            fn_kwargs={"prefix": 'Evaluate OCL : '})
        return dataset2

    def train_eva(self, eva_data,save_dict):
        #调用这个方法，进行评估任务训练
        self.data2 = self.load_eva(eva_data)
        self.train2(save_dict)

    def train2(self,save_dict):
        self.args2 = Seq2SeqTrainingArguments(
            save_dict,
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=1000,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=24,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=100,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False,
            # report_to="wandb"
        )
        self.trainer2 = Seq2SeqTrainer(
            self.model,
            self.args2,
            train_dataset=self.data2,
            eval_dataset=self.data2,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        self.trainer2.train()

    def evaluate_eva(self):
        #评估测试集上，predictor效果
        dataset = self.split_datasets['test']
        ret = []
        NL = dataset['NL']
        OCL = dataset['OCL']
        metric = key_match_metric()
        
        sep = self.tokenizer.sep_token
        cls = self.tokenizer.cls_token
        prefix = 'Evaluate OCL : '
        
        for i in tqdm(range(len(NL))):
            item = cls + prefix + NL[i] + sep + OCL[i]+ sep
            input_ids = self.tokenizer(item, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.device)

            generated_ids = self.model.generate(input_ids, max_length=2048)
            result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # score=metric.compute(predictions=ocl,references=OCL[i])
            # ret.append([NL[i],OCL[i],result,score['score']])
            # df = pd.DataFrame(ret,columns=['NL','OCL','Predict','SCORE'],dtype=float)
            ret.append([NL[i], OCL[i], result])
        df = pd.DataFrame(ret, columns=['NL', 'OCL', 'Predict'], dtype=float)
        df.to_csv('./test_eva.csv')

    def generate_test_data(self):
        #生成测试所用材料
        ret = []
        NL = self.split_datasets['train']['NL']
        # NL+=generator.split_datasets['test']['NL']
        OCL = self.split_datasets['train']['OCL']
        # OCL+=generator.split_datasets['test']['OCL']
        metric = key_match_metric()
        for i in tqdm(range(len(NL))):
            item = NL[i]
            input_ids = self.tokenizer(item, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.device)

            generated_ids = self.model.generate(
                input_ids,
                # do_sample=True,
                max_length=2048,
                num_beams=5,
                num_return_sequences=3
            )

            result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # ret.append([item,OCL[i],1,100])
            for ocl in result:
                score = metric.compute(predictions=ocl, references=OCL[i])
                if (score['score'] == 100):
                    continue

                ret.append([item, ocl, 0, score['score']])
        df = pd.DataFrame(ret, columns=['NL', 'OCL', 'TYPE', 'SCORE'], dtype=float)
        df.drop_duplicates(subset='OCL', keep='first', inplace=False)
        df.to_csv('./eva_data.csv')
        
 

'''
生成数据：     generator.generate_test_data()
进行生成训练： generator.gen_train()
进行评估训练   generator.train_eva('./eva_data.csv')
评估最终结果   generator.gen_evaluate(with_predict=True,eva_predict=True)


'''
if __name__ == "__main__":
    '''
    #checkpoint="Salesforce/codet5-base"
    #checkpoint='./checkpoints_generate/checkpoint-44000'
    #checkpoint='./checkpoints_eva/checkpoint-44000'
    checkpoint='./checkpoints_generate2/checkpoint-7000'
    eva_checkpoint='./checkpoints_eva2/checkpoint-7000'
    generator = OCLgenerator(checkpoint=checkpoint, datasource='./OCLpairs.csv',predictor=eva_checkpoint)#generate
    #generator.train()
    generator.gen_evaluate(with_predict=True)
    #generator.gen_evaluate()
    generator.gen_evaluate(with_predict=True,eva_predict=True)
    #generator.train_eva('./eva_data.csv')
    #generator.evaluate_eva()
    '''
    '''
    # 1st train naive model
    checkpoint="Salesforce/codet5-base"
    generator = OCLgenerator(checkpoint=checkpoint,datasource='./OCLpairs.csv')
    generator.gen_train('./checkpoints_naive')
    

    
    # get data for predictor training
    checkpoint='./checkpoints_naive/checkpoint-7000'
    generator = OCLgenerator(checkpoint=checkpoint,datasource='./OCLpairs.csv')
    generator.generate_test_data()

    #  2nd train generator
    #  figure the epoch 2 100
    checkpoint='./checkpoints_naive/checkpoint-7000'
    generator = OCLgenerator(checkpoint=checkpoint,datasource='./OCLpairs.csv')
    generator.gen_train('./checkpoints_generator')
   
    # 3rd train predictor
    checkpoint='./checkpoints_naive/checkpoint-7000'
    generator = OCLgenerator(checkpoint=checkpoint,datasource='./OCLpairs.csv')
    
    generator.train_eva('./eva_data_true.csv','./checkpoints_selector')
'''
    # 4 evaluate
    #generator = OCLgenerator(checkpoint='./checkpoints_generator/checkpoint-14000',datasource='./OCLpairs.csv',predictor='./checkpoints_selector/checkpoint-19000')
    generator = OCLgenerator(checkpoint='./checkpoints_naive/checkpoint-1000',datasource='./OCLpairs.csv')
    generator.gen_evaluate(with_predict=False,eva_predict=False)

