from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
import torch.nn as nn
import pandas as pd
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
import numpy as np
from key_match import key_match_metric
from datetime import datetime


class OCLgenerator():
    def __init__(self, checkpoint, datasource, random_seed=17):
        self.args = None
        self.trainer = None

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.data=datasource

        self.seed=random_seed

        self.model = self.model.to(self.device)

        self.dataset = load_dataset('csv', data_files=self.data, encoding='utf8')

        self.split_datasets = self.dataset["train"].train_test_split(train_size=0.9, seed=self.seed)

        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        self.bleu = load_metric("sacrebleu")

        self.tokenized_datasets = self.split_datasets.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.split_datasets["train"].column_names, )
        self.pre_train()

    def preprocess_function(self, examples):
        # max_nl_length=674
        # max_ocl_length=1742
        max_input_length = 1024
        max_target_length = 1024
        inputs = [ex for ex in examples["NL"]]
        targets = [ex for ex in examples["OCL"]]
        model_inputs = self.tokenizer(inputs, max_length=max_input_length, truncation=True)
        # Set up the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_preds):
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

    def pre_train(self):
        self.args = Seq2SeqTrainingArguments(
            "./checkpoints_k=1",
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=500,
            learning_rate=2e-5,
            per_device_train_batch_size=10,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=200,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False,
            # report_to="wandb"
        )
        self.trainer = testtrainer(
            self.model,
            self.args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

    def train(self):
        self.trainer.train()

    def get_test_result(self):
        length_test = len(self.split_datasets['test'])
        add = []

        for i in range(length_test):
            input_ids = self.tokenizer(self.split_datasets['test'][i]['NL'], return_tensors="pt").input_ids
            input_ids = input_ids.to(self.device)
            generated_ids = self.model.generate(input_ids, max_length=2048)
            add.append(self.tokenizer.decode(generated_ids[0], skip_special_tokens=True))

        return add

    def get_test_score(self, predictions, references, metric):
        scores = []
        score = 0
        for i in range(len(predictions)):
            if (metric.name == 'sacrebleu'):
                score = metric.compute(predictions=[predictions[i]], references=[[references['OCL'][i]]])
            else:
                score = metric.compute(predictions=predictions[i], references=references['OCL'][i])
            scores.append(score['score'])
        return scores

    def evaluate(self, type='two'):
        scores_list = []
        result = self.get_test_result()
        scores = {}
        if type == 'bleu':

            scores['score'] = self.get_test_score(predictions=result, references=self.split_datasets['test'],
                                                  metric=self.bleu)
            scores['name'] = 'bleu'
            scores_list.append((scores))
        elif type == 'key_match':
            scores['score'] = self.get_test_score(predictions=result, references=self.split_datasets['test'],
                                                  metric=key_match_metric())
            scores['name'] = 'key_match'
            scores_list.append((scores))

        if type == 'two':
            scores = {
                'score': self.get_test_score(predictions=result, references=self.split_datasets['test'],
                                             metric=self.bleu),
                'name': 'bleu'}
            scores_list.append(scores)
            scores = {'score': self.get_test_score(predictions=result, references=self.split_datasets['test'],
                                                   metric=key_match_metric()),
                      'name': 'key_match'}
            scores_list.append(scores)

        self.eval_result = self.split_datasets['test']
        self.eval_result = self.eval_result.flatten_indices()  # 虽然不知道为什么但没有这一行长度就对不上
        self.eval_result = self.eval_result.add_column("Candidate", result)
        for scores in scores_list:
            # print(scores)
            self.eval_result = self.eval_result.add_column(scores['name'], scores['score'])
            #self.eval_result = self.eval_result.add_column(scores['name']+'avg',[sum(scores['score'])/len(scores['score'])])

        pd.DataFrame(self.eval_result).to_csv('./eval/seed=' +str(self.seed)+ 'data='+str(self.data).replace('./','')+str(datetime.now()) + 'eval_result.csv')


class testtrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        wlist=torch.tensor(weight_list)
        wlist=wlist.float()
        wlist=wlist.to(device)
        loss_fct = nn.CrossEntropyLoss(weight=wlist,ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    '''
     loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    '''

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    keywords = [' and ', ' atrr ', 'body', 'context', ' def ', ' if ', 'else',
                'endif', 'endpackage', 'implies', ' in ', ' inv ', 'let',
                ' not ', 'oper', ' or ', 'package', 'post', ' pre ', 'then', 'xor',
                'derive']
    built_in_obj_properties = ['oclIsTypeOf', 'oclIsKindOf', 'allInstances()', 'oclInState',
                               'oclIsInState()'  # ignore specifications  diversity
                               'oclIsNew', 'oclAsType', 'oclIsUndefined()',
                               'hasReturned()', 'result()', 'isSignalSent', 'isOperationCall()']
    # including opts in collection,set,bag and sequence
    standard_lib_operations = ['div(', 'mod(', 'abs(', 'max(', 'min(',
                               'concat(', '<>', 'substring(', 'toUpper(', 'toLower(', 'toInteger(', 'toReal('
                                                                                                    'size()',
                               'includes(', 'excludes(', 'count(', 'includesAll(', 'excludesAll(',
                               'isEmpty()', 'notEmpty()', 'sum()', 'exists(', 'forall(', 'isUnique(', 'sortedBy(',
                               'iterate(',
                               'union(', 'intersection(', 'including(', 'excluding(', 'symmetricDifference(', 'select(',
                               'reject(', 'collect(', 'flatten()', 'asSequence()', 'asBag()',
                               'asSet()',
                               'append(', 'prepend(', 'insertAt(', 'subSequence(', 'at(', 'first()', 'indexOf(',
                               'any(', 'collectNested(', 'one('
                               ]
    keys = keywords + built_in_obj_properties + standard_lib_operations
    tokenized_keys = tokenizer(keys)['input_ids']
    weight_keys = []
    for i in tokenized_keys:
        weight_keys += i
    weight_keys = list(set(weight_keys))
    weight_list = [1] * len(tokenizer.get_vocab())
    for i in weight_keys:
        weight_list[i] = 1
    weight_list[1] = 1
    weight_list[2] = 1

    #generator = OCLgenerator(checkpoint="Salesforce/codet5-base", datasource='./SUM-ALL3.csv')#random_seed=17
    generator = OCLgenerator(checkpoint="Salesforce/codet5-base", datasource='./SUM-ALL4.csv',random_seed=29)
    #generator = OCLgenerator(checkpoint="./checkpoints_custom/checkpoint-20000", datasource='./SUM-ALL4concode.csv',random_seed=37)
    generator.train()
   # generator.evaluate()
