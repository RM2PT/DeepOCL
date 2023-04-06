from datasets import load_metric
from OCLMetric.Sacrebleu import Sacrebleu
from OCLMetric.AST_match.TreeMatch import Treematcher

class key_match_metric():
    '''
    设计：
    keys including 3 parts:
    keywords: inv,pre,etc. weight 3
    object properties:ociskindof(),etc. weight 1
    standard library: notempty(),exists( ,etc. weight 2

    arguments should be able to point the keys that should be used,default all 3.

    mainidea:
    label the used keys in refs ,count how many times each key appears,
    then div the times appeared in the candidate by the times that appeared in refs.
    then calculate the weighted average of 3 kinds keys. the weights are 3,1,2
    then normalized the score. the full score should be 100
    '''
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

    def __init__(self, key=keys):
        self.key = key
        self.name = 'key_match_metric'
        self.bleu = Sacrebleu()

    def compute_func(self, predictions, references, words):
        count_ref = {}
        count_pre = {}
        match = {}
        ref_num, pre_num = 0, 0
        score = 0
        for word in words:
            if (references.count(word) != 0):
                count_ref[word] = references.count(word)
                ref_num += count_ref[word]
        for key in count_ref:
            count_pre[key] = predictions.count(key)

            # avoid too much key in the prediction making the score too high
            if (count_pre[key] > count_ref[key]):
                count_pre[key] = count_ref[key]
            pre_num += count_pre[key]
        # no such kind key in the reference ,so get None
        if (len(count_ref) == 0):
            score = None
            match['valid'] = 0
        else:
            score = pre_num / ref_num
            score *= 100
            match['valid'] = 1
        match['ref_num'] = ref_num
        match['pre_num'] = pre_num
        match['score'] = score
        return match

    def key_compute(self, pre, ref):
        # print(pre)
        # print(ref)
        predictions, references = pre.upper(), ref.upper()
        result = {}
        result['keywords'] = self.compute_func(predictions, references, words=[i.upper() for i in self.keywords])
        result['obj_properties'] = self.compute_func(predictions, references,
                                                     words=[i.upper() for i in self.built_in_obj_properties])
        result['lib_opt'] = self.compute_func(predictions, references,
                                              words=[i.upper() for i in self.standard_lib_operations])

        if (result['keywords']['ref_num'] == 0 and result['obj_properties']['ref_num'] == 0 and result['lib_opt']['ref_num'] == 0):
            result['positive'] = False
        else:
            result['positive'] = True

        score = 3 * result['keywords']['score'] if result['keywords']['score'] is not None else 0
        score+= 1 * result['obj_properties']['score'] if result['obj_properties']['score'] is not None else 0 
        score+= 2 * result['lib_opt']['score'] if result['lib_opt']['score'] is not None else 0

        if ( 3 * result['keywords']['valid'] + 1 * result['obj_properties']['valid'] + 2 * result['lib_opt']['valid']==0):
            result['score']=0
        else :
            score = score / (
                3 * result['keywords']['valid'] + 1 * result['obj_properties']['valid'] + 2 * result['lib_opt'][
            'valid'])
            result['score'] = score
        return result

    def bleu_compute(self, pre, ref):
        metric = self.bleu
        result = metric.compute(predictions=[pre], references=[[ref]])
        if (result['score'] == 0):
            if (result['precisions'][3] == 0):
                if (result['precisions'][2] == 0):
                    result['score'] = pow(result['precisions'][0] * result['precisions'][1], 0.5)
                else:
                    result['score'] = pow(result['precisions'][0] * result['precisions'][1] * result['precisions'][2],
                                          1 / 3)

        result['score'] = round(result['score'], 5)

        return result

    def compute(self, predictions, references):
        result = {}
        result['bleu'] = self.bleu_compute(pre=predictions, ref=references)
        result['key'] = self.key_compute(pre=predictions, ref=references)
        # from 0~2,the importance of key-score
        key_weight = (3 * result['key']['keywords']['valid'] + 1 * result['key']['obj_properties']['valid'] + 2 *
                      result['key']['lib_opt']['valid']) / 3
        result['score'] = pow(result['key']['score'], key_weight)
        result['score'] = pow(result['score'] * result['bleu']['score'], 1 / (key_weight + 1))
        '''
        if (result['key']['positive']):
            result['score'] = pow(result['bleu']['score'] * result['key']['score'] * result['key']['score'], 1 / 3)
        else:
            result['score'] = result['bleu']['score']
        '''
        ascore = Treematcher.calculate(predictions,references)*100
        
        if ascore>0:
            result['score'] = pow(result['score'], 10+key_weight)
            result['score'] = pow(result['score'] * ascore, 1 / (key_weight + 11))        
        
        
        result['score'] = round(result['score'], 5)
        

        
        return result
    
    def getresult(self,predictions, references):
        return Treematcher.calculate(predictions, references)
        
