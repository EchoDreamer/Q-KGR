import json
def re_ground(ground_path,re_ground_path,num_choice):
    with open(ground_path, 'r') as fin:
        lines = [line for line in fin]
        
        
        
    res=[]
    for i in range(0,len(lines),num_choice):
        dic={}
        if lines[i] == "":
            continue
        j=json.loads(lines[i])
        dic['sent']=j['sent']
        dic['qc']=j['qc']
        sub_ans=[]
        sub_ac=[]
        for start in range(num_choice):
            
            if lines[i+start] == "":
                continue
            jj=json.loads(lines[i+start])
            sub_ans.append(jj['ans'])
            sub_ac+=jj['ac']
        dic['ac']=sub_ac
        dic['ans']=sub_ans
        res.append(dic)
    with open(re_ground_path, 'w') as fout:
        for line in res:
            fout.write(json.dumps(line) + '\n')
if __name__=='__main__':
    dict_choice_dataset={}
    dict_choice_dataset['csqa']=5
    dict_choice_dataset['obqa']=4
    dict_choice_dataset['medqa']=4
    re_ground('data/obqa/grounded/test.grounded.jsonl','data/obqa/re_grounded/test.re_grounded.jsonl',4)
    re_ground('data/obqa/grounded/train.grounded.jsonl','data/obqa/re_grounded/train.re_grounded.jsonl',4)
    re_ground('data/obqa/grounded/dev.grounded.jsonl','data/obqa/re_grounded/dev.re_grounded.jsonl',4)