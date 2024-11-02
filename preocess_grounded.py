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
def re_ground_(ground_path,re_ground_path,statement_path):
    with open(statement_path,'r') as fin_statement:
        len_choices = [len(json.loads(line)['question']['choices']) for line in fin_statement]
    with open(ground_path, 'r') as fin:
        lines = [line for line in fin]
        
        
        
    res=[]
    i=0
    for k in len_choices:
        dic={}
        if lines[i] == "":
            continue  
        j=json.loads(lines[i])
        dic['sent']=j['sent']
        dic['qc']=j['qc']
        sub_ans=[]
        sub_ac=[]
        for start in range(k):
            
            if lines[i+start] == "":
                continue
            jj=json.loads(lines[i+start])
            sub_ans.append(jj['ans'])
            sub_ac+=jj['ac']
        i+=k
        dic['ac']=sub_ac
        dic['ans']=sub_ans
        res.append(dic)
    with open(re_ground_path, 'w') as fout:
        for line in res:
            fout.write(json.dumps(line) + '\n')
if __name__=='__main__':
    re_ground_('data/obqa/grounded/dev.grounded.jsonl','data/obqa/re_grounded/dev.re_grounded.jsonl','data/test.statement.jsonl')
