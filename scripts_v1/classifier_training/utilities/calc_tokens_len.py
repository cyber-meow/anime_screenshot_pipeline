import ast
import statistics
import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df = pd.read_csv("/edahome/pcslab/pcs05/edwin/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/compressed/dafre_tags_symbolsremoved_minlen2_minapp2_profsremoved_filledempty.csv")
len_tags_list = []
len_tags_tokenized_list = []
for i, tag in enumerate(df.tags_cat0):
    tag_list = ast.literal_eval(tag)
    tag_str = ' '.join(tag_list)
    tag_tokenized = tokenizer(tag_str)['input_ids']

    #print(len(tag_list), len(tag_tokenized))
    len_tags_list.append(len(tag_list))
    len_tags_tokenized_list.append(len(tag_tokenized))
    if i % 10000 == 0:
        print('{}/{}'.format(i, len(df)))

print('Median, mean and standard deviation of tags in list form :', 
statistics.median(len_tags_list), statistics.mean(len_tags_list), statistics.stdev(len_tags_list))
print('Median, mean and standard deviation of tags after being tokenized using BERT default tokenizer: ', 
statistics.median(len_tags_tokenized_list), statistics.mean(len_tags_tokenized_list), statistics.stdev(len_tags_tokenized_list))
