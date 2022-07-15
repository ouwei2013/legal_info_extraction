#  法律文本信息提取 

### Quick Start 

安装：进入项目根目录，使用 pip install . 
安装好后，就可以调用拉

```
from legal_info_extraction.extract import LegalInfoExtractor as LIE  
doc='原告黄某。被告市城管局。关于强制拆迁一案，本院认为，黄某行为不当'  
model = LIE('model_path')  
result = model.extract(doc)  
print(result)  

{
  "原告":[[2,4,"原告", "黄某" ]],
  "被告":[[7,11, "行政主体","市城管局" ]] ,
  "基本信息":[[ 14,18,"案由" "强制拆迁",]],
  "本院认为":[[ 26,31, "理由" "黄某行为不当",]],
}
```  


### 大致方法

#### Training 

- 使用spacy 3.0 +transformer (BERT-Chinese-base)，在100条标注数据上训练一个NER模型

#### Test: 

-  对于一个长测试文本，首先根据句号进行分割
-  将分割得到的句子输入到训练好的tranformer，得到初步结果
-  根据一些行政类判决文书的结构特点，对初步结果过滤、优化


### TODO 
- 待测试
- 加入main.py 
- 加入logging
- 引入一些额外规则对行政处罚决定与诉求优化


