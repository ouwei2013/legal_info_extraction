#  法律文本信息提取 

### Quick Start 

安装
- 进入项目根目录，输入 “ pip install . "，进行安装   
- 下载模型文件 https://drive.google.com/drive/folders/1D5ZVVp1Vo71Wy17sNVM6RuK_6X_cuPEp?usp=sharing
- 安装并且下载文件后，如果模型文件路径是model-path,可以这样调用

```
from legal_info_extraction.extract import LegalInfoExtractor as LIE  
doc = '''行政判决书（2018）豫1403行审39号申请人：商丘市睢阳区卫生和计划生育委员会，
    住所地商丘市睢阳区香君路296号。法定代表人：林若平，主任。
    委托诉讼代理人：宋晓，工作人员，特别授权。被申请人：毕某，男，汉族，1985年3月22日出生，
    住河南省商丘市睢阳区。被申请人：孔某，女，汉族，1985年11月13日出生，系毕某妻子，住址同上。
    申请人商丘市睢阳区卫生和计划生育委员会于2018年4月18日向本院提出申请，
    要求强制执行其于2017年8月18日作出的商睢卫计证字（2017）MGD106号计划生育行政征收决定。
    本院受理后，依法组成合议庭，对前述决定进行合法性审查。
    本院经审查认为，前述决定合法适当，符合有关法律规定的强制执行条件，
    依照《最高人民法院关于适用〈中华人民共和国行政诉讼法〉的解释》第一百六十条之规定，裁定如下：
    对申请人商丘市睢阳区卫生和计划生育委员会作出的商睢卫计证字（2017）MGD106号计划生育行政征收决定，
    准予强制执行。本裁定为终局裁定。审判长陈宝凤审判员陈超人民陪审员任宗杰二〇一八年四月十九日书记员杨现有'''
model = LIE('model_path')  
result = model.extract(doc)  
print(result)  


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


