import requests
from readabilipy import simple_json_from_html_string
from SGDRegressor import *

req = requests.get('https://www.cnbc.com/2018/12/14/google-ceo-sundar-pichai-had-a-tough-and-terrible-year.html')
article = simple_json_from_html_string(req.text, use_readability=True)
text_list = article['plain_text']
text = ''
for i in range(len(text_list)):
    tval = text_list[i]['text']
    if tval[0] == '©':
        break
    elif tval[0] == '*':
        pass
    elif len(tval) > 30:
        text += tval+'\n'
    if len(text) > 30000:
        break


d = {'text': [text], 'sm': ['$GOOGL'], 'price + 1': [51.28]}
news = pd.DataFrame(data=d)

print(news.head())

prediction = regressor.predict(news[['text', 'sm']])
print(prediction)
pred_score = mean_absolute_error(news['price + 1'], prediction)
print(pred_score)
