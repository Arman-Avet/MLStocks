import json
import os
import csv
import pandas as pd
import re
import calendar
import numpy as np


path = 'data/dataset.csv'
d = open(path, 'w', encoding='utf8')
header = ['date', 'text', 'sm', 'sentiment', 'price + 1', 'price + 2', 'price + 3',
                          'price + 7']
writer = csv.writer(d)
writer.writerow(header)
d.close()

my_orgs = {
            'apple': 'AAPL',
            'amazon': 'AMZN',
            'amazon.com inc': 'AMZN',
            'google': 'GOOGL',
            'alphabet inc': 'GOOGL',
            'netflix': 'NFLX',
            'microsoft': 'MSFT',
            'facebook': 'META',
            'meta inc': 'META',
            'meta': 'META',
            'volkswagen': 'VWAGY',
            'credit suisse': 'CS'
}


def get_close(name, stock_date):
    df = pd.read_csv('data/stocks/' + name + '.csv')
    while stock_date not in df.values:
        stock_date = get_date(stock_date, 1)
    return df.loc[df.Date.shift(0).eq(stock_date), 'Close'].values[0]


def get_price(date_, incr_):
    return get_close(my_orgs[org_names], get_date(date_, incr_))


def get_date(current_date, incr):
    curr = current_date.split("-", 3)
    day = int(curr[2])
    month = int(curr[1])
    year = int(curr[0])
    cal = np.array(calendar.monthcalendar(year, month))
    cal = cal.flatten()
    index = int(np.where(cal == day)[0])
    for x in range(1, incr+1):
        index += 1
        if index >= len(cal) or cal[index] == 0:
            month += 1
            if month > 12:
                year += 1
                month = 1
            cal = np.array(calendar.monthcalendar(year, month))
            cal = cal.ravel()
            index = 0
            while cal[index] == 0:
                index += 1

    if cal[index] < 10:
        curr[2] = '0' + str(cal[index])
    else:
        curr[2] = str(cal[index])
    if month < 10:
        curr[1] = '0' + str(month)
    else:
        curr[1] = str(month)
    curr[0] = str(year)
    return "-".join(curr)


counter = 0
for i in range(1, 6):
    folder_to_view = "archive/with_orgs"

    for file in os.listdir(folder_to_view + str(i)):
        print(counter)
        if counter == 483:
            print("ada")
        fname = folder_to_view + str(i) + '/' + file
        # print(fname)
        f = open(fname, encoding="utf8")

        data = json.loads(f.read())

        orgs = data['entities']['organizations']

        for j in range(len(orgs)):
            org_names = orgs[j]['name']
            org_sentiment = orgs[j]['sentiment']

            with open(path, 'a', encoding='utf8', newline='') as d:
                writer = csv.writer(d)
                text = data['text']
                date = data['published']
                url = data['url']
                date = str(date.split("T", 2)[0])
                if org_names in my_orgs:
                    text = re.sub("[\n\"]", " ", text.strip())
                    if len(text) > 30000:
                        text = text[0:30000]
                    data_row = [date, text, '$' + my_orgs[org_names], org_sentiment,
                                get_price(date, 1), get_price(date, 2), get_price(date, 3), get_price(date, 7)]
                    writer.writerow(data_row)
                    counter += 1

        f.close()
