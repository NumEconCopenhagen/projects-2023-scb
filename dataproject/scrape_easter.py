# scrape ugenr.dk
import scrapy
from scrapy.crawler import CrawlerProcess
import pandas as pd 
import os

# a. create spider 
class LoginSpider(scrapy.Spider):
    name = '<3'

    def start_requests(self):
        first_year = 2000
        end_year = 2022

        url_list = [f'https://ugenr.dk/{i}/påske' for i in range(first_year,end_year+1)]
        # url_list = ["https://ugenr.dk/2020/påske"]
        for url_short in url_list:
            yield scrapy.Request(url = url_short,
                                 callback = self.parse_text)
    
    def parse_text(self, response):
        # Create a SelectorList of the course titles text
        year_txt = response.xpath('//span[contains(@id,"query")]/strong/text()')
        uge_txt = response.xpath('//span[contains(@id,"ugenr")]/span/text()')
        dato_txt = response.xpath('//p[contains(@id,"description")]/text()')
        # skærtorsdag_txt = response.xpath('//div[contains(@id,"holidays")]/dl/dt[1]/text()')
        # langfredag_txt = response.xpath('//div[contains(@id,"holidays")]/dl/dt[2]/text()')
        # påskedag_txt = response.xpath('//div[contains(@id,"holidays")]/dl/dt[3]/text()')
        # påskedag_2_txt = response.xpath('//div[contains(@id,"holidays")]/dl/dt[4]/text()')

        # Extract the text and strip it clean
        year_ext = year_txt.get().strip()[-4:]
        uge_ext = uge_txt.get().strip()
        dato_ext = dato_txt.get().strip()
        # skærtorsdag_ext = skærtorsdag_txt.get().strip()
        # langfredag_ext = langfredag_txt.get().strip()
        # påskedag_ext = påskedag_txt.get().strip()
        # påskedag_2_ext = påskedag_2_txt.get().strip()

        month_ext = dato_txt.get().strip()[-10:-5]
        end_date_ext = dato_txt.get().strip()[-14:-12]
        start_date_ext = dato_txt.get().strip()[7:9].replace('.','')


        # insert in list
        year.append(year_ext)
        uge.append(uge_ext)
        dato.append(dato_ext)
        # skærtorsdag.append(skærtorsdag_ext)
        # langfredag.append(langfredag_ext)
        # påskedag.append(påskedag_ext)
        # påskedag_2.append(påskedag_2_ext)

        month.append(month_ext)
        end_date.append(end_date_ext)
        start_date.append(start_date_ext)

# b. create empty list for datastorage 
year = []
month = []
end_date = []
start_date = []
uge = []
dato = []
# skærtorsdag = []
# langfredag = []
# påskedag = []
# påskedag_2 = [] 

# c. start spider
process = CrawlerProcess()
process.crawl(LoginSpider)
process.start()

# d. make dataset of datastorage.

# output to store
foo = 'insert path here'

df = pd.DataFrame(data=zip(year, month, start_date, end_date, uge, dato), columns=['year','month','start_date', 'end_date' ,'uge', 'dato'])

print(df)

df.to_pickle(f'{foo}/paaskedage.pkl')
print(f'exported to: {foo}/paaskedage.pkl')


