from splinter import Browser
from bs4 import BeautifulSoup as bs
import pandas as pd
import datetime as dt
import requests
import time


def scrape_wiki(filename):

    bird = load_model(filename)
    all_tables = {}
    bird_data={}
        
    executable_path = {'executable_path': 'chromedriver.exe'}
    browser = Browser('chrome', **executable_path, headless=False)

    my_url = 'https://en.wikipedia.org/wiki/' + bird
    browser.visit(my_url)
    
    ## giving it a bit of time to load before pulling info ###
    time.sleep(1) 
    url_html = browser.html

    ### Table Pull ###

    tables = pd.read_html(url_html)
    #tables[0]
    df = tables[0]
    df.columns = ['About']
    bird_facts_html = df.to_html(index=False, classes="table-hover table-dark table-sm")
    bird_data["facts_table"] = bird_facts_html

    # x = len(tables)
    # x    

    # for i in range (0, x):
    #     table = tables[i]
    #     all_tables[f'tables_{i}']=table

    # scrape_data = tables[0]
    # scrape_data

    ### Image pulls ###

    response = requests.get(url_html)
    soup = bs(response.text, 'html.parser')

    image_tags = soup.findAll('img')

    bird_img = image_tags[1].get("src")
    
    loc_img = image_tags[4].get("src")
