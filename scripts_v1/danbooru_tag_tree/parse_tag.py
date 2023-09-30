import requests
from bs4 import BeautifulSoup as Soup

from parse_page import parse_page

def query(d, l):
    for i, _ in l:
        d = d[i]
    return d

def clean_title(x: str):
    return x.replace('tag group:', '').replace('Tag group:', '').replace('List of ', '')


html = requests.get('https://danbooru.donmai.us/wiki_pages/tag_groups').text
soup = Soup(html)

content = soup.find(id='wiki-page-body')
elements = list(content.children)[5:-3]

group_dict = {}
now_stack = []
before = ''
for i in elements:
    match i.name:
        case 'h3'|'h4'|'h5'|'h6' as level:
            title = clean_title(i.text)
            print(level, title, now_stack)
            
            pop = 0
            while (before>=level or before=='ul') and now_stack:
                prev_title, before = now_stack.pop()
                print(prev_title, before)
                pop += 1
            if before<level and pop:
                now_stack.append((prev_title, before))
            
            query(group_dict, now_stack)[title] = {}
            now_stack.append((title, level))
            before = level
        case 'ul':
            print('ul')
            prev = ''
            for entries in i.children:
                title = clean_title(entries.text)
                if entries.name == 'li' and entries.find('a') is not None:
                    href = entries.find(name='a')['href']
                    query(group_dict, now_stack)[title] = parse_page(href)
                    prev = title
                elif entries.name == 'ul':
                    query(group_dict, now_stack)[prev] = {
                        'self': query(group_dict, now_stack)[prev]
                    }
                    now_stack.append((prev, 'ul'))
                    for sub_entries in entries.children:
                        title = clean_title(sub_entries.text)
                        if sub_entries.name == 'li' and sub_entries.find('a') is not None:
                            href = sub_entries.find(name='a')['href']
                            query(group_dict, now_stack)[title] = parse_page(href)
                    now_stack.pop()
            before = 'ul'

from json import dumps

with open('tag_tree.json', 'w', encoding='utf-8') as f:
    f.write(dumps(group_dict, indent=2, ensure_ascii=False))