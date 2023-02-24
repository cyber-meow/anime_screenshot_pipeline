import requests
from bs4 import BeautifulSoup as Soup


def query(d, l):
    for (i, _) in l:
        d = d[i]
    return d

def clean_title(x: str):
    return x.replace('tag group:', '').replace('Tag group:', '').replace('List of ', '')


def parse_page(url):
    try:
        html = requests.get(f'https://danbooru.donmai.us{url}').text
        soup = Soup(html)

        content = soup.find(id='wiki-page-body')
        if content is None:
            print(url, 'None')
            return None
        elements = list(content.children)

        group_dict = {}
        now_stack = []
        before = ''
        for i in elements:
            match i.name:
                case 'h3'|'h4'|'h5'|'h6' as level:
                    print(level)
                    title = clean_title(i.text)
                    if title == 'See also': break
                    
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
                            query(group_dict, now_stack)[title] = entries.find(name='a')['href']
                            prev = title
                        elif entries.name == 'ul':
                            if prev != '':
                                query(group_dict, now_stack)[prev] = {
                                    'self': query(group_dict, now_stack)[prev]
                                }
                                now_stack.append((prev, 'ul'))
                            for sub_entries in entries.children:
                                title = clean_title(sub_entries.text)
                                if sub_entries.name == 'li' and sub_entries.find('a') is not None:
                                    query(group_dict, now_stack)[title] = sub_entries.find(name='a')['href']
                            
                            if prev != '':
                                now_stack.pop()
                    before = 'ul'
        
        return group_dict
    except Exception as e:
        print(url, now_stack)
        print(dumps(group_dict, indent=2, ensure_ascii=False))
        print(before)
        raise e


if __name__ == '__main__':
    from json import dumps
    res = parse_page('/wiki_pages/list_of_azur_lane_characters')
    print(dumps(res, indent=2, ensure_ascii=False))