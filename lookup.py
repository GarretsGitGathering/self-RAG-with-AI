from googlesearch import search

def initial_lookup(query, source_count):
    urls = []
    for i in search(query, tld="co.in", num=source_count, stop=source_count, pause=2):
        urls.append(i)

    return urls
