"""
@auther yueshenglin
@date 2025-03-24
描述：爬取百度网页图片
"""
import requests
import re
import os

headers = {
    "user-agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-encoding" : "gzip, deflate, br, zstd",
    "accept-language" : "zh-CN,zh;q=0.9"
}
url = "https://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&ie=utf8&word=%E5%B0%8F%E5%A7%90%E5%A7%90%E5%9B%BE%E7%89%87&fr=ala&ala=1&alatpl=normal&pos=0&dyTabStr=MCwxMiwzLDEsMiwxMyw3LDYsNSw5"
html = requests.get(url, headers = headers)
html.encoding = "utf-8"
html = html.text
#print(html)



img_url_list = re.findall(r'^https?://', html)
print(img_url_list)
for i in img_url_list:
    print(i)
print(len(img_url_list))
img_path = "./picture"
if not os.path.exists(img_path):
    os.mkdir(img_path)

count = 0
for img_url in img_url_list:
    count += 1
    img = requests.get(img_url)
    file_name = img_path + "/" + str(count) + ".jpg"
    f = open(file_name, "wb")
    f.write(img.content)
    print(img_url)

