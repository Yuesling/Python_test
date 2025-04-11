from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import csv
import time


def scrape_multiple_pages():
    # 配置 Selenium
    chrome_driver_path = "D:/chromedriver_win32/chromedriver.exe"  # 替换为你的 chromedriver 路径
    service = Service(chrome_driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 无界面模式
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # 加载主页面
        url = "https://www.lottery.gov.cn/kj/kjlb.html?pls"
        driver.get(url)

        # 等待 iframe 加载
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "iFrame1"))
        )

        # 切换到 iframe
        iframe = driver.find_element(By.ID, "iFrame1")
        driver.switch_to.frame(iframe)

        # 找到分页标签
        page_ul = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "m-pager"))
        )

        # 找到尾页的 onclick 属性，提取最大页数
        last_page_button = page_ul.find_elements(By.TAG_NAME, "li")[-1]
        last_page = int(last_page_button.get_attribute("onclick").split("(")[1].split(")")[0])

        print(f"发现总页数：{last_page}")

        # 存储抓取结果
        results = []

        # 遍历所有页数
        for page in range(1, last_page + 1):
            print(f"抓取第 {page} 页数据...")
            # 调用翻页 JS 方法
            driver.execute_script(f"kjCommonFun.goNextPage({page});")

            # 等待表格数据更新
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.ID, "historyData"))
            )
            time.sleep(2)  # 给页面一些时间加载

            # 抓取表格 tbody 的内容
            tbody = driver.find_element(By.ID, "historyData")
            tbody_html = tbody.get_attribute("outerHTML")

            # 用 BeautifulSoup 解析 HTML
            soup = BeautifulSoup(tbody_html, "html.parser")
            rows = soup.find_all("tr")

            for row in rows:
                columns = row.find_all("td")
                period = columns[0].text.strip()  # 期号
                date = columns[1].text.strip()  # 日期
                numbers = [
                    columns[2].text.strip(),  # 开奖号码1
                    columns[3].text.strip(),  # 开奖号码2
                    columns[4].text.strip()   # 开奖号码3
                ]

                # 将数据保存到一个字典中
                results.append({
                    "期号": period,
                    "日期": date,
                    "开奖号码": numbers
                })

        # 保存数据到 CSV 文件
        with open("lottery_results_all_pages.csv", "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["期号", "日期", "开奖号码"])
            writer.writeheader()
            writer.writerows(results)

        print(f"全部抓取完成，共抓取 {len(results)} 条记录！结果已保存至 lottery_results_all_pages.csv 文件。")

    except Exception as e:
        print(f"出现错误：{e}")

    finally:
        driver.quit()


if __name__ == "__main__":
    scrape_multiple_pages()
