from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def scrape_lottery_results():
    # 配置 Selenium
    chrome_driver_path = "D:/chromedriver_win32/chromedriver"  # 替换为你的 chromedriver 路径
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
        print("切换到 iframe 内容中...")

        # 切换到 iframe
        iframe = driver.find_element(By.ID, "iFrame1")
        driver.switch_to.frame(iframe)

        # 抓取 iframe 内部内容
        time.sleep(3)  # 等待 iframe 中的内容加载完成
        page_source = driver.page_source

        # 输出 iframe 的内容（调试使用）
        with open("iframe_content.html", "w", encoding="utf-8") as file:
            file.write(page_source)
        print("iframe 内容已保存到 iframe_content.html")

        # 此处可以进一步解析 iframe HTML 内容，找到需要的数据

    except Exception as e:
        print(f"出现错误：{e}")

    finally:
        driver.quit()


if __name__ == "__main__":
    scrape_lottery_results()
