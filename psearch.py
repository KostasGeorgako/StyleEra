from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

class PinterestScraper:
    def __init__(self, headless=True):
        self.headless = headless

    def _init_driver(self):
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36")
        return webdriver.Chrome(options=options)

    def scrape(self, search_term="fashion", scrolls=5, top_n=10):
        driver = self._init_driver()
        url = f"https://www.pinterest.com/search/pins/?q={search_term}"
        driver.get(url)
        time.sleep(5)  # wait for initial load

        descriptions_set = set()
        images = []

        for _ in range(scrolls):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)  # wait for new pins to load

            pins = driver.find_elements(By.CSS_SELECTOR, 'div[data-test-id="pin"]')
            for pin in pins:
                try:
                    img = pin.find_element(By.CSS_SELECTOR, 'img')
                    alt = img.get_attribute('alt')
                    src = img.get_attribute('src')

                    # Replace /236x/ with /736x/ for higher res images
                    if src and "/236x/" in src:
                        src = src.replace("/236x/", "/736x/")

                    if alt and alt not in descriptions_set:
                        descriptions_set.add(alt)
                        images.append(src)
                        if len(images) >= top_n:
                            break
                except Exception:
                    continue
            if len(images) >= top_n:
                break

        driver.quit()

        return {"top_images": images}

# Example usage:
if __name__ == "__main__":
    scraper = PinterestScraper()
    search = input("Enter search term (default 'fashion'): ").strip() or "fashion"
    result = scraper.scrape(search_term=search, scrolls=1, top_n=10)
    print(result)
