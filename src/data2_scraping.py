import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os


def scrape_deweloperuch(number_of_pages = None, transaction_type=None):
    """ Funkcja scrapuje dane transakcyjne ze strony deweloperuch.pl i zwraca listę słowników,
        gdzie każdy słownik to jeden rekord """

    if transaction_type is None:
        transaction_type = ['transakcja']
    filter_string = ",".join(transaction_type)
    print(f"Wybrane filtry transakcji: {filter_string}")

    def _scrape_one_page(page, scrape_number_of_pages=False):
        BASE_URL = "https://deweloperuch.pl/ceny-transakcyjne/polska/mieszkania"
        HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7"
        }

        try:
            params = {
                "perPage": 100,
                "page": page,
                "filterClassification": filter_string
            }
            
            response = requests.get(BASE_URL, headers=HEADERS, params=params)
            
            if response.status_code != 200:
                print(f"Błąd przy stronie {page}: Status {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            
            if scrape_number_of_pages:
                DEFAULT_PAGES = 401

                pagination_div = soup.find('div', class_='text-sm text-gray-600')
                if pagination_div:

                    pagination_text = pagination_div.get_text(strip=True)

                    numbers = re.findall(r'\d+', pagination_text)
                    
                    if numbers:
                        last_number = int(numbers[-1])
                        total_pages = last_number
                            
                        print(f"Wykryta liczba stron: {total_pages}")
                    else:
                        print("Nie znaleziono liczb w divie.")
                        total_pages = DEFAULT_PAGES
                else:
                    print("Nie znaleziono żadnego diva.")
                    total_pages = DEFAULT_PAGES

            table = soup.find('table')
            if not table:
                print(f"Nie znaleziono tabeli na stronie {page}.")
                return None
                
            rows = table.find_all('tr')
            if not rows:
                print(f"Nie znaleziono wierszy na stronie {page}.")
                return None

            data = []
            for row in rows:
                cols = row.find_all('td')
                if len(cols) == 8:
                    
                    # ID i UDZIAŁ
                    title_div = cols[0].find(attrs={"title": True})
                    
                    full_id = ""
                    if title_div:
                        full_id = title_div['title'].strip()
                    else:
                        full_id = cols[0].get_text(strip=True).split("Udział:")[0]

                    full_text = cols[0].get_text(strip=True)
                    share_text = "1/1"
                    
                    if "Udział:" in full_text:
                        parts = full_text.split("Udział:")
                        if len(parts) > 1:
                            share_text = parts[1].strip()

                    # ADRES i MIASTO
                    a_tag = cols[1].find('a')
                    adress_text = a_tag.get_text(strip=True) if a_tag else ""
                    
                    divs = cols[1].find_all('div')
                    
                    city_text = ""
                    if len(divs) >= 2:
                        city_text = divs[-1].get_text(strip=True)
                    
                    record = {
                        "ID": full_id,
                        "Udział": share_text,
                        "Adres": adress_text,
                        "Miasto": city_text,
                        "Metraż": cols[2].get_text(strip=True),
                        "Pokoje": cols[3].get_text(strip=True),
                        "Piętro": cols[4].get_text(strip=True),
                        "Cena": cols[6].get_text(strip=True),
                        "Data_transakcji": cols[7].get_text(strip=True)
                    }
                    data.append(record)
            
            if scrape_number_of_pages:
                return data, total_pages
            else:
                return data

        except Exception as e:
            print(f"Wystąpił błąd na stronie {page}: {e}")
            return None

    all_data = []

    print(f"Rozpoczynam scrapowanie.")

    if not number_of_pages:
        data, number_of_pages = _scrape_one_page(1, True)
    else:
        data = _scrape_one_page(1)
    if data:
        all_data.extend(data)
    print(f"Strona 1/{number_of_pages} pobrana. Rekordów w sumie: {len(all_data)}")
    time.sleep(random.uniform(2, 4))
    
    for page in range(2, number_of_pages + 1):
        data = _scrape_one_page(page)
        if data:
            all_data.extend(data)
        print(f"Strona {page}/{number_of_pages} pobrana. Rekordów w sumie: {len(all_data)}")
        time.sleep(random.uniform(2, 4))
    
    return all_data


def save_to_csv(data, output_file = "data.csv",):
    """ Funkcja zapisuje listę słowników data do pliku csv """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data', 'scrapped')

    # Tworzymy folder data jeśli nie istnieje
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    output_file = os.path.join(data_dir, output_file)

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=',')
    print(f"Zapisano {len(df)} transakcji do pliku {output_file}")


if __name__ == "__main__":

    # Jak jest number_of_pages=None to sprawdza ile jest i scrapuje wszystkie 
    # Dostępne typy transakcji: 'transakcja', 'transakcjaBonifikata', 'transakcjaHurt'
    scrapped_data = scrape_deweloperuch(number_of_pages=None, transaction_type=['transakcja'])
    
    save_to_csv(scrapped_data, output_file="deweloperuch_1.csv")
