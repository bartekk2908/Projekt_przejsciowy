import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random


def scrape_deweloperuch(total_pages = None):

    def _scrape_one_page():
        try:
            params = {
                "perPage": 100,
                "page": page
            }
            
            response = requests.get(BASE_URL, headers=HEADERS, params=params)
            
            if response.status_code != 200:
                print(f"Błąd przy stronie {page}: Status {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            


            # Szukamy głównej tabeli i jej wierszy wewnątrz <tbody>
            table = soup.find('table')
            if not table:
                print(f"Nie znaleziono tabeli na stronie {page}.")
                break
                
            rows = table.find_all('tr')

            # Pomijamy nagłówek tabeli (zazwyczaj pierwszy wiersz)
            if len(rows) > 0 and 'th' in str(rows[0]):
                rows = rows[1:]

            for row in rows:
                cols = row.find_all('td')
                # Upewniamy się, że wiersz ma dane (czasem są puste wiersze techniczne)
                if len(cols) >= 6: 
                    # Poniższe indeksy [0], [1] itd. zależą od kolejności kolumn na stronie.
                    # Musisz zweryfikować, czy kolejność na stronie to faktycznie:
                    # ID | Miasto | Metraż | Pokoje | Piętro | Cena
                    
                    record = {
                        "ID": cols[0].get_text(strip=True),       # Zakładam, że ID jest w 1 kolumnie
                        "Miasto": cols[1].get_text(strip=True),
                        "Metraz": cols[2].get_text(strip=True),
                        "Pokoje": cols[3].get_text(strip=True),
                        "Pietro": cols[4].get_text(strip=True),
                        "Cena": cols[5].get_text(strip=True)
                    }
                    all_data.append(record)



            print(f"Strona {page}/{total_pages} pobrana. Rekordów w sumie: {len(all_data)}")
            
            # Opóźnienie, żeby nie zablokowali IP
            time.sleep(random.uniform(2, 4))

        except Exception as e:
            print(f"Wystąpił błąd na stronie {page}: {e}")


    BASE_URL = "https://deweloperuch.pl/ceny-transakcyjne/polska/mieszkania"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7"
    }

    if not total_pages:
        total_pages = 401

    all_data = []

    print(f"Rozpoczynam scrapowanie.")
    for page in range(1, total_pages + 1):
        
    return all_data


def save_to_csv(data, output_file = "data2.csv",):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=';')
    print(f"Zapisano {len(df)} transakcji do pliku {output_file}")


if __name__ == "__main__":

    data = scrape_deweloperuch()
    save_to_csv(data)
