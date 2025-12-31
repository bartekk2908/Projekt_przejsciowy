import kagglehub


def data_download(dataset_name):
    """ Funkcja pobiera dataset do folderu .cache i zwraca ścieżkę """
    
    print("Pobieranie danych.")
    path = kagglehub.dataset_download(dataset_name)
    print(f"Dane pobrane do: {path}")
    return path
