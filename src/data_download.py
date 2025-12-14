import kagglehub


def data_download(dataset_name):
    print("Pobieranie danych.")
    path = kagglehub.dataset_download(dataset_name)
    print(f"Dane pobrane do: {path}")
    return path
