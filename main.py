import data_loader

def main():

    dl = data_loader.DataLoader()
    ds = dl.night_to_dataset()

    for elem in ds.take(1):
        print(elem[0].shape)
        print(elem[1].shape)



if __name__ == '__main__':
    main()

