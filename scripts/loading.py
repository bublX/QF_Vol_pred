import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from source.data_processors.data_loader import DataLoader

def main():

    tickers = ['VTBR', 'SBER', 'LKOH', 'ROSN', 'NVTK',
               'TATN', 'GAZP', 'GMKN', 'NLMK', 'MGNT']

    loader = DataLoader(
        tickers = tickers,
        start_year = 2014,
        end_year = 2025,
        interval = 10,
        delay = 0.01,
        save_dir = 'data/raw/', 
        filename = 'all_prices.csv'
    )

    df = loader.load_all()
    print('MOEX data loaded✅')

if __name__ == '__main__':
    main()