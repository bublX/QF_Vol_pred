import sys
from pathlib import Path 

sys.path.insert(0, str(Path(__file__).parent.parent))

from source.data_processors.data_processor import DataProcessor

def main():
    processor = DataProcessor(  
        input_dir = 'data/raw/',
        output_dir = 'data/processed/',
        hampel_window = 5,
        hampel_sigma = 3.0,
        max_return = 5
    )

    results = processor.process_all('all_prices.csv')
    print('MOEX data processed✅')

if __name__ == '__main__':
    main()