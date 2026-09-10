import sys
from pathlib import Path 

sys.path.insert(0, str(Path(__file__).parent.parent))

from source.data_processors.data_processor import DataProcessor

def main():
    processor = DataProcessor(
        input_dir = 'data/raw/',
        output_dir = 'data/processed/',
        min_bars_per_day = 40,
        fat_finger_window = 11,
        fat_finger_threshold = 0.03,
        max_return = 5.0
    )

    results = processor.process_all('all_prices.csv')
    print('MOEX data processed✅')

if __name__ == '__main__':
    main()