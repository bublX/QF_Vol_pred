import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.loading import main as run_loading
from scripts.processing import main as run_processing

def main():
    print("🔍 run_pipeline: начал работу")
    print("🔍 run_pipeline: вызываю run_loading")
    run_loading()
    print("🔍 run_pipeline: run_loading завершён")
    print("🔍 run_pipeline: вызываю run_processing")
    run_processing()
    print("🔍 run_pipeline: run_processing завершён")
    print('Pipeline done✅')

if __name__ == '__main__':
    main()