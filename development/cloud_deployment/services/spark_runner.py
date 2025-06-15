import sys
import argparse
import report

def main():
    parser = argparse.ArgumentParser(description='Run Spark X-Ray Pipeline')
    parser.add_argument('--max_images', type=int, required=True, help='Maximum number of images to process')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Validate arguments
    assert 0 < args.max_images <= 101, f"max_images must be between 1 and 101, got {args.max_images}"
    assert 1 < args.batch_size <= 20, f"batch_size must be between 2 and 20, got {args.batch_size}"
    
    print(f"Starting pipeline with max_images={args.max_images}, batch_size={args.batch_size}")
    
    try:
        result = report.main(max_images_user=args.max_images, batch_size_user=args.batch_size)
        print(f"Pipeline completed successfully: {result}")
        return 0
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
