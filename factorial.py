from prefect import flow

@flow
def fact(n):
    if (n < 0):
        return None
    if(n == 0 or n == 1):
        return 1
    else:
        return n * fact(n-1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Factorial of a number")
    parser.add_argument('--number', type=int, required=True, help="Enter an integer number")
    args = parser.parse_args()

    result = fact(n=args.number)
    print(f'Factorial of {args.number} is {result}')