import pandas as pd


def check_data_distribution(csv_path, column_name, task='classification'):
    """Check and print data distribution"""
    df = pd.read_csv(csv_path)
    print(f"\nDataset shape: {df.shape}")
    
    if task == 'classification':
        ratio = (df[column_name] == 1).sum() / (df[column_name] == 0).sum()
        counts = df[column_name].value_counts()
        print(f"Class distribution:\n{counts}")
        print(f"Ratio (positive/negative): {ratio:.3f}")
    elif task == 'regression':
        print(f"Target variable: {column_name}")
        print(f"Mean: {df[column_name].mean():.2f}")
        print(f"Std: {df[column_name].std():.2f}")
        print(f"Min: {df[column_name].min():.2f}")
        print(f"Max: {df[column_name].max():.2f}")
        print(f"Median: {df[column_name].median():.2f}")
    
    return df
