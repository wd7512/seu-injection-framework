
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def read_results_by_index(results_dir):
    # df1 = pd.read_json("results_idx1_2025-06-16_18-49-15.json")
    # df1['index'] = 1

    # df0 = pd.read_json("results_2025-06-16_13-48-57_idx0.json")
    # df1 = pd.read_json("results_2025-06-16_13-46-48_idx1.json")
    # df3 = pd.read_json("results_2025-06-16_13-47-20_idx3.json")
    # df5 = pd.read_json("results_2025-06-16_13-47-21_idx5.json")

    # df0['index'] = 0
    # df1['index'] = 1
    # df3['index'] = 3
    # df5['index'] = 5

    # Combine them into one dataframe
    # df_all = pd.concat([df0, df1, df3, df5], ignore_index=True)
    # df_all = pd.concat([df1], ignore_index=True)

    # Load or assume you have df0, df1, df3 already
    # For this example, assume they're already defined
    idx2results = dict()
    directory = results_dir  # "." # assume results file are in current dir, may take a cli argument
    for file_name in os.listdir(directory):
        if file_name.startswith('results_idx') and file_name.endswith('.json'):
            # Parse name to get index
            # results_idx1_2025-06-16_18-49-15.json
            tokens = file_name.split("_")
            idx_term = tokens[1]  # idx1
            idx_str = idx_term.replace("idx", "")
            idx = int(idx_str)
            if idx not in idx2results:
                idx2results[idx] = list()
            results_files = idx2results[idx]
            # Need full path to open file
            file_path = os.path.join(directory, file_name)
            results_files.append(file_path)

    # Label the index for each dataframe
    idx2df = dict()
    for idx in idx2results.keys():
        results_files = idx2results[idx]
        df = None
        for file_path in results_files:
            if df is None:
                print(f"[{idx}] new dataframe for {file_path}")
                df = pd.read_json(file_path)
            else:
                print(f"[{idx}] concat dataframe for {file_path}")
                df_other = pd.read_json(file_path)
                df = pd.concat([df, df_other], axis=0, ignore_index=True)
        idx2df[idx] = df

    return idx2df


def main():
    if len(sys.argv) != 2:
        print("Usage <dir containing results json files>")
        print("Example python results.py ./results")
        return
    results_dir = sys.argv[1]

    # Read all the results files for an index into one dataframes, do for each index
    idx2df = read_results_by_index(results_dir)

    # Combine all dataframes into onde dataframe and add the index as a column
    dfs = list()
    for idx in idx2df.keys():
        df = idx2df[idx]
        # Add the index to the dataframe
        df['bitidx'] = idx
        print(f"[{idx}] = {len(df)} rows")
        dfs.append(df)
    # Dataframe with all
    df_all = pd.concat(dfs, ignore_index=True)

    # df_all = None
    # for idx in idx2df.keys():
    #     df = idx2df[idx]
    #     # Add the index to the dataframe
    #     df['bitidx'] = idx
    #     print(f"[{idx}] = {len(df)} rows")
    #     if df_all is None:
    #         df_all = df
    #     else:
    #         df_all = pd.concat([df_all, df], axis=0, ignore_index=True)

    df_all.sort_values(by='bitidx', inplace=True)

    # Critical, make bitidx str, does odd things when it is an integer
    df_all['bitidx'] = df_all['bitidx'].astype(str)

    print(df_all.head())
    print(f"Indices {df_all['bitidx'].unique()}")
    print(f"Length {len(df_all)}")



    # Extract numeric layer number (for sorting and group plots)
    df_all['layer'] = df_all['layer_name'].str.extract(r'encoder_layer_(\d+)').astype(float)

    # Compute absolute change in weight
    df_all['abs_change'] = abs(df_all['value_after'] - df_all['value_before'])

    # Set Seaborn theme
    sns.set_context("paper")
    sns.set(style="whitegrid")

    # Plot 1: Stripplot - layer_name vs criterion_score
    plt.figure(figsize=(14, 6))
    sns.stripplot(data=df_all, x='layer_name', y='abs_change', hue='bitidx', jitter=True, dodge=True)
    plt.xticks(rotation=90)
    plt.title("Criterion Score by Layer Name")
    plt.tight_layout()
    plt.savefig(f"fig_stripplot.pdf", dpi=300)  # Save as publication-quality PDF
    plt.show()

    # Plot 2: Lineplot - index vs criterion_score (row index as x-axis)
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df_all.reset_index(), x='bitidx', y='abs_change', hue='bitidx', marker='o')
    plt.title("Criterion Score by Dataset Index")
    plt.tight_layout()
    plt.savefig(f"fig_lineplot.pdf", dpi=300)  # Save as publication-quality PDF
    plt.show()

    # # Plot 3: Boxplot + Swarmplot by numeric layer
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(data=df_all, x='layer', y='criterion_score', hue='index', palette="pastel")
    # sns.swarmplot(data=df_all, x='layer', y='criterion_score', hue='index', dodge=True, color=".25", alpha=0.6)
    # plt.title("Distribution of Criterion Score by Layer Number")
    # plt.legend([],[], frameon=False)  # Hide duplicate legend
    # plt.tight_layout()
    # plt.savefig(f"fig_swarmplot.pdf", dpi=300)  # Save as publication-quality PDF
    # plt.show()
    #
    # # Plot 4: Barplot - Bottom 5 Criterion Scores per Index
    # #top_n = df_all.groupby('index').apply(lambda df: df.nlargest(5, 'criterion_score')).reset_index(drop=True) \
    # bottom_n = df_all.groupby('index').apply(lambda df: df.nsmallest(5, 'criterion_score')).reset_index(drop=True)
    #
    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=bottom_n, x='layer_name', y='criterion_score', hue='index')
    # plt.xticks(rotation=90)
    # plt.title("Bottom 5 Criterion Scores per Index")
    # plt.tight_layout()
    # plt.savefig(f"fig_barplot.pdf", dpi=300)  # Save as publication-quality PDF
    # plt.show()
    #
    # # # Plot 5: Scatterplot - abs_change vs criterion_score
    # # plt.figure(figsize=(10, 6))
    # # #sns.scatterplot(data=df_all, x='abs_change', y='criterion_score', hue='index')
    # # sns.scatterplot(data=df_all, x='criterion_score', y='abs_change', hue='index')
    # # plt.xscale('log')
    # # plt.title("Absolute Value Change vs Criterion Score")
    # # plt.tight_layout()
    # # plt.show()
    #
    # # Plot 6: Heatmap - Mean criterion score per index/layer
    # heatmap_data = df_all.groupby(['index', 'layer'])['criterion_score'].mean().unstack()
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".3f")
    # plt.title("Mean Criterion Score by Index and Layer")
    # plt.ylabel("Index")
    # plt.xlabel("Layer Number")
    # plt.tight_layout()
    # plt.savefig(f"fig_heatmap.pdf", dpi=300)  # Save as publication-quality PDF
    # plt.show()



if __name__ == "__main__":
    main()


"""
INDEX: 0
  tensor_location  criterion_score                                         layer_name  value_before  value_after
0      (491, 367)           0.7608  encoder.layers.encoder_layer_4.self_attention....     -0.004571     0.004571
1      (845, 190)           0.7608        encoder.layers.encoder_layer_4.mlp.0.weight      0.019360    -0.019360
2     (3044, 744)           0.7608        encoder.layers.encoder_layer_4.mlp.0.weight     -0.005737     0.005737
3      (424, 369)           0.7608        encoder.layers.encoder_layer_5.mlp.0.weight     -0.002396     0.002396
4     (435, 2481)           0.7608        encoder.layers.encoder_layer_5.mlp.3.weight     -0.012972     0.012972
5      (89, 1431)           0.7608        encoder.layers.encoder_layer_7.mlp.3.weight     -0.035871     0.035871
6     (313, 1800)           0.7608        encoder.layers.encoder_layer_8.mlp.3.weight     -0.010392     0.010392
7     (1589, 600)           0.7608       encoder.layers.encoder_layer_10.mlp.0.weight     -0.035000     0.035000

INDEX: 1
  tensor_location  criterion_score                                         layer_name  value_before   value_after
0     (386, 1075)           0.0010        encoder.layers.encoder_layer_0.mlp.3.weight      0.030956  1.053373e+37
1     (1318, 398)           0.7342  encoder.layers.encoder_layer_1.self_attention....      0.076440  2.601105e+37
2       (1954, 8)           0.0010  encoder.layers.encoder_layer_1.self_attention....      0.000746  2.539693e+35
3      (30, 1434)           0.0010        encoder.layers.encoder_layer_3.mlp.3.weight     -0.008814 -2.999218e+36
4     (1973, 393)           0.4438        encoder.layers.encoder_layer_6.mlp.0.weight     -0.009877 -3.361003e+36
5     (2677, 423)           0.0846        encoder.layers.encoder_layer_7.mlp.0.weight      0.028580  9.725167e+36
6      (234, 286)           0.0010  encoder.layers.encoder_layer_10.self_attention...     -0.026520 -9.024126e+36
7     (1187, 453)           0.3660       encoder.layers.encoder_layer_10.mlp.0.weight      0.029416  1.000958e+37

INDEX: 3
  tensor_location  criterion_score                                         layer_name  value_before   value_after
0      (591, 120)           0.7608        encoder.layers.encoder_layer_0.mlp.0.weight     -0.028345 -6.599487e-12
1     (239, 2508)           0.7608        encoder.layers.encoder_layer_0.mlp.3.weight      0.029454  6.857812e-12
2     (1035, 274)           0.7608  encoder.layers.encoder_layer_1.self_attention....     -0.013610 -3.168768e-12
3     (2825, 520)           0.7608        encoder.layers.encoder_layer_1.mlp.0.weight      0.006175  1.437708e-12
4     (1233, 702)           0.7608  encoder.layers.encoder_layer_2.self_attention....      0.005628  1.310479e-12
5     (535, 1306)           0.7608        encoder.layers.encoder_layer_2.mlp.3.weight      0.004542  1.057463e-12
6     (1763, 565)           0.7608        encoder.layers.encoder_layer_4.mlp.0.weight      0.005955  1.386611e-12
7      (489, 154)           0.7608        encoder.layers.encoder_layer_4.mlp.3.weight     -0.030154 -7.020747e-12
8       (50, 565)           0.7608        encoder.layers.encoder_layer_5.mlp.3.weight      0.006350  1.478521e-12
"""