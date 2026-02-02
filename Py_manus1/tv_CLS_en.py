import os, csv
import pandas as pd 
 
# don't run this line in ipython. It allows us to run python TV/tv_CLS_train.py from the root directory on Linux
import sys
in_ipython = 'get_ipython' in globals()
if not in_ipython:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


### Set arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--HSV', type=str, default="2", help='1 or 2')
parser.add_argument('--sample_set', type=str, help='e.g., 201907_202505')
parser.add_argument('--modelSuffix', type=str, help='e.g., _201608-201702_Tr12_pretrained')
parser.add_argument('--cv', action="store_true", help='cross validaton results')
args = parser.parse_known_args()[0]
print(args, '\n')

html_header="""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Image Viewer</title>
        <style>
            .image-row { 
                display: flex; 
                flex-wrap: wrap;       /* Allows images to wrap to next line if many */
                flex-direction: row; 
                gap: 3px; 
                font-family: sans-serif; 
            }
            .image-item {
                flex: 0 0 auto;        /* Prevents stretching */
                width: 25px;           /* Adjusted from 50px to be slightly more visible */
            }
            img { 
                display: block; 
                width: 100%; 
                height: auto; 
                border: 1px solid #ccc; 
            }
        </style>
    </head>
    <body>        
    """

folder='Class_Label/pred/'

def en(df1,df2,df3,saveto, print_2by2=False):
    all_predicted_ar = pd.concat([df1['FinalPredicted'], df2['FinalPredicted'], df3['FinalPredicted']], axis=1)
    final_predicted = all_predicted_ar.mode(axis=1)[0] # find the most common value in each row
    # compare final_predicted with the ground truth, which is in the second column of the first dataframe
    ground_truth = df1['GroundTruth']
    accuracy = (final_predicted == ground_truth).sum()
    print(f"accuracy: {accuracy}/{len(ground_truth)} = {accuracy/len(ground_truth):.4f}")
    strip_ids = df1['StripID']

    if print_2by2:
        print("\n2x2 table:")
        print(pd.crosstab(ground_truth, final_predicted, rownames=['True'], colnames=['Predicted'], margins=True))

    # save ensemble results to CSV
    header = ["FinalPredicted", "GroundTruth", "StripID"]
    with open(folder + saveto, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in zip(final_predicted, ground_truth, strip_ids):
            writer.writerow(row)

    results_df = pd.DataFrame({
        "FinalPredicted": final_predicted,
        "GroundTruth": ground_truth,
        "StripID": strip_ids
    })
    return results_df


img_base_path = f"../../../Image/blots/{args.sample_set}/DET_dS_strips" 

if args.cv:
    rs = []
    for fold in range(5):
        print(f"\n\nFold {fold}")
        df1       = pd.read_csv(folder + f"{args.sample_set}/HSV{args.HSV}_Final_2classes_DET_dS_strips_{args.modelSuffix}_fold{fold}.csv")
        df2       = pd.read_csv(folder + f"{args.sample_set}/HSV{args.HSV}_Final_2classes_SEG_sS1_strips_v4_{args.modelSuffix}_fold{fold}.csv")
        df3       = pd.read_csv(folder + f"{args.sample_set}/HSV{args.HSV}_Final_2classes_SEG_sS1_strips_v6_{args.modelSuffix}_fold{fold}.csv")
        rs.append(en(df1,df2,df3, saveto=f"{args.sample_set}/HSV{args.HSV}_Final_2classes_ensemble_{args.modelSuffix}_fold{fold}.csv"))
    total_results = pd.concat(rs, ignore_index=True)

    # save all 5 fold results to a single CSV
    filename = f"{args.sample_set}/HSV{args.HSV}_Final_2classes_ensemble_{args.modelSuffix}"
    header = ["FinalPredicted", "GroundTruth", "StripID"]
    total_results.to_csv(folder + filename + ".csv", index=False)

    # save misclassified strips from 5 folds into a single html
    en_false_pos = total_results.loc[(total_results['FinalPredicted'] == 1) & (total_results['GroundTruth'] == 0), 'StripID'].tolist()
    en_false_neg = total_results.loc[(total_results['FinalPredicted'] == 0) & (total_results['GroundTruth'] == 1), 'StripID'].tolist()
    with open(folder + filename + "_miss.html", "w") as f:
        # Write the Header and Styles
        f.write(html_header)
        f.write(f'{filename} <br><br><br>')

        f.write(f'False positives:<br> {"<br> ".join(map(str, en_false_pos))} \n <div class="image-row">')
        for item in en_false_pos:
            # Assuming item is the filename without .png, e.g., '2021.10.28_ASCL_01_235'
            f.write(f'        <div class="image-item">\n')
            f.write(f'            <img src="{img_base_path}/{item}.png" title="{item}">\n')
            f.write(f'        </div>\n')
        f.write("</div><br><br><br>")

        f.write(f'False negatives:<br> {"<br> ".join(map(str, en_false_neg))} \n <div class="image-row">')
        for item in en_false_neg:
            # Assuming item is the filename without .png, e.g., '2021.10.28_ASCL_01_235'
            f.write(f'        <div class="image-item">\n')
            f.write(f'            <img src="{img_base_path}/{item}.png" title="{item}">\n')
            f.write(f'        </div>\n')
        f.write("</div>")

        f.write("</body></html>")

        
else:
    df1 = pd.read_csv(folder + f"{args.sample_set}/HSV{args.HSV}_Final_2classes_DET_dS_strips_{args.modelSuffix}.csv")
    df2 = pd.read_csv(folder + f"{args.sample_set}/HSV{args.HSV}_Final_2classes_SEG_sS1_strips_v4_{args.modelSuffix}.csv")
    df3 = pd.read_csv(folder + f"{args.sample_set}/HSV{args.HSV}_Final_2classes_SEG_sS1_strips_v6_{args.modelSuffix}.csv")
    en(df1,df2,df3,     saveto=f"{args.sample_set}/HSV{args.HSV}_Final_2classes_ensemble_{args.modelSuffix}.csv", print_2by2=True)
