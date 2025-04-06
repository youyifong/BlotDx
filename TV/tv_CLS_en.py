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

parser.add_argument('--HSV', default=1, help='1 or 2')

args = parser.parse_known_args()[0]
print(args, '\n')

folder='Class_Label/'


def f1(df1,df2,df3,saveto, print_2by2=False):
    all_predicted_ar = pd.concat([df1['FinalPredicted'], df2['FinalPredicted'], df3['FinalPredicted']], axis=1)
    final_predicted = all_predicted_ar.mode(axis=1)[0] # find the most common value in each row
    # compare final_predicted with the ground truth, which is in the second column of the first dataframe
    ground_truth = df1['GroundTruth']
    accuracy = (final_predicted == ground_truth).sum()
    print(f"accuracy: {accuracy}/{len(ground_truth)} = {accuracy/len(ground_truth):.4f}")
    strip_ids = df1['StripID']
    print('Misclassified strips:')
    print(strip_ids[(final_predicted != ground_truth).values])

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


print("With transfer learning")
df1 = pd.read_csv(folder + "HSV" + args.HSV + "_validation_DET_dS_strips_pretrained.csv")
df2 = pd.read_csv(folder + "HSV" + args.HSV + "_validation_SEG_sS1_strips_v4_pretrained.csv")
df3 = pd.read_csv(folder + "HSV" + args.HSV + "_validation_SEG_sS1_strips_v6_pretrained.csv")
f1(df1,df2,df3,     saveto="HSV" + args.HSV + '_validation_3methods_pretrained_ensemble.csv')


df1 = pd.read_csv(folder + "HSV" + args.HSV + "_test_DET_dS_strips_pretrained.csv")
df2 = pd.read_csv(folder + "HSV" + args.HSV + "_test_SEG_sS1_strips_v4_pretrained.csv")
df3 = pd.read_csv(folder + "HSV" + args.HSV + "_test_SEG_sS1_strips_v6_pretrained.csv")
f1(df1,df2,df3,     saveto="HSV" + args.HSV + '_test_3methods_pretrained_ensemble.csv')


df1 = pd.read_csv(folder + "HSV" + args.HSV + "_alltest_DET_dS_strips_pretrained.csv")
df2 = pd.read_csv(folder + "HSV" + args.HSV + "_alltest_SEG_sS1_strips_v4_pretrained.csv")
df3 = pd.read_csv(folder + "HSV" + args.HSV + "_alltest_SEG_sS1_strips_v6_pretrained.csv")
f1(df1,df2,df3,     saveto="HSV" + args.HSV + '_alltest_3methods_pretrained_ensemble.csv', print_2by2=True)


print("\n\nWithout transfer learning")
df1 = pd.read_csv(folder + "HSV" + args.HSV + "_alltest_DET_dS_strips_nopretrained.csv")
df2 = pd.read_csv(folder + "HSV" + args.HSV + "_alltest_SEG_sS1_strips_v4_nopretrained.csv")
df3 = pd.read_csv(folder + "HSV" + args.HSV + "_alltest_SEG_sS1_strips_v6_nopretrained.csv")
f1(df1,df2,df3,     saveto="HSV" + args.HSV + '_alltest_3methods_nopretrained_ensemble.csv', print_2by2=True)

