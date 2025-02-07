import pandas as pd

input_file = 'DATA/convert_files/P_DEMO.xpt'
output_file = 'DATA/convert_files/P_DEMO.xlsx'



try:
    df = pd.read_sas(input_file, format='xport', encoding='ISO-8859-1')
    df.to_excel(output_file, index=False)
    print(f"Success! {output_file}")
except UnicodeDecodeError:
    print("Failed to read the file due to encoding issues.")


#
# import pyreadstat
#
# input_file = 'DSII.XPT'
# output_file = 'DSII.xlsx'
# encodings = ['ISO-8859-1', 'cp1252', 'latin1']  # add more if needed
#
# for enc in encodings:
#     try:
#         print(f"Trying encoding: {enc}")
#         df, meta = pyreadstat.read_xport(input_file)
#         df.to_excel(output_file, index=False)
#         print(f"Success with encoding: {enc}")
#         break
#     except UnicodeDecodeError:
#         print(f"Failed with encoding: {enc}")
