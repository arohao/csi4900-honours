import pandas as pd
import feature_Generator_with_brands as pfe


datasetname = "Mapping_File.xlsx"  #please give the path of the file
datasetpath = r"All_HTML"   #please give the path of the folder

def create_URLDict():
    dataframe1 = pd.read_excel(
        "{0}/{1}".format(datasetpath, datasetname), usecols=['serial_number', 'url','label'])
    print(dataframe1.shape)
    URLdict = dataframe1.to_dict(orient='tight')
    return URLdict


# Example usage
if __name__ == "__main__":
    URLdict = create_URLDict()
    #pfe.main(URLdict,"Feature Extraction\Our Dataset")
    #pfe.main(URLdict,"Feature Extraction\PublicDataset_Zenodo")
    pfe.main(URLdict,"All_HTML")
