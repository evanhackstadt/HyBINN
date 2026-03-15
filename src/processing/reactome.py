import pandas as pd


# BUILD REACTOME MAP
# https://download.reactome.org/95/Ensembl2Reactome_All_Levels.txt
def build_reactome_map(path):
    """
    Creates a mapping between Ensembl IDs and Reactome Pathways
    
    Args:
        path (str): filepath to the Ensembl2Reactome text file from Reactome
    
    Returns:
        map (dict): given the Ensembl ID of a gene, what Reactome pathways does it map to
    """
    
    df = pd.read_csv(path, delimiter='\t',
                     names=["EnsemblID", "ReactomePathwayID", "URL", 
                            "PathwayName", "Evidence", "Species"])
    
    print("--RAW--")
    print(df.shape)
    print(df.head())
    
    # Filter to only human pathways
    df2 = df[df["Species"] == "Homo sapiens"]
    
    print("--HUMAN--")
    print(df2.head())
    print(df2.shape)
    
    # Initial filter to pathways with 15 ≤ n_genes ≤ 300
    # Will perform this filter again based on dataset genes
    pathway_sizes = df2.groupby("ReactomePathwayID")["EnsemblID"].nunique()
    valid_pathways = pathway_sizes[(pathway_sizes >= 15) & (pathway_sizes <= 300)].index
    df3 = df2[df2["ReactomePathwayID"].isin(valid_pathways)]

    print("--COUNTS--")
    print(df3.shape)
    print(df3.head())
    
    # Filter to only EnsemblIDs we are using
    indices_to_drop = [row.Index for row in df3.itertuples()
                       if row.EnsemblID.startswith("ENSG")]
    df4 = df3.drop(index=indices_to_drop)
    
    print("--IDs--")
    print(df4.shape)
    print(df4.head())
    
    # Aggregate pathways per gene
    pathway_map = df4.groupby("EnsemblID")["ReactomePathwayID"].apply(list).to_dict()
    
    return pathway_map


# BUILD SPARSE MASK MATRIX
# This will zero-out weights between genes and pathways they aren't part of
def build_mask_matrix():
    return