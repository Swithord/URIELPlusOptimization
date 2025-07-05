from urielplus import urielplus as uriel
import pandas as pd


def replace(imputed_values):
    u = uriel.URIELPlus()
    dep_df = pd.read_csv('langrank/dep/dep_updated.csv')
    el_df = pd.read_csv('langrank/el/el_updated.csv')
    mt_df = pd.read_csv('langrank/mt/mt_updated.csv')
    pos_df = pd.read_csv('langrank/pos/pos_updated.csv')

    selected_df = pd.DataFrame(imputed_values)
    
    syn_df = selected_df[[col for col in selected_df.columns if col.startswith("S_")]]
    inv_df = selected_df[[col for col in selected_df.columns if col.startswith("INV_")]]
    phon_df = selected_df[[col for col in selected_df.columns if col.startswith("P_")]]

    iso_to_glot_mapping = pd.read_csv('langrank/code_mapping.csv', index_col=0).to_dict()['glottocode']

    for index, row in dep_df.iterrows():
        target_lang = iso_to_glot_mapping[row['Target lang']]
        transfer_lang = iso_to_glot_mapping[row['Transfer lang']]
        lang_1_idx = selected_df.index.get_loc(target_lang)
        lang_2_idx = selected_df.index.get_loc(transfer_lang)
        dep_df.at[index, "SYNTACTIC"] = u._angular_distance(syn_df.iloc[lang_1_idx].to_numpy(), syn_df.iloc[lang_2_idx].to_numpy())
        dep_df.at[index, "INVENTORY"] = u._angular_distance(inv_df.iloc[lang_1_idx].to_numpy(), inv_df.iloc[lang_2_idx].to_numpy())
        dep_df.at[index, "PHONOLOGICAL"] = u._angular_distance(phon_df.iloc[lang_1_idx].to_numpy(), phon_df.iloc[lang_2_idx].to_numpy())
        dep_df.at[index, "FEATURAL"] = u._angular_distance(selected_df.iloc[lang_1_idx].to_numpy(), selected_df.iloc[lang_2_idx].to_numpy())

    dep_df.to_csv('langrank/dep/dep_selected.csv')

    for index, row in el_df.iterrows():
        target_lang = iso_to_glot_mapping[row['Target lang']]
        transfer_lang = iso_to_glot_mapping[row['Transfer lang']]
        lang_1_idx = selected_df.index.get_loc(target_lang)
        lang_2_idx = selected_df.index.get_loc(transfer_lang)
        el_df.at[index, "SYNTACTIC"] = u._angular_distance(syn_df.iloc[lang_1_idx].to_numpy(), syn_df.iloc[lang_2_idx].to_numpy())
        el_df.at[index, "INVENTORY"] = u._angular_distance(inv_df.iloc[lang_1_idx].to_numpy(), inv_df.iloc[lang_2_idx].to_numpy())
        el_df.at[index, "PHONOLOGICAL"] = u._angular_distance(phon_df.iloc[lang_1_idx].to_numpy(), phon_df.iloc[lang_2_idx].to_numpy())
        el_df.at[index, "FEATURAL"] = u._angular_distance(selected_df.iloc[lang_1_idx].to_numpy(), selected_df.iloc[lang_2_idx].to_numpy())

    el_df.to_csv('langrank/el/el_selected.csv')

    for index, row in mt_df.iterrows():
        target_lang = iso_to_glot_mapping[row['Target lang']]
        source_lang = iso_to_glot_mapping[row['Source lang']]
        transfer_lang = iso_to_glot_mapping[row['Transfer lang']]
        lang_1_idx = selected_df.index.get_loc(target_lang)
        lang_2_idx = selected_df.index.get_loc(transfer_lang)
        lang_3_idx = selected_df.index.get_loc(source_lang)
        mt_df.at[index, "SYNTACTIC"] = u._angular_distance(syn_df.iloc[lang_1_idx].to_numpy(), syn_df.iloc[lang_2_idx].to_numpy())
        mt_df.at[index, "INVENTORY"] = u._angular_distance(inv_df.iloc[lang_1_idx].to_numpy(), inv_df.iloc[lang_2_idx].to_numpy())
        mt_df.at[index, "PHONOLOGICAL"] = u._angular_distance(phon_df.iloc[lang_1_idx].to_numpy(), phon_df.iloc[lang_2_idx].to_numpy())
        mt_df.at[index, "FEATURAL"] = u._angular_distance(selected_df.iloc[lang_1_idx].to_numpy(), selected_df.iloc[lang_2_idx].to_numpy())
        mt_df.at[index, "SYNTACTIC_2"] = u._angular_distance(syn_df.iloc[lang_1_idx].to_numpy(), syn_df.iloc[lang_3_idx].to_numpy())
        mt_df.at[index, "INVENTORY_2"] = u._angular_distance(inv_df.iloc[lang_1_idx].to_numpy(), inv_df.iloc[lang_3_idx].to_numpy())
        mt_df.at[index, "PHONOLOGICAL_2"] = u._angular_distance(phon_df.iloc[lang_1_idx].to_numpy(), phon_df.iloc[lang_3_idx].to_numpy())
        mt_df.at[index, "FEATURAL_2"] = u._angular_distance(selected_df.iloc[lang_1_idx].to_numpy(), selected_df.iloc[lang_3_idx].to_numpy())

    mt_df.to_csv('langrank/mt/mt_selected.csv')

    for index, row in pos_df.iterrows():
        target_lang = iso_to_glot_mapping[row['Task lang']]
        aux_lang = iso_to_glot_mapping[row['Aux lang']]
        lang_1_idx = selected_df.index.get_loc(target_lang)
        lang_2_idx = selected_df.index.get_loc(aux_lang)
        pos_df.at[index, "SYNTACTIC"] = u._angular_distance(syn_df.iloc[lang_1_idx].to_numpy(), syn_df.iloc[lang_2_idx].to_numpy())
        pos_df.at[index, "INVENTORY"] = u._angular_distance(inv_df.iloc[lang_1_idx].to_numpy(), inv_df.iloc[lang_2_idx].to_numpy())
        pos_df.at[index, "PHONOLOGICAL"] = u._angular_distance(phon_df.iloc[lang_1_idx].to_numpy(), phon_df.iloc[lang_2_idx].to_numpy())
        pos_df.at[index, "FEATURAL"] = u._angular_distance(selected_df.iloc[lang_1_idx].to_numpy(), selected_df.iloc[lang_2_idx].to_numpy())

    pos_df.to_csv('langrank/pos/pos_selected.csv')

def replace_in_memory(imputed_values) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    u = uriel.URIELPlus()
    dep_df = pd.read_csv('langrank/dep/dep_updated.csv')
    el_df = pd.read_csv('langrank/el/el_updated.csv')
    mt_df = pd.read_csv('langrank/mt/mt_updated.csv')
    pos_df = pd.read_csv('langrank/pos/pos_updated.csv')

    selected_df = pd.DataFrame(imputed_values)
    
    syn_df = selected_df[[col for col in selected_df.columns if col.startswith("S_")]]
    inv_df = selected_df[[col for col in selected_df.columns if col.startswith("INV_")]]
    phon_df = selected_df[[col for col in selected_df.columns if col.startswith("P_")]]

    iso_to_glot_mapping = pd.read_csv('langrank/code_mapping.csv', index_col=0).to_dict()['glottocode']

    for index, row in dep_df.iterrows():
        target_lang = iso_to_glot_mapping[row['Target lang']]
        transfer_lang = iso_to_glot_mapping[row['Transfer lang']]
        lang_1_idx = selected_df.index.get_loc(target_lang)
        lang_2_idx = selected_df.index.get_loc(transfer_lang)
        dep_df.at[index, "SYNTACTIC"] = u._angular_distance(syn_df.iloc[lang_1_idx].to_numpy(), syn_df.iloc[lang_2_idx].to_numpy())
        dep_df.at[index, "INVENTORY"] = u._angular_distance(inv_df.iloc[lang_1_idx].to_numpy(), inv_df.iloc[lang_2_idx].to_numpy())
        dep_df.at[index, "PHONOLOGICAL"] = u._angular_distance(phon_df.iloc[lang_1_idx].to_numpy(), phon_df.iloc[lang_2_idx].to_numpy())
        dep_df.at[index, "FEATURAL"] = u._angular_distance(selected_df.iloc[lang_1_idx].to_numpy(), selected_df.iloc[lang_2_idx].to_numpy())

    for index, row in el_df.iterrows():
        target_lang = iso_to_glot_mapping[row['Target lang']]
        transfer_lang = iso_to_glot_mapping[row['Transfer lang']]
        lang_1_idx = selected_df.index.get_loc(target_lang)
        lang_2_idx = selected_df.index.get_loc(transfer_lang)
        el_df.at[index, "SYNTACTIC"] = u._angular_distance(syn_df.iloc[lang_1_idx].to_numpy(), syn_df.iloc[lang_2_idx].to_numpy())
        el_df.at[index, "INVENTORY"] = u._angular_distance(inv_df.iloc[lang_1_idx].to_numpy(), inv_df.iloc[lang_2_idx].to_numpy())
        el_df.at[index, "PHONOLOGICAL"] = u._angular_distance(phon_df.iloc[lang_1_idx].to_numpy(), phon_df.iloc[lang_2_idx].to_numpy())
        el_df.at[index, "FEATURAL"] = u._angular_distance(selected_df.iloc[lang_1_idx].to_numpy(), selected_df.iloc[lang_2_idx].to_numpy())

    for index, row in mt_df.iterrows():
        target_lang = iso_to_glot_mapping[row['Target lang']]
        source_lang = iso_to_glot_mapping[row['Source lang']]
        transfer_lang = iso_to_glot_mapping[row['Transfer lang']]
        lang_1_idx = selected_df.index.get_loc(target_lang)
        lang_2_idx = selected_df.index.get_loc(transfer_lang)
        lang_3_idx = selected_df.index.get_loc(source_lang)
        mt_df.at[index, "SYNTACTIC"] = u._angular_distance(syn_df.iloc[lang_1_idx].to_numpy(), syn_df.iloc[lang_2_idx].to_numpy())
        mt_df.at[index, "INVENTORY"] = u._angular_distance(inv_df.iloc[lang_1_idx].to_numpy(), inv_df.iloc[lang_2_idx].to_numpy())
        mt_df.at[index, "PHONOLOGICAL"] = u._angular_distance(phon_df.iloc[lang_1_idx].to_numpy(), phon_df.iloc[lang_2_idx].to_numpy())
        mt_df.at[index, "FEATURAL"] = u._angular_distance(selected_df.iloc[lang_1_idx].to_numpy(), selected_df.iloc[lang_2_idx].to_numpy())
        mt_df.at[index, "SYNTACTIC_2"] = u._angular_distance(syn_df.iloc[lang_1_idx].to_numpy(), syn_df.iloc[lang_3_idx].to_numpy())
        mt_df.at[index, "INVENTORY_2"] = u._angular_distance(inv_df.iloc[lang_1_idx].to_numpy(), inv_df.iloc[lang_3_idx].to_numpy())
        mt_df.at[index, "PHONOLOGICAL_2"] = u._angular_distance(phon_df.iloc[lang_1_idx].to_numpy(), phon_df.iloc[lang_3_idx].to_numpy())
        mt_df.at[index, "FEATURAL_2"] = u._angular_distance(selected_df.iloc[lang_1_idx].to_numpy(), selected_df.iloc[lang_3_idx].to_numpy())

    for index, row in pos_df.iterrows():
        target_lang = iso_to_glot_mapping[row['Task lang']]
        aux_lang = iso_to_glot_mapping[row['Aux lang']]
        lang_1_idx = selected_df.index.get_loc(target_lang)
        lang_2_idx = selected_df.index.get_loc(aux_lang)
        pos_df.at[index, "SYNTACTIC"] = u._angular_distance(syn_df.iloc[lang_1_idx].to_numpy(), syn_df.iloc[lang_2_idx].to_numpy())
        pos_df.at[index, "INVENTORY"] = u._angular_distance(inv_df.iloc[lang_1_idx].to_numpy(), inv_df.iloc[lang_2_idx].to_numpy())
        pos_df.at[index, "PHONOLOGICAL"] = u._angular_distance(phon_df.iloc[lang_1_idx].to_numpy(), phon_df.iloc[lang_2_idx].to_numpy())
        pos_df.at[index, "FEATURAL"] = u._angular_distance(selected_df.iloc[lang_1_idx].to_numpy(), selected_df.iloc[lang_2_idx].to_numpy())

    return dep_df, el_df, mt_df, pos_df