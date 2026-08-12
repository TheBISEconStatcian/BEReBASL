import pandas as pd

if __name__ == "__main__":
    path_to_excel = 'C:/Users/mateo/Documents/Camino a Dios/Camino/2. Gemeinschaft Maria Frieden.xlsx'

    gem = pd.read_excel(path_to_excel, skiprows=2)

    mask_can_prepare = gem["Kann normalerweise?"] & (gem["Ist weg?"] != "Ja")

    namen_can_prepare: pd.Series = gem.loc[mask_can_prepare]["Vorname"]
    print(namen_can_prepare.values)

    group_size = 6
    groups = 2

    preparations = namen_can_prepare.sample(n=group_size*groups).values.reshape(groups, group_size)
    daten = ["25.07.", "08.08."]

    for group, d in zip(preparations, daten):
        print("Für Sa. den", d)
        for member in group:
            print("-", member)

        print()
