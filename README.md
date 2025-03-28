# EU green liquids targets
## Overview
This script calculates EU green liquids targets for ammonia, methanol and jet fuel, in 2030 and 2035, based on the RED3 directive. We use two data input files, stored in the `inputdata/` directory. The first file, `EU_Transportenergy_data.xlsx`,
contains the energy consumption of the transport sector, according to [Eurostat's Energy Balances](https://ec.europa.eu/eurostat/databrowser/view/nrg_bal_c__custom_15815633/default/table?lang=en). The second data file, `H2EU_Industry_consumption_data.xlsx`,
contains the hydrogen consumption of the industry sector in the EU according to [Hydrogen Europe](https://observatory.clean-hydrogen.europa.eu/hydrogen-landscape/end-use/hydrogen-demand). For both data files, we use 2023 as a reference year.

## Usage
To run the code, clone this repository, navigate to the project repository and run the main script to analyse the data:
```
python analysis.py
```
This returns two figures for the EU's projected demand in green liquids for 2030 and 2035, as well as the analysed data saved as a csv under `EU_green_liquids_RED3Targets.csv`.

## License
This project is licensed under the MIT license.
