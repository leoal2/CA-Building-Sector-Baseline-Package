import os
import shutil
import glob
from pathlib import Path
import yaml
import re
import pandas as pd

# Define system paths
user_profile = os.environ.get("USERPROFILE")
local_appdata = os.environ.get("LOCALAPPDATA")

# Dynamically set R_HOME
home = os.path.join(user_profile, "AppData", "Local", "Programs", "R")
r_version = os.listdir(home)[0]
rfolder = os.path.join(home, r_version)
os.environ["R_HOME"] = rfolder
os.environ["PATH"] = rfolder + "/bin/x64" + ";" + os.environ["PATH"]

# Dynamically set STATEIOR_DATADIR
stateio_dir = os.path.join(local_appdata, "stateio")
os.environ["STATEIOR_DATADIR"] = stateio_dir

# Ensure stateio directory exists
os.makedirs(stateio_dir, exist_ok=True)

# Check and copy pre-generated .rds files if missing
repo_stateio_dir = os.path.join(os.path.dirname(__file__), "stateio_data")
for rds_file in glob.glob(os.path.join(repo_stateio_dir, '*.rds')):
    filename = os.path.basename(rds_file)
    target_file = os.path.join(stateio_dir, filename)
    if not os.path.exists(target_file):
        print(f"Copying {filename} to {stateio_dir}")
        shutil.copy2(rds_file, target_file)

import rpy2.robjects.packages as rpackages
import rpy2.robjects as ro
ro.r(f'Sys.setenv(STATEIOR_DATADIR = "{stateio_dir.replace(os.sep, "/")}")')
print("STATEIOR_DATADIR in R is:", ro.r('Sys.getenv("STATEIOR_DATADIR")')[0])

from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

model_path = os.path.join(rfolder, "library", "useeior", "extdata", "modelspecs")
output_folder = os.path.join(user_profile, "Documents", "CBP")

# Import R's utility package
useeio = rpackages.importr('useeior')

#Available models
"""
model_list = [x for x in useeio.seeAvailableModels()]
print("List of Valid Models:")
print(model_list)"""

LOCAL_PARQUET_DIR = os.path.expanduser("~") + r"\AppData\Local\flowsa\FlowByActivity"

def r_to_pandas(r_object):
    
    if not isinstance(r_object, ro.vectors.FloatMatrix):
        raise Exception("An invalid rpy2 object has been selected. Must be a float matrix table. Passed {}".format(type(r_object)))
    
    columns = r_object.colnames
    indexes = r_object.rownames

    with localconverter(ro.default_converter + pandas2ri.converter):
      array = ro.conversion.rpy2py(r_object)

    return pd.DataFrame(array, index = indexes, columns=columns)

class USEEIOConfig():
    def __init__(self, filename, bea_year=2017, ghg_year=2022, detailed=True, state=None, preserve=False):
        self.name = filename
        self.filename = None
        self.model = os.path.splitext(str(filename))[0]
        self.yaml = self._read_yaml(filename)
        self.preserve = preserve  # flag to control overwriting

        # only apply setters if we want to overwrite the yaml
        if not preserve:
            self.set_bea(bea_year, detailed, state=state)
            self.set_ghg(ghg_year, state=state)
            self.set_state(state=state, ghg_year=ghg_year)

    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def _read_yaml(self, filename):
        file = os.path.join(model_path, filename)
        self.filename = file

        if not os.path.exists(file):
            raise FileNotFoundError(f"ERROR: The YAML file '{filename}' does not exist in '{model_path}'. Please create or copy it before running the script.")
        
        print(f"Using existing YAML file: {file}")
        
        with open(file, 'r') as f:
            yaml_file = yaml.safe_load(f)
            yaml_file["Model"] = self.model

        return yaml_file



    def set_state(self, state=None, ghg_year=None):
        """       
        Defaults to total US, unless a specific state is defined.
        State must be two-letter US code.
        """
        #TODO: Should really add error checking
        
        if state is None:
            ModelRegionAcronyms = ["US"]
            self.yaml["IODataSource"] = "BEA"
            if "IODataVersion" in self.yaml.keys():
                del self.yaml["IODataVersion"]
                self.yaml["BasePriceType"] = "PRO"
            
        elif len(state)==2:
            ModelRegionAcronyms = [f"US-{state.upper()}","RoUS"]
            year, version = list_state_files(return_tuple=True).pop()
            if ghg_year:
                year = ghg_year
            self.set_bea(int(year), state=state)
            self.yaml["IODataSource"] = "stateior"
            self.yaml["BaseIOLevel"] = "Summary"
            self.yaml["IODataVersion"] = version
            self.yaml["BasePriceType"] = "PRO"
            self.set_ghg(year, state=state)
            print("Model Region Acroynsms Identified: {}".format(ModelRegionAcronyms))
            
        else:
            raise Exception("Invalid State provided. Must be 2 letters.")
            
        self.yaml["ModelRegionAcronyms"] = ModelRegionAcronyms
        self.yaml['SatelliteTable']['GHG']['Locations'] = ModelRegionAcronyms
        self.save()
    
    
    def set_ghg(self, year, state=None):
        """
        Sets the year for the GHG inventory used in the YAML file used by EEIOr

        Parameters
        ----------
        year : int
            The year for the GHG inventroy file to use.

        Raises
        ------
        Exception
            If the year provided cannot be found on the amazon datacloud EPA posts data to.

        """
        valid_ghg = list_parquet_files(state=state)
        file_name = None
        pattern = re.compile("(^.*_)([0-9]{4})(_.*$)")
        years = [pattern.match(x).groups()[1] for x in valid_ghg]
        for entry in valid_ghg:
            if str(year) == pattern.match(entry).groups()[1]:
                file_name = entry
        
        if file_name is None:
            raise Exception("Invalid data entered for GHG inventory. Choose from {}".format(years))
            
        self.yaml["SatelliteTable"]["GHG"]["SectorListYear"] = self.yaml["BaseIOSchema"]
        #if not self.preserve:
        #    self.yaml["SatelliteTable"]["GHG"]["StaticFile"] = r"flowsa/FlowBySector/" + file_name

        self.yaml["SatelliteTable"]["GHG"]["DataYears"] = [year]
        
        if state:
            self.yaml['SatelliteTable']['GHG']['Locations'] = [f"US-{state.upper()}","RoUS"]
            
        self.save()




        
    def set_bea(self, year, detailed=True, state=None):
        """
        Sets the year for the BEA document in the YAML file used by EEIOr.

        Parameters
        ----------
        year : int
            The year for the BEA Input-Output matrix.
        detailed : Boolean, optional
            DESCRIPTION. The default is True, which will pull the 402 industry input output tables.
            If set to false it pulls the summary table, which is only 71 industry NAICS.

        Raises
        ------
        Exception
            Invalid BEA data or detail-level specified.

        Returns
        -------
        None.

        """
        
        ##TODO:  Update the statically assigned years. 
        # Can pull from the BEA API to get the list of valid years
        valid_years = []
        valid_detailed_years = [2007, 2012, 2017]
        valid_summary_years = [x for x in range(1997, 2023)]
        if year in valid_summary_years:
            detail = "Summary"
            valid_years = valid_summary_years
        
        if year in valid_detailed_years and state is None:
            detail = "Detail"
            self.set_state()
            valid_years = valid_detailed_years
        
        if detailed is False:
            # If a detailed false flag is passed, then it overwrites the detail even if
            detail = "Summary"
            
        if year not in valid_years:
            raise Exception("Year '{}' is not valid. Please choose from: {}".format(year, valid_years))
        if year<2017:
            self.yaml['BaseIOSchema'] = year
        else:
            self.yaml['BaseIOSchema'] = 2017  # There are some issues with how GHG inventory/flowsa data is mapped.
        
        self.yaml['IOYear'] = year
        self.yaml["BaseIOLevel"] = detail
        self.yaml["SatelliteTable"]["GHG"]["SectorListYear"] = year
        self.save()
        
        
    def save(self):
        with open(self.filename, 'w') as yaml_file:
            yaml.dump(self.yaml, yaml_file, default_flow_style=False)

class Results():
    def __init__(self, name="temp_file", bea_year=2017, ghg_year=2022, region=None, detailed=False, preserve=False):
        self.N_d = None
        self.U_d = None
        self.N = None
        self.U = None
        self.A = None
        self.consumption = {}
        self.consumption_domestic = {}
        self.crosswalk = None
        self._model = None
        if region is None:
            self.type = "National"
        else:
            self.type = region

        self._make_matrices(name, bea_year, ghg_year, region, detailed, preserve)
    
    def get_inputs(self, sector, domestic=False):
        """
        

        Parameters
        ----------
        sector : string, sector code -- the BEA specific NAICS code identifier
        domestic : boolean, optional
            WIll use the domestic-only parameter if set to true. The default is False.

        Returns
        -------
        dataframe. Returns a dataframe of all inputs into the sector that are non-zero.
            

        """
        if domestic:
            use = self.U_d
        else:
            use = self.U
        return use[use[sector]>0][sector]
    
    def get_ef(self, sector, domestic=False):
        if domestic:
            ef = self.N_d
        else:
            ef = self.N
        
        if sector in ef:
            return ef[sector]
        else:
            raise Exception(f"{sector} is not a valid sector.")
    
    def get_use(self, naics,domestic=False):
        if domestic:
            use = self.U_d
        else:
            use = self.U
        sectors = [x for x in use.columns if x.startswith(naics)]
        
        consumption = sum([use.loc[x].sum() for x in sectors])
        return consumption
            
    def to_file(self, outputfolder=None):
        """
        

        Parameters
        ----------
        outputfolder : string, optional
            The default output folder is None.  Will create an excel file with all relevant matrices for the created EEIO model

        Returns
        -------
        None.

        """
        
        if outputfolder is None:
            specs = self._model.rx2["specs"]
            IOYear = specs.rx2["IOYear"][0]
            GHGYear = specs.rx2["SatelliteTable"].rx2["GHG"].rx2["DataYears"][0]
            region = specs.rx2["ModelRegionAcronyms"][0]
            outputfolder = f"IO{IOYear}_GHG{GHGYear}_{region}"
        
        if not os.path.exists(outputfolder):
            print(f"Creating new folder: {outputfolder}")
            os.mkdir(outputfolder)
        print(f"Writing excel file to {outputfolder}")
        useeio.writeModeltoXLSX(self._model, outputfolder)

      
    def _make_crosswalk(self, model):
        table = model.rx2["crosswalk"]
        summary = [x for x in table.rx2["BEA_Summary"]]
        detail = [x for x in table.rx2["BEA_Detail"]]
        df = pd.DataFrame()
        df["Summary"] = summary
        df["Detail"] = detail
        df.index = detail
        df = df.drop_duplicates()
        
        return df
        
    def _make_matrices(self, name, bea_year=2017, ghg_year=2022, region=None, detailed=True, preserve=False):
        if ".yml" not in name:
            name = name + ".yml"

        y = USEEIOConfig(name, bea_year, ghg_year, detailed, state=region, preserve=preserve)

        model = useeio.buildModel(y.model)
        self._model = model

        self.crosswalk = self._make_crosswalk(model)
        self.N_d = r_to_pandas(model.rx2["N_d"])
        self.U_d = r_to_pandas(model.rx2["U_d"])
        self.N = r_to_pandas(model.rx2["N"])
        self.U = r_to_pandas(model.rx2["U"])
        self.L = r_to_pandas(model.rx2["L"])
        self.L_d = r_to_pandas(model.rx2["L_d"])
        self.D = r_to_pandas(model.rx2["D"])
        self.V = r_to_pandas(model.rx2["V"])
        self.A = r_to_pandas(model.rx2["A"])

        if "x" in model.names:
            x_vector = model.rx2["x"]
            self.x = pd.DataFrame({"x": list(x_vector)}, index=x_vector.names)
        else:
            self.x = None

        demandvectors = model.rx2["DemandVectors"].rx2["vectors"]
        if region is not None:
            table = f"US-{region.upper()}_Consumption_Complete"
            table_domestic = f"US-{region.upper()}_Consumption_Domestic"
            rous = True
            default_region = region
        else:
            table = "US_Consumption_Complete"
            table_domestic = "US_Consumption_Domestic"
            rous = False
            default_region = "Default"

        matching_tables = [x for x in demandvectors.names if table in x]
        matching_tables_domestic = [x for x in demandvectors.names if table_domestic in x]

        if not matching_tables or not matching_tables_domestic:
            raise ValueError(
                f"Could not find demand vector '{table}' or '{table_domestic}' in available names:\n{list(demandvectors.names)}"
        )

        table_name = matching_tables[0]
        table_name_domestic = matching_tables_domestic[0]

        demand = demandvectors.rx2[table_name]
        self.consumption[default_region] = pd.DataFrame(demand, index=demand.names, columns=["TotalDemand"])

        demand_d = demandvectors.rx2[table_name_domestic]
        self.consumption_domestic[default_region] = pd.DataFrame(demand_d, index=demand_d.names, columns=["TotalDemand"])

        if rous:
            table = "RoUS_Consumption_Complete"
            table_domestic = "RoUS_Consumption_Domestic"
            table_name = [x for x in demandvectors.names if table in x][0]
            table_name_domestic = [x for x in demandvectors.names if table_domestic in x][0]
            rousdemand = demandvectors.rx2[table_name]
            rousdemand_d = demandvectors.rx2[table_name_domestic]
            self.consumption["RoUS"] = pd.DataFrame(rousdemand, index=rousdemand.names, columns=["TotalDemand"])
            self.consumption_domestic["RoUS"] = pd.DataFrame(rousdemand_d, index=rousdemand_d.names, columns=["TotalDemand"])


