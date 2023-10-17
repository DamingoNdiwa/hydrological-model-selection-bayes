from functools import partia
l
import dateutil.parser
import numpy as np
import pandas as pd

dayfirst_parse = partial(dateutil.parser.parse, dayfirst=True)

date = np.loadtxt("data/megala_creek_australia/date.txt",
                  converters={0: dayfirst_parse}, dtype=np.datetime64)
evapotranspiration = np.loadtxt(
    "data/megala_creek_australia/evapotranspiration.txt")
observed_discharge = np.loadtxt(
    "data/megala_creek_australia/observed_discharge.txt")
precipitation = np.loadtxt("data/megala_creek_australia/precipitation.txt")
temperature = np.loadtxt("data/megala_creek_australia/temperature.txt")

assert(len(date) == len(evapotranspiration) == len(
    observed_discharge) == len(precipitation) == len(temperature))

df = pd.DataFrame({"date": date, "evapotranspiration": evapotranspiration, "observed_discharge": observed_discharge,
    "precipitation": precipitation, "temperature": temperature})

df.to_pickle("data/megala_creek_australia.pkl.gz")


