# Author: Anais Moller
#         Taken from https://github.com/anaismoller/reformatting_snippets
#
# created on 14 June 2019
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
from astropy.table import Table

__all__ = ['read_fits']

def read_fits(fname, drop_separators=False):
    """Load SNANA formatted data and cast it to a PANDAS dataframe.

    Parameters
    ----------
    fname: str 
        Path to photometric FITS files.
    drop_separators: bool (optional) 
        If -777 are to be dropped. Default is False.

    Returns
    -------
    df_head: pandas.DataFrame
         Dataframe containing header information.
    df_phot: pandas.DataFrame
        DataFrame containing photometry information (and ids).
    """

    # load photometry
    dat = Table.read(fname, format='fits')
    
    if len(dat) > 0:
        df_phot = dat.to_pandas()

        # failsafe
        if df_phot.MJD.values[-1] == -777.0:
            df_phot = df_phot.drop(df_phot.index[-1])
        if df_phot.MJD.values[0] == -777.0:
            df_phot = df_phot.drop(df_phot.index[0])

        # load header
        header = Table.read(fname.replace("PHOT", "HEAD"), format="fits")
        df_header = header.to_pandas()
        df_header["SNID"] = df_header["SNID"].astype(np.int32)

        # add SNID to phot for skimming
        arr_ID = np.zeros(len(df_phot), dtype=np.int32)

        # New light curves are identified by MJD == -777.0
        arr_idx = np.where(df_phot["MJD"].values == -777.0)[0]
        arr_idx = np.hstack((np.array([0]), arr_idx, np.array([len(df_phot)])))

        # Fill in arr_ID
        for counter in range(1, len(arr_idx)):
            start, end = arr_idx[counter - 1], arr_idx[counter]
            # index starts at zero
            arr_ID[start:end] = df_header.SNID.iloc[counter - 1]

        df_phot["SNID"] = arr_ID

        if drop_separators:
            df_phot = df_phot[df_phot.MJD != -777.000]

        return df_header, df_phot
    
    else:
        return pd.DataFrame([]), pd.DataFrame([])

