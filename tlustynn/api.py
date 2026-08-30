"""
High-level API for TLUSTY NN atmosphere prediction.

Usage:
    from tlustynn import predict_atmosphere

    df = predict_atmosphere(teff=10000, logg=3.7, log_he_h=0.0)
    # Returns a pandas DataFrame with columns:
    # teff, logg, log_he_h, M, T, ne, rho, level_1 ... level_55
"""

import os
import numpy as np
import pandas as pd

from .predict import TlustyPredictor
import matplotlib
import matplotlib.pyplot as plt
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import shutil
import glob
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from astropy.io import fits


_predictor = None


def _get_predictor():
    """Get or create the default TlustyPredictor."""
    global _predictor
    if _predictor is None:
        _predictor = TlustyPredictor()
    return _predictor


def write_tlusty_model7(filename, model_data):

    df = model_data['dataframe']
    n_depth = model_data['n_depth']
    n_params = model_data['n_params']
    
    with open(filename, 'w') as f:
        # Write header: number of depths and parameters
        f.write(f"   {n_depth:3d}   {n_params:3d}\n")
        
        # Write mass values (first parameter column)
        mass_values = df['M'].values
        for i in range(0, n_depth, 6):
            line_values = mass_values[i:i+6]
            line_str = "".join(f"{val:13.6e}" for val in line_values)
            f.write(line_str + "\n")
        
        # Write all other parameters for each depth
        for depth_idx in range(n_depth):
            # Skip mass column (col 0), keep T, ne, rho, level_1 ... level_55
            row_data = df.iloc[depth_idx, 1:].values
            
            for i in range(0, n_params, 5):
                line_values = row_data[i:i+5]
                line_str = "".join(f" {val:13.6e}" for val in line_values)
                f.write(line_str + "\n")


def format_abundance(abn):
    if isinstance(abn, (int, float)):
        if abn == 0:
            return "0"
        else:
            # 使用 .6e 格式，然后手动调整指数位数
            formatted = f"{abn:.6e}"
            # 检查指数部分
            if 'e-' in formatted:
                mantissa, exponent = formatted.split('e-')
                exponent_int = int(exponent)
                # 确保指数有两位数字
                return f"{mantissa}e-{exponent_int:02d}"
            elif 'e+' in formatted:
                mantissa, exponent = formatted.split('e+')
                exponent_int = int(exponent)
                return f"{mantissa}e+{exponent_int:02d}"
            else:
                return formatted
    else:
        return str(abn)


def create_tlusty_input(output_path, teff, logg, lte_flag, ltgray_flag, nst_mode, nfread, natoms, modes, ions):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if nst_mode==None:

        with open(output_path, 'w') as f:
            f.write(f" {teff:.0f}  {logg}      ! TEFF, GRAV\n")  # 改为保留2位小数
            f.write(f" {lte_flag}  {ltgray_flag}                ! LTE,  LTGRAY\n")
            f.write(f" ''                  ! no change of general optional parameters\n")
            f.write("*\n")
            f.write("* frequencies\n")
            f.write("*\n")
            f.write(f" {nfread:3d}                  ! NFREAD\n")
            f.write("*" + "-" * 64 + "\n")
            f.write("* data for atoms   \n")
            f.write("*\n")
            f.write(f" {natoms:2d}                   ! NATOMS\n")
            f.write("* mode abn modpf\n")
    

            for mode, abn, modpf in modes:

                if isinstance(abn, (int, float)):
                    if abn == 0:
                        abn_str = "0.000000e+00"  
                    else:
                        abn_str = format_abundance(abn)
                else:
                    abn_str = str(abn)

                f.write(f"   {mode:>2d}  {abn_str:>12s}      {modpf:>2d}\n")


            f.write("*" + "-" * 64 + "\n")
            f.write("* data for ions\n")
            f.write("*\n")
            f.write("*iat   iz   nlevs  ilast ilvlin  nonstd typion  filei\n")
            f.write("*\n")
            

            for ion_data in ions:
                iat, iz, nlevs, ilast, ilvlin, nonstd, typion, filei = ion_data
                
                if iat == 0 and iz == 0 and nlevs == "" and ilast == "" and ilvlin == "" and nonstd == "" and typion == "":
                    filei_str = " " * 38 + f"'{filei}'" 
                    f.write(f"   0    0{filei_str}\n")
                else:
                    # 修复：根据值的类型使用不同的格式化方式
                    iat_str = "  " if iat == "" else f"{int(iat):3d}"
                    iz_str = "  " if iz == "" else f"{int(iz):2d}"
                    
                    # 处理可能为空的字段
                    if nlevs == "":
                        nlevs_str = "     "
                    else:
                        nlevs_str = f"{int(nlevs):5d}"
                    
                    if ilast == "":
                        ilast_str = "     "
                    else:
                        ilast_str = f"{int(ilast):6d}"
                    
                    if ilvlin == "":
                        ilvlin_str = "      "
                    else:
                        ilvlin_str = f"{int(ilvlin):6d}"
                    
                    if nonstd == "":
                        nonstd_str = "       "
                    else:
                        nonstd_str = f"{int(nonstd):6d}"
                    
                    typion_str = "       " if typion == "" else f"'{typion}'"
                    filei_str = "       " if filei == "" else f"'{filei}'"

                    if all(x == "" for x in [iat, iz, nlevs, ilast, ilvlin, nonstd, typion, filei]):
                        f.write(" \n")
                    else:
                        line = f" {iat_str}   {iz_str} {nlevs_str} {ilast_str} {ilvlin_str} {nonstd_str}    {typion_str} {filei_str}\n"
                        f.write(line)

            f.write("*\n")
            f.write("* end\n")
    else:
        with open(output_path, 'w') as f:
            f.write(f" {teff:.0f}  {logg}      ! TEFF, GRAV\n")
            f.write(f" {lte_flag}  {ltgray_flag}                ! LTE,  LTGRAY\n")
            f.write(f" '{nst_mode}'                  ! no change of general optional parameters\n")
            f.write("*" + "-" * 64 + "\n")
            f.write("* frequencies\n")
            f.write("*\n")
            f.write(f" {nfread:3d}                  ! NFREAD\n")
            f.write("*" + "-" * 64 + "\n")
            f.write("* data for atoms   \n")
            f.write("*\n")
            f.write(f" {natoms:2d}                   ! NATOMS\n")
            f.write("* mode abn modpf\n")
            
            for mode, abn, modpf in modes:

                if isinstance(abn, (int, float)):
                    if abn == 0:
                        abn_str = "0.000000e+00"  
                    else:
                        abn_str = format_abundance(abn)
                else:
                    abn_str = str(abn)

                f.write(f"   {mode:>2d}  {abn_str:>12s}      {modpf:>2d}\n")


            f.write("*" + "-" * 64 + "\n")
            f.write("* data for ions\n")
            f.write("*\n")
            f.write("*iat   iz   nlevs  ilast ilvlin  nonstd typion  filei\n")
            f.write("*\n")
            
            for ion_data in ions:
                iat, iz, nlevs, ilast, ilvlin, nonstd, typion, filei = ion_data
                
                if iat == 0 and iz == 0 and nlevs == "" and ilast == "" and ilvlin == "" and nonstd == "" and typion == "":
                    filei_str = " " * 38 + f"'{filei}'" 
                    f.write(f"   0    0{filei_str}\n")
                else:
                    iat_str = "  " if iat == "" else f"{int(iat):3d}"
                    iz_str = "  " if iz == "" else f"{int(iz):2d}"
                    
                    if nlevs == "":
                        nlevs_str = "     "
                    else:
                        nlevs_str = f"{int(nlevs):5d}"
                    
                    if ilast == "":
                        ilast_str = "     "
                    else:
                        ilast_str = f"{int(ilast):6d}"
                    
                    if ilvlin == "":
                        ilvlin_str = "      "
                    else:
                        ilvlin_str = f"{int(ilvlin):6d}"
                    
                    if nonstd == "":
                        nonstd_str = "       "
                    else:
                        nonstd_str = f"{int(nonstd):6d}"
                    
                    typion_str = "       " if typion == "" else f"'{typion}'"
                    filei_str = "       " if filei == "" else f"'{filei}'"

                    if all(x == "" for x in [iat, iz, nlevs, ilast, ilvlin, nonstd, typion, filei]):
                        f.write(" \n")
                    else:
                        line = f" {iat_str}   {iz_str} {nlevs_str} {ilast_str} {ilvlin_str} {nonstd_str}    {typion_str} {filei_str}\n"
                        f.write(line)

            f.write("*\n")
            f.write("* end\n")


def create_ff_model(output_dir, teff, logg, log_he_h, lte_flag, ltgray_flag, nstmode, frequency, natoms_num):
    os.makedirs(output_dir, exist_ok=True)
    
    abn_he = 10**log_he_h
    abn_he_str = format_abundance(abn_he)

    filename = f"{teff}_{logg}_{log_he_h}.5"
    output_path = os.path.join(output_dir, filename)
        
    modes = []
    elements = [
        (2, 0, 0),  # H
        (2, abn_he_str, 0),  # He abundance = 10**log_he_h
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, 0),
        (1, 0, 0)
        ]
    
    modes.extend(elements)
    
    ions = [
        ( 1,   0,   9,   0,   0,   0, ' H 1', 'data/h1.dat'),
        ( 1,   1,   1,   1,   0,   0, ' H 2', ' '),
        ( 2,   0,  24,   0,   0,   0, 'He 1', 'data/he1.dat'),
        ( 2,   1,  20,   0,   0,   0, 'He 2', 'data/he2.dat'),
        ( 2,   2,   1,   1,   0,   0, 'He 3', ' '),
        ( 0,   0,   0,  -1,   0,   0, '    ', ' ')
        ]
    
    create_tlusty_input(output_path, teff, logg, lte_flag, ltgray_flag, nstmode, frequency, natoms_num, modes, ions)


def predict_atmosphere(teff, logg, log_he_h, output_dir=None, filename=None, output_format='csv'):

    predictor = _get_predictor()
    result = predictor.predict(teff, logg, log_he_h)
    y_pred = result['prediction'][0]  # [50, n_outputs]

    if predictor.stats and 'output_cols' in predictor.stats:
        output_cols = predictor.stats['output_cols']
    else:
        output_cols = [f'col_{i}' for i in range(y_pred.shape[1])]

    df = pd.DataFrame(y_pred, columns=output_cols)

    if predictor.avg_mass_physical is not None:
        if predictor.stats and 'M' in predictor.stats.get('log_transform_cols', []):
            mass_vals = 10 ** predictor.avg_mass_physical
        else:
            mass_vals = predictor.avg_mass_physical
    else:
        mass_vals = np.zeros(50, dtype=np.float32)
    df.insert(0, 'M', mass_vals)

    filepath = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        ext = output_format.lower()
        if filename is None:
            filename = f"{teff}_{logg}_{log_he_h}.{ext}"
        elif not filename.endswith(f'.{ext}'):
            filename = f"{filename}.{ext}"
        
        filepath = os.path.join(os.path.abspath(output_dir), filename)
        
        # Write in requested format
        if output_format.lower() == 'csv':

            # Insert stellar parameters (replicated for every depth row)
            df.insert(0, 'log_he_h', float(log_he_h))
            df.insert(0, 'logg', float(logg))
            df.insert(0, 'teff', float(teff))


            df.to_csv(filepath, index=False)
        elif output_format.lower() == '7':
            # Prepare model data dictionary for .7 format
            # n_params is number of columns excluding teff, logg, log_he_h, mass
            model_data = {
                'dataframe': df,
                'n_depth': len(df),
                'n_params': 58
            }
            write_tlusty_model7(filepath, model_data)
        else:
            raise ValueError(f"Unsupported output_format: {output_format}. Use 'csv' or '7'")

    return df, filepath


class TlustyAtmosphere:
    """Convenient wrapper around :class:`TlustyPredictor`.

    Example
    -------
    >>> atm = TlustyAtmosphere()
    >>> df = atm.predict(10000, 3.7, 0.0)
    >>> df.to_csv('my_model.csv', index=False)
    """

    def __init__(self, checkpoint_path=None, device=None):
        self.predictor = TlustyPredictor(
            checkpoint_path=checkpoint_path, device=device
        )

    def predict(self, teff, logg, log_he_h, output_dir=None, filename=None, 
                output_format='csv'):
        """Same interface as :func:`predict_atmosphere`."""
        result = self.predictor.predict(teff, logg, log_he_h)
        y_pred = result['prediction'][0]

        if self.predictor.stats and 'output_cols' in self.predictor.stats:
            output_cols = self.predictor.stats['output_cols']
        else:
            output_cols = [f'col_{i}' for i in range(y_pred.shape[1])]

        df = pd.DataFrame(y_pred, columns=output_cols)

        if self.predictor.avg_mass_physical is not None:
            if self.predictor.stats and 'M' in self.predictor.stats.get('log_transform_cols', []):
                mass_vals = 10 ** self.predictor.avg_mass_physical
            else:
                mass_vals = self.predictor.avg_mass_physical
        else:
            mass_vals = np.zeros(50, dtype=np.float32)
        df.insert(0, 'M', mass_vals)


        filepath = None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine filename extension
            ext = output_format.lower()
            if filename is None:
                filename = f"{teff}_{logg}_{log_he_h}.{ext}"


            elif not filename.endswith(f'.{ext}'):
                filename = f"{filename}.{ext}"
            
            filepath = os.path.join(os.path.abspath(output_dir), filename)
            
            # Write in requested format
            if output_format.lower() == 'csv':

                df.insert(0, 'log_he_h', float(log_he_h))
                df.insert(0, 'logg', float(logg))
                df.insert(0, 'teff', float(teff))


                df.to_csv(filepath, index=False)


            elif output_format.lower() == '7':
                model_data = {
                    'dataframe': df,
                    'n_depth': len(df),
                    'n_params': 58
                }
                write_tlusty_model7(filepath, model_data)
            else:
                raise ValueError(f"Unsupported output_format: {output_format}. Use 'csv' or '7'")

        return df, filepath
    

def create_fort55_lin(output_dir, filename,
                     imode, idstd, iprin,
                     inmod, intrpl, ichang, ichemc,
                     iophli, nunalp, nunbet, nungam, nunbal,
                     ifreq, inlte, icontl, inlist, ifhe2,
                     ihydpr, ihe1pr, ihe2pr,
                     alam0, alast, cutof0, cutofs, relop, space,
                     nmlist,iunitm):

    output_path = os.path.join(output_dir, filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # 第一行: imode idstd iprin
        f.write(f"{imode:8d} {idstd:7d} {iprin:7d}                                ! tmode,idstd,tprin\n")
        
        # 第二行: inmod intrpl ichang ichemc
        f.write(f"{inmod:8d} {intrpl:7d} {ichang:7d} {ichemc:7d}                        ! tnmod,intrpl,ichang,ichemc\n")
        
        # 第三行: iophli nunalp nunbet nungam nunbal
        f.write(f"{iophli:8d} {nunalp:7d} {nunbet:7d} {nungam:7d} {nunbal:7d}                ! tophlt,nunalp,nunbet,nungan,nunbal\n")
        
        # 第四行: ifreq inlte icontl inlist ifhe2
        f.write(f"{ifreq:8d} {inlte:7d} {icontl:7d} {inlist:7d} {ifhe2:7d}                ! tfreq,inite,icontl,inlist,tfhe2\n")
          
        # 第五行: ihydpr ihe1pr ihe2pr
        f.write(f"{ihydpr:8d} {ihe1pr:7d} {ihe2pr:7d}                                ! thydpr,the1pr,the2pr\n")
        
        # 第六行: alam0 alast cutof0 cutofs relop space
        f.write(f"    {alam0:.0f}    {alast:.0f}       {cutof0:.0f}       {cutofs:.0f}  {relop}    {space}\n")    
            
        # 第七行: nmlist
        f.write(f"{nmlist:8d}{iunitm:8d}                                        ! nnlist\n")


def run_synspec(synspec_dir, model_name,linelist_file):

    command = ["$TLUSTY/RSynspec", model_name, "fort.55.lin", linelist_file]
    
    try:
        result = subprocess.run(
            ' '.join(command), 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True,
            cwd=synspec_dir
        )
        return True
    except subprocess.CalledProcessError as e:
        print("Synspec failed!")
        print("Error:", e.stderr)
        return False


def synthesize_spectrum(teff, logg, log_he_h, spec_dir, down, up, res, format, plot=False):

# synthesize_spectrum( 32200, 5.7, -1.635, '/home/ubuntu/phd/lyp/tlusty', 3000, 9000, 0.001, 'csv', plot=True)

    create_ff_model(spec_dir, teff, logg, log_he_h, lte_flag='F', ltgray_flag='F', nstmode='nst', frequency=2000, natoms_num=8)
    
    predict_atmosphere(teff, logg, log_he_h, output_dir=spec_dir, output_format='7')

    create_fort55_lin(spec_dir, "fort.55.lin",
        imode=0, idstd=50, iprin=1,           
        inmod=1, intrpl=0, ichang=0, ichemc=0,
        iophli=0, nunalp=0, nunbet=0, nungam=0, nunbal=0,          
        ifreq=1, inlte=1, icontl=0, inlist=0, ifhe2=0,         
        ihydpr=0, ihe1pr=0, ihe2pr=0,         
        alam0=down, alast=up, cutof0=10, cutofs=0.0, relop=res, space=0.5,
        nmlist=0, iunitm=0)
    
    run_synspec(spec_dir, f"{teff}_{logg}_{log_he_h}", "data/gfATO.dat")
    
    spec_file = os.path.join(spec_dir, ".spec")
    if not os.path.exists(spec_file):
        spec_file = os.path.join(spec_dir, f"{teff}_{logg}_{log_he_h}.spec")
    
    if os.path.exists(spec_file):
        spec = pd.read_csv(spec_file, sep='\s+', header=None, names=['waveobs', 'flux'])

        if plot:

            output_file=os.path.join(os.path.abspath(spec_dir), f'spec.pdf')

            fig, ax =plt.subplots(figsize=(10, 5), dpi=300)
            
            ax.plot(spec['waveobs'], spec['flux'], color='r', linestyle='-', linewidth=0.6)
            ax.set_xlabel('Wavelength (Å)', fontsize=12)
            ax.set_ylabel('Flux', fontsize=12)
            ax.legend(loc='upper right', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_xlim(3600, 7500)
            ax.set_ylim(None, None)

            fig.savefig(output_file, dpi=600, bbox_inches='tight')
            plt.close()
        else:
            pass

    else:
        raise FileNotFoundError(f"Spectrum output file not found in {spec_dir}")
    
    if format == "csv":
        filename = f"{teff}_{logg}_{log_he_h}.csv"
        filepath = os.path.join(os.path.abspath(spec_dir), filename)
        spec.to_csv(filepath, index=False)

    else:  
        filename = f"{teff}_{logg}_{log_he_h}.fits"
        filepath = os.path.join(os.path.abspath(spec_dir), filename)
        
        col1 = fits.Column(name='waveobs', format='E', array=spec['waveobs'].values)
        col2 = fits.Column(name='flux', format='E', array=spec['flux'].values)
        
        hdu = fits.BinTableHDU.from_columns([col1, col2])
        
        hdu.header['TEFF'] = (teff, 'Effective Temperature (K)')
        hdu.header['LOGG'] = (logg, 'Surface Gravity (log10 cm/s^2)')
        hdu.header['LOGHEH'] = (log_he_h, 'log(n_He/n_H) (dex)')
        hdu.header['WAVEMIN'] = (down, 'Minimum wavelength (Angstrom)')
        hdu.header['WAVEMAX'] = (up, 'Maximum wavelength (Angstrom)')
        
        hdu.writeto(filepath, overwrite=True)

