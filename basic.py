from tlustynn import predict_atmosphere, create_ff_model
from tlustynn import synthesize_spectrum

work_dir = '/home/ubuntu/phd/test/wd'

# Create .5 input file - will be saved as 10000_3.7_0.0.5
# create_ff_model(
#     output_dir=work_dir,
#     teff=10000, 
#     logg=3.7, 
#     log_he_h=0.0,
#     lte_flag='F', 
#     ltgray_flag='F',
#     nstmode='nst',
#     frequency=2000, 
#     natoms_num=8
# )

# # Predict atmosphere and save as CSV
# df, path = predict_atmosphere(
#     10000, 3.7, 0.1, 
#     output_dir=work_dir,
#     output_format='csv'
# )

# # Predict atmosphere and save as TLUSTY .7 format
# df, path = predict_atmosphere(
#     10000, 3.7, 0.1, 
#     output_dir=work_dir, 
#     output_format='7'
# )



# synthesize_spectrum returns no value; spectra are written to the working directory:
#   <teff>_<logg>_<log_he_h>.spec (raw output), .csv or .fits (controlled by format), spec.pdf (when plot=True)
# synthesize_spectrum(
#     teff=35000,
#     logg=4.5,
#     log_he_h=0.0,
#     spec_dir='/home/ubuntu/phd/test/wd',          # Working directory for intermediate files and results
#     down=3000, up=9000,  # Wavelength range (Å)
#     res=0.1,             # Wavelength step (Å)
#     format='csv',        # 'csv' or 'fits'
#     plot=True
# )




# # synthesize_spectrum(teff, logg, log_he_h, spec_dir,  down, up, res,  format, plot=False)
synthesize_spectrum(30500,5.6,0.1,'/home/ubuntu/phd/test/wd',3000, 9000, 0.1, 'csv', plot=True)



