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



# synthesize_spectrum 无返回值, 光谱写入工作目录:
#   <teff>_<logg>_<log_he_h>.spec(原始输出)、.csv 或 .fits(format 控制)、spec.pdf(plot=True 时)
# synthesize_spectrum(
#     teff=35000,
#     logg=4.5,
#     log_he_h=0.0,
#     spec_dir='/home/ubuntu/phd/test/wd',          # 工作目录，所有中间文件和结果都会放在这里
#     down=3000, up=9000,  # 波长范围 (Å)
#     res=0.1,             # 波长步长 (Å)
#     format='csv',        # 'csv' 或 'fits'
#     plot=True
# )




# # synthesize_spectrum(teff, logg, log_he_h, spec_dir,  down, up, res,  format, plot=False)
synthesize_spectrum(30500,5.6,0.1,'/home/ubuntu/phd/test/wd',3000, 9000, 0.1, 'csv', plot=True)



