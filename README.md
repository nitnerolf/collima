# collima
A collection of image reconstruction recipes

- LFF: Low Frequency Filling ([Millour et al. 2012](https://ui.adsabs.harvard.edu/abs/2012SPIE.8445E..1BM/abstract))
Low Frequency filling is based on the principles exposed in [Lachaume et al. 2003](https://ui.adsabs.harvard.edu/abs/2003A%26A...400..795L/abstract): at low frequency, the visibility function does not depend anymore on the shape of the object, and can be defined only by the source size. Based on a fit on the low frequencies existing in a dataset, LFF extrapolates the visibility function down to zero frequency, filling the central hole in the (u,v) coverage that is usually very annoying for image reconstruction softwares.

- Self-Cal: Self-calibration applied to optical interferometry ([Millour et al. 2011](https://ui.adsabs.harvard.edu/abs/2011A%26A...526A.107M/abstract), [2012](https://ui.adsabs.harvard.edu/abs/2012SPIE.8445E..1BM/abstract), [2015](https://ui.adsabs.harvard.edu/abs/2015IAUGA..2257065M/abstract), [Mourard et al. 2015](https://ui.adsabs.harvard.edu/abs/2015A%26A...577A..51M/abstract))

