from astropy.time import Time

# Offset: différence entre epoch Unix (1970-01-01) et epoch Fermi (2001-01-01)
FERMI_EPOCH = Time('2001-01-01T00:00:00', format='isot', scale='utc')
FERMI_OFFSET = FERMI_EPOCH.unix  # = 978307200.0 secondes

# Conversion MET → JD
met_seconds = 501643824
t = Time(met_seconds + FERMI_OFFSET, format='unix', scale='utc')

print(f"MET Fermi : {met_seconds:.4e} s")
print(f"JD        : {t.jd:.8f}")
print(f"MJD       : {t.mjd:.8f}")
print(f"ISO       : {t.iso}")
