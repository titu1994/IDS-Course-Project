from census import Census
from us import states

if __name__ == "__main__":
    c = Census("9f987445528752bd5d969ce0f061d34cf45417db")
    print(c.acs5.get(('B01001_004E'), {'for': 'state:{}'.format(states.MD.fips)}))