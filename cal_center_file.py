from center import read_location_pos, cal_center_split

filename = 'trainid-id-dataset_TSMC2014_NYC.txt'

location_pos = read_location_pos(filename, split_sig='\t', iin=1, lain=4, loin=5)

with open('loc-center-' + filename, 'w') as f:
    for loc in location_pos.keys():
        la, lo = cal_center_split(location_pos[loc])
        f.write(','.join([str(loc),str(la),str(lo)])+'\n')