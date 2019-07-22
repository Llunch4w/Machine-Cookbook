import warnings
warnings.filterwarnings("ignore")

def txt_read(filename,delim):
    x_data = []
    y_data = []
    with open(filename,'r') as f:
        for line in f:
            data = [float(x) for x in line.split(delim)]
            x_data.append(data[:-1])
            y_data.append(data[-1])

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data,y_data 


def txt_read2(filename,delim):
	x_data = []
    with open(filename,'r') as f:
        for line in f:
            data = line.strip().split(delim)
        #     data = [float(x) for x in line.strip().split(delim)]
            x_data.append(data)

    x_data = np.array(x_data)
    return x_data


def build_arg_parser():
	import argparse
	parser = argpase.ArgumentParser(description='Compress the image')
	parser.add_argument('--input-file',dest='input_file',required=True,help='Input image')
	parser.add_argument('--num-bits',dest='num_bits',required=False,type=int,
	help='Number of bits used to represent each pixel')
	return parser
	

