import numpy as np 
import cPickle
import matplotlib.pyplot as plt
import sys

def main():
	loss_history_file = sys.argv[1]
	K = int(sys.argv[2])
	k = int(sys.argv[3])
	loss_history_all = np.genfromtxt(loss_history_file, delimiter = ' ')
	loss_history = loss_history_all[:, k]
	avg_period = 50

	print "minimum loss:", np.min(loss_history)
	print "maximum loss:", np.max(loss_history)
	loss_history_to_plot = np.zeros(len(loss_history)/avg_period)
	for i in range(0, len(loss_history_to_plot)):
		loss_history_to_plot[i] = np.average(loss_history[i*avg_period:(i+1)*avg_period])
	plt.plot(loss_history_to_plot)
	plt.ylabel('loss')
	plt.xlabel('iterations')
	plt.title(loss_history_file[:loss_history_file.find('.')] + '_domain_{}'.format(k))
	if (len(sys.argv) > 4):
		# save to file
		out_file = sys.argv[4]
		plt.savefig(out_file)	
	else:
		plt.show()

	return

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print "Usage: python {0} loss_history_file K k [out_file]".format(sys.argv[0])
		print "K - total number of domains"
		print "k - the specific domain to plot loss for"
		print "out_file - optional, the file to save the plot to"
		exit()
	main()