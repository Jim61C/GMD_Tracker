import numpy as np 
import cPickle
import matplotlib.pyplot as plt
import sys

def main():
	loss_history_file = sys.argv[1]
	loss_history = np.genfromtxt(loss_history_file, delimiter = ' ')
	avg_period = 100

	print "minimum loss:", np.min(loss_history)
	print "maximum loss:", np.max(loss_history)
	loss_history_to_plot = np.zeros(len(loss_history)/avg_period)
	for i in range(0, len(loss_history_to_plot)):
		loss_history_to_plot[i] = np.average(loss_history[i*avg_period:(i+1)*avg_period])
	plt.plot(loss_history_to_plot)
	plt.ylabel('loss')
	plt.xlabel('iterations')
	plt.title(loss_history_file[:loss_history_file.find('.')])
	plt.show()

	return

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Usage: python {0} loss_history_file".format(sys.argv[0])
		exit()
	main()