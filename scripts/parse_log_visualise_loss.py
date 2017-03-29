import sys
import numpy as np 
import matplotlib.pyplot as plt
import re

def main():
	log_file = sys.argv[1]
	k = int(sys.argv[2])
	TOTAL_NUM_DOMAIN = int(sys.argv[3])

	loss_log_regex = r"Train net output #(.*): loss_k" + re.escape(str(k)) + r" = (.*) " + re.escape("(")
	f = open(log_file, 'rb')
	lines = f.readlines()

	# loss_log_lines = []
	loss_nums = []
	for line in lines:
		search_obj = re.search(loss_log_regex, line, re.M)
		if search_obj:
			# print search_obj.group(2)
			# loss_log_lines.append(search_obj.group())
			loss_nums.append(float(search_obj.group(2)))

	# print "len(loss_log_lines):", len(loss_log_lines)

	

	i = 0
	loss_history = []
	while(i*TOTAL_NUM_DOMAIN + k < len(loss_nums)):
	# while(i*TOTAL_NUM_DOMAIN + k < len(loss_log_lines)):
		# this_loss_line = loss_log_lines[i*TOTAL_NUM_DOMAIN + k]
		# this_loss_num = float(this_loss_line[this_loss_line.find(loss_log_regex)+len(loss_log_regex):this_loss_line.find('(')])
		# print "this_loss_num:", this_loss_num
		# loss_nums.append(this_loss_num)
		loss_history.append(loss_nums[i*TOTAL_NUM_DOMAIN + k])
		i += 1

	loss_history = np.asarray(loss_history)
	avg_period = 1

	print "minimum loss:", np.min(loss_history)
	print "maximum loss:", np.max(loss_history)
	loss_history_to_plot = np.zeros(len(loss_history)/avg_period)
	for i in range(0, len(loss_history_to_plot)):
		loss_history_to_plot[i] = np.average(loss_history[i*avg_period:(i+1)*avg_period])
	plt.plot(loss_history_to_plot)
	plt.title(log_file + "_domain_" +str(k))
	plt.ylabel('loss_domain_' + str(k))
	plt.xlabel('# cycles')
	plt.show()

	f.close()
	return


if __name__ == "__main__":
	if len(sys.argv) != 4:
		print "Usage: python {0} log_file k TOTAL_NUM_DOMAIN".format(sys.argv[0])
		exit()
	main()