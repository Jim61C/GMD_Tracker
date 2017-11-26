import sys
import os

def main():
	out_file = sys.argv[1]

	K = 89
	layer_name_prefix = "fc6_k"
	blob_name_prefix = "fc6_k"
	loss_layer_prefix = "loss_k"
	flatten_fc_layer_prefix = "flatten_fc6_k"

	bottom_label_layer = "label_flat"

	lr_mult_weights = 10
	decay_mult_weights = 0

	lr_mult_bias = 20
	decay_mult_bias = 0

	f = open(out_file, 'wb')

	f.write("# k domains\n")
	for i in range(0, K):
		f.write(
		"""# domain {k}
layer {{
  name: \"{fc_layer_name}\" 
  type: "Convolution" 
  bottom: "fc5" 
  top: "{fc_blob_name}" 
  param {{
  lr_mult: {lr_mult_weights} 
  decay_mult: {decay_mult_weights} 
  }}
  param {{
  lr_mult: {lr_mult_bias} 
  decay_mult: {decay_mult_bias} 
  }}
  convolution_param {{
  num_output: 2
  kernel_size: 1
  weight_filler {{
  type: "gaussian"
  std: 0.01
  }}
  bias_filler {{
  type: "constant"
  value: 0
  }}
  }}
}}
layer {{
  name: "{flatten_fc_layer_name}"
  type: "Flatten"
  bottom: "{fc_blob_name}"
  top: "{flatten_fc_layer_name}"
}}
layer {{
  name: "{loss_layer_name}"
  type: "SoftmaxWithLoss"
  bottom: "{flatten_fc_layer_name}"
  bottom: "{bottom_label_layer}"
  top: "{loss_layer_name}"
}}\n""".format(k = i, \
			fc_layer_name = layer_name_prefix+str(i), \
			fc_blob_name = blob_name_prefix+str(i), \
			lr_mult_weights = lr_mult_weights, \
			decay_mult_weights = decay_mult_weights, \
			lr_mult_bias = lr_mult_bias , \
			decay_mult_bias = decay_mult_bias, \
			loss_layer_name = loss_layer_prefix + str(i), \
			bottom_label_layer = bottom_label_layer, \
  flatten_fc_layer_name = flatten_fc_layer_prefix + str(i)))

	f.close()



if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Usage: python {0} out_file".format(sys.argv[0])
		exit(0)
	main()