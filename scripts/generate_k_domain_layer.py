import sys
import os

def main():
	out_file = sys.argv[1]

	K = 89
	layer_name_prefix = "fc8_k"
	blob_name_prefix = "fc8_k"
	loss_layer_prefix = "loss_k"

	bottom_label_layer = "label_flat"

	lr_mult_weights = 10
	decay_mult_weights = 1

	lr_mult_bias = 20
	decay_mult_bias = 0

	f = open(out_file, 'wb')

	f.write("# k domains\n")
	for i in range(0, K):
		f.write(
		"""layer {{
  name: \"{fc_layer_name}\"  
  type: "InnerProduct"  
  bottom: "fc7b"  
  top: "{fc_blob_name}"  
  param {{
    lr_mult: {lr_mult_weights} 
    decay_mult: {decay_mult_weights} 
  }}
  param {{
    lr_mult: {lr_mult_bias} 
    decay_mult: {decay_mult_bias} 
  }}
  inner_product_param {{
    num_output: 250
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
  name: "{loss_layer_name}"
  type: "SigmoidCrossEntropyLoss"
  bottom: "{fc_blob_name}"
  bottom: "{bottom_label_layer}"
  top: "{loss_layer_name}"
  include {{ phase: TRAIN }}
}}\n""".format(fc_layer_name = layer_name_prefix+str(i), \
			fc_blob_name = blob_name_prefix+str(i), \
			lr_mult_weights = lr_mult_weights, \
			decay_mult_weights = decay_mult_weights, \
			lr_mult_bias = lr_mult_bias , \
			decay_mult_bias = decay_mult_bias, \
			loss_layer_name = loss_layer_prefix + str(i), \
			bottom_label_layer = bottom_label_layer))

	f.close()



if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Usage: python {0} out_file".format(sys.argv[0])
		exit(0)
	main()