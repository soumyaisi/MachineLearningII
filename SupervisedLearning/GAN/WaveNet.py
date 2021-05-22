pip install virtualenv
mkdir ~/virtualenvs && cd ~/virtualenvs
virtualenv wavenet
source wavenet/bin/activate
cd ~
git clone https://github.com/basveeling/wavenet.git
cd wavenet
pip install -r requirements.txt

https://github.com/basveeling/wavenet


python wavenet.py with 'data_dir=your_data_dir_name'

python wavenet.py predict with 'models/[run_folder]/config.json predict_seconds=1'


def residual_block(x):
	original_x = x
	xtanh_out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i,
	border_mode='valid', causal=True, bias=use_bias,
	name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh',
	W_regularizer=l2(res_l2))(x)
	sigm_out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i,
	border_mode='valid', causal=True, bias=use_bias,
	name='dilated_conv_%d_sigm_s%d' % (2 ** i, s), activation='sigmoid',
	W_regularizer=l2(res_l2))(x)
	x = layers.Merge(mode='mul',
	name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])
	res_x = layers.Convolution1D(nb_filters, 1, border_mode='same', bias=use_bias,
	W_regularizer=l2(res_l2))(x)
	skip_x = layers.Convolution1D(nb_filters, 1, border_mode='same', bias=use_bias,
	W_regularizer=l2(res_l2))(x)
	res_x = layers.Merge(mode='sum')([original_x, res_x])
	return res_x, skip_x


https://deepmind.com/blog/wavenet-generative-model-raw-audio/


#https://deepmind.com/blog/wavenet-generative-model-raw-audio/