import numpy as np
import torch
import os 
from colour import Color
import pickle 
import glob
import math


X_SHAPE = (128,128)

############## Smoothness analysis functions ###########

def total_change(list_of_changes,type='l1'):

    '''
    takes as input a list of changes within trajectory, outputs 
    a list of metrics indicating total amount of change within trajectory
    and mean changes within trajectory
    '''

    tc = []
    tcm = []

    if type == 'l1':
        for traj in list_of_changes:
            changes = np.sum(np.abs(traj),axis=1)
            tc.append(changes)
            tcm.append(np.mean(changes))

    elif type == 'l2':
        for traj in list_of_changes:
            changes = np.sum(traj ** 2,axis=1)
            tc.append(changes)
            tcm.append(np.mean(changes))

    elif type == 'linf':
        for traj in list_of_changes:
            changes = np.argmax(traj,axis=1)
            tc.append(changes)
            tcm.append(np.mean(changes))
    else: 
        print('Not implemented!')
        raise NotImplementedError


    return tc,tcm


def cosine_angle_change(list_of_trajs):

    '''
    takes as input a list of latent vectors within a trajectory, outputs a list of cosine similarities,angles between vectors
    in addition to averages of the two
    '''

    cos_sims = []
    angles = []
    mean_cos_sims = []
    mean_angles = []

    for traj in list_of_trajs:
        tmp_cos = []
        tmp_ang = []

        for vec_ind in range(traj.shape[0] - 1):
            v1 = traj[vec_ind,:]
            v2 = traj[vec_ind +1,:]

            num = v1 @ v2 
            denom = np.sqrt(v1 @ v1) * np.sqrt(v2 @ v2)

            cos = num/denom
            tmp_cos.append(cos)

            ang = math.degrees(math.acos(cos))
            tmp_ang.append(ang)

        cos_sims.append(tmp_cos)
        mean_cos_sims.append(np.mean(tmp_cos))

        angles.append(tmp_ang)
        mean_angles.append(np.mean(tmp_ang))

    return (angles, cos_sims), (mean_angles,mean_cos_sims)




#####################  MMD Functions ################
def rbf_dot(patterns1,patterns2,sig):

    size1 = patterns1.shape
    size2 = patterns2.shape

    G = np.pow(patterns1,2).sum(axis=1)
    H = np.pow(patterns2,2).sum(axis=1)

    Q = np.tile(G,[1,size2[0]])
    R = np.tile(H.T,[size1[0],1])

    H = Q + R - 2*patterns1 @ patterns2.T

    H=np.exp(-H/2/sig**2);

    return H 

def mmd_fxn(lat1,lat2,sig):

    if sig == -1:
        Z = np.hstack([lat1,lat2])

        size1 = Z.shape[0]
        if size1 > 100:
            Zmed = Z[0:100,:]
            size1 = 100
        else:
            Zmed = Z 

        G = np.sum(Zmed **2,axis=1)
        Q = np.tile(G,[1,size1])
        R = np.tile(G.T,[size1,1])

        dists = Q + R - 2* Zmed @ Zmed.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists,[size1**2,1])
        sig = np.sqrt(0.5 * np.median(dists[dists > 0]))

    mmds = []
    for l in lat1:
        l = np.expand_dims(l,0)
        mmd = np.sum(rbf_dot(l,lat1) - rbf_dot(l,lat2))
        mmds.append(mmd)

    return mmds

############# Dataframe functions ###############
def numpy_to_tensor(x):
	"""Transform a numpy array into a torch.FloatTensor."""
	return torch.from_numpy(x).type(torch.FloatTensor)


############### Old dataframe & latent dynamics fxns - necessary?? #########

def stitch_images(list_of_images,thresh=0.05):

	# initialize with first image
	i1 = list_of_images[0]

	# iterate through images to stitch
	orig_total_len = i1.shape[1]
	for ind in range(len(list_of_images) - 1):
		i2 = list_of_images[ind + 1]

		orig_total_len += i2.shape[1]

		ldif = i1.shape[1] - i2.shape[1]

		corr_vec = np.zeros((i2.shape[1],))
		for offset in range(i2.shape[1]):

			tmpvec_i1 = i1[:,ldif + offset::].flatten()
			if offset == 0:
				tmpvec_i2 = i2.flatten()
			else:
				tmpvec_i2 = i2[:,0:-offset].flatten()
			# calculate correlation of overlap
			a = (tmpvec_i1 - np.mean(tmpvec_i1))/(np.std(tmpvec_i1)* len(tmpvec_i1))
			b = (tmpvec_i2 - np.mean(tmpvec_i2))/np.std(tmpvec_i2)
			corr_vec[offset] = np.correlate(a,b)

		max_corr_ind = np.argmax(corr_vec)
		if corr_vec[max_corr_ind] > thresh:
			#print('Overlap detected! Stitching from offset {0}'.format(max_corr_ind))
			#best_offset = max_corr_ind
			if max_corr_ind == 0:
			#	print('stitchin')
			#	print('old shape: {0}'.format(i1.shape))

				i1[:,ldif::] = (i1[:,ldif::] + i2)/2
			#	print('new shape: {0}'.format(i1.shape))
			else:
				i1[:,ldif+max_corr_ind::] = (i1[:,ldif+max_corr_ind::] + i2[:,0:-max_corr_ind])/2

				i1 = np.hstack((i1, i2[:,max_corr_ind::]))

		else:
			i1 = np.hstack((i1,i2))
			#print('No overlap! Concatenating')
	new_total_len = i1.shape[1]

	print('original total length: {0} \n new total length: {1}'.format(orig_total_len,new_total_len))
	return i1


#@numba.njit(fastmath=True)
#@numba.njit(fastmath = False)
def stitch_and_decode(model,trajectories, day,path='.'):

	#for trajectory in trajectories:

	#audio = []
#	stitcher = cv2.Stitcher.create(mode = 1)
	max_decode = 50
	if not os.path.isdir(path):
		os.mkdir(path)

	specs = []
	spec_ind = 0
	print('decoding trajectories')
	print(trajectories.shape)
	for trajectory in trajectories:

		trajectory = torch.tensor(trajectory).type(torch.FloatTensor).to(model.device)
		if trajectory.shape[1] != 32:
			trajectory = trajectory.T[:,:32]
		#print(trajectory.shape)
		tmpspecs = []
		with torch.no_grad():
			#print(trajectory.shape)
			traj_rec = model.decode(trajectory).view(-1, X_SHAPE[0], X_SHAPE[1]).detach().cpu().numpy()
			#print(np.range(traj_rec))
			#print(np.min(traj_rec))
			#####print(np.max(traj_rec))
			#traj_rec -= 2
			#traj_rec /= 5
			#traj_rec = np.clip(traj_rec, 0.0, 1.0)
			for t in traj_rec:
				#gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
				#gray_three = cv2.merge([t,t,t])
				tmpspecs.append(t)

			result = stitch_images(tmpspecs)
			#result = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
			#if status != cv2.Stitcher_OK:
			#	print('uh oh')
			specs.append(result)
			fig, ax = plt.subplots()
			ax.imshow(np.flipud(result))
			ax.set_aspect('auto')
			plt.savefig(os.path.join(path,'simulated_spec_'+ str(day) + 'dph_' + str(spec_ind) + '.png'))
			plt.close(fig)
			spec_ind += 1

		if spec_ind >= max_decode:
			break

	#print(np.hstack(tmpspecs).shape)
	'''
	print('writing audio')
	spec_ind = 0
	for spec in specs:
		tmp_aud = librosa.core.spectrum.griffinlim(spec)
		scipy.io.wavfile.write(os.path.join(path,'simulated_audio_' + str(spec_ind) + '.wav'),\
					FS,np.array(tmp_aud,np.int16))
		spec_ind += 1
		#audio.append(tmp_aud)
		### reshape to be many T by 128
		### turn to .wav
		### return spectrogram
	'''
	return specs

def latents_to_pickle(latents, list_of_latent_days,path,n_per_file=30,just_read=False,max_files=300):

	if not os.path.isdir(path):
		os.mkdir(path)

	if just_read:

		fnames = glob.glob(os.path.join(path,'*.pkl'))
		voc_data = {
			'latent_fnames':fnames,
			'n_files':len(fnames),
			}
	else:
		write_file_num = 0
		voc_data = {
			'latent_fnames':[],
			'n_files':0,
			}
		n_in_file = 1
		save_filename = \
			"latent_vocs_" + str(write_file_num).zfill(4) + '.pkl'
		save_filename = os.path.join(path, save_filename)

		data_to_pickle = {}

		latent_in_file = []
		length_voc = []
		latent_day = []
		voc_data['latent_fnames'].append(save_filename)
		voc_data['n_files'] += 1
		for ind,latent in enumerate(latents):


			if n_in_file <= n_per_file:
				latent_in_file.append(latent)
				length_voc.append(len(latent))
				latent_day.append(list_of_latent_days[ind])
				n_in_file += 1
			else:

				write_file_num += 1
				#print(len(latent_in_file))
				#print(latent_in_file)
				data_to_pickle['latent_in_file'] = latent_in_file
				data_to_pickle['length_voc'] = length_voc
				data_to_pickle['latent_day'] = latent_day
				data_to_pickle['length_file'] = n_per_file
				with open(save_filename, "wb") as f:
					pickle.dump(data_to_pickle,f)

				latent_in_file = []
				length_voc = []
				data_to_pickle = {}
				voc_data['latent_fnames'].append(save_filename)
				voc_data['n_files'] += 1
				save_filename = \
					"latent_vocs_" + str(write_file_num).zfill(4) + '.pkl'
				save_filename = os.path.join(path, save_filename)

				n_in_file = 1

			if write_file_num >= max_files:
				return voc_data

		if len(latent_in_file) > 0:
			data_to_pickle['latent_in_file'] = latent_in_file
			data_to_pickle['length_voc'] = length_voc
			data_to_pickle['latent_day'] = latent_day
			data_to_pickle['length_file'] = len(latent_in_file)
			with open(save_filename, "wb") as f:
				pickle.dump(data_to_pickle,f)

			voc_data['latent_fnames'].append(save_filename)
			voc_data['n_files'] += 1

	return voc_data

############ Plotting Functions ###########

def set_colors(vec,min,max):

	mapped_vals = []
	#min mako: (0.1819,0.119,0.231) #2e1e3b
	#max mako: (0.546,0.854,0.698) #8bdab3

	cmin = Color('#2e1e3b')
	cmax = Color('#8bdab3')

	val_range = np.linspace(-1,1,500)
	crange = list(cmin.range_to(cmax,500))
	for data in vec:
		val = 2 * (data - min)/max - 1 # range between zero and 1
		min_ind = np.argmin(np.abs(val_range - val))

		mapped_vals.append(val_range[min_ind])


	return mapped_vals





##############################################
if __name__ == '__main__':

    pass