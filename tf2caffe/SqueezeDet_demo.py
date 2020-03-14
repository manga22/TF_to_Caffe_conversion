"""To run : python3 SqueezeDet_demo.py -m SqueezeDet.prototxt -w SqueezeDet.caffemodel -i 000021.png"""

import caffe
caffe.set_mode_cpu()
import sys, os, getopt
import time
import numpy as np
from kitti_squeezeDet_config import kitti_squeezeDet_config
from utils import util
import cv2
import matplotlib.pyplot as plt


def interpret_output(net_out,mc):

	out=np.transpose(net_out,[0,2,3,1])

	#calculate probabilities
	num_class_probs = mc.ANCHOR_PER_GRID*mc.CLASSES
	prob_in=np.reshape(out[:,:,:,:num_class_probs],[-1,mc.CLASSES])
	exp_prob_in = np.exp(prob_in)
	probs_out = exp_prob_in / np.sum(exp_prob_in, axis=1, keepdims=True)
	pred_class_probs=np.reshape(probs_out,[1, mc.ANCHORS, mc.CLASSES])

	#calculate confidence_scores
	num_confidence_scores = mc.ANCHOR_PER_GRID+num_class_probs
	conf_in=np.reshape(out[:, :, :, num_class_probs:num_confidence_scores],[1, mc.ANCHORS])
	pred_conf=1/(1+np.exp(-(conf_in)))
	conf=np.reshape(pred_conf, [1, mc.ANCHORS, 1])

	#detection probabilities
	probs=np.multiply(pred_class_probs,conf)
	det_probs=np.max(probs,axis=2)

	#detected class
	det_class = np.argmax(probs, 2)

        # bbox_delta
	pred_box_delta = np.reshape(out[:, :, :, num_confidence_scores:],[1, mc.ANCHORS, 4])
)
	delta_x=pred_box_delta[:,:,0]
	delta_y=pred_box_delta[:,:,1]
	delta_w=pred_box_delta[:,:,2]
	delta_h=pred_box_delta[:,:,3]

	anchor_x = mc.ANCHOR_BOX[:, 0]
	anchor_y = mc.ANCHOR_BOX[:, 1]
	anchor_w = mc.ANCHOR_BOX[:, 2]
	anchor_h = mc.ANCHOR_BOX[:, 3]

	box_center_x = anchor_x + delta_x * anchor_w
	box_center_y = anchor_y + delta_y * anchor_h
	box_width = anchor_w * util.safe_exp(delta_w, mc.EXP_THRESH)
	box_height = anchor_h * util.safe_exp(delta_h, mc.EXP_THRESH)

	xmins, ymins, xmaxs, ymaxs = util.bbox_transform([box_center_x, box_center_y, box_width, box_height])

        # The max x position is mc.IMAGE_WIDTH - 1 since we use zero-based pixels. Same for y.
	xmins = np.minimum(np.maximum(0.0, xmins), mc.IMAGE_WIDTH-1.0)
	ymins = np.minimum(np.maximum(0.0, ymins), mc.IMAGE_HEIGHT-1.0)
	xmaxs = np.maximum(np.minimum(mc.IMAGE_WIDTH-1.0, xmaxs), 0.0)
	ymaxs = np.maximum(np.minimum(mc.IMAGE_HEIGHT-1.0, ymaxs), 0.0)

	det_boxes = np.transpose(np.stack(util.bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),(1, 2, 0))

	return det_boxes, det_probs, det_class

def filter_prediction(mc, boxes, probs, cls_idx):

	if (mc.TOP_N_DETECTION < len(probs)) and (mc.TOP_N_DETECTION > 0) :
		order = probs.argsort()[:-mc.TOP_N_DETECTION-1:-1]
		probs = probs[order]
		boxes = boxes[order]
		cls_idx = cls_idx[order]
	else:
		filtered_idx = np.nonzero(probs>mc.PROB_THRESH)[0]
		probs = probs[filtered_idx]
		boxes = boxes[filtered_idx]
		cls_idx = cls_idx[filtered_idx]

	final_boxes = []
	final_probs = []
	final_cls_idx = []

	for c in range(mc.CLASSES):
		idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
		keep = util.nms(boxes[idx_per_class], probs[idx_per_class], mc.NMS_THRESH)
		for i in range(len(keep)):
			if keep[i]:
				final_boxes.append(boxes[idx_per_class[i]])
				final_probs.append(probs[idx_per_class[i]])
				final_cls_idx.append(c)
	return final_boxes, final_probs, final_cls_idx

def _draw_box(im, box_list, label_list, color=(0,255,0), cdict=None, form='center'):
	assert form == 'center' or form == 'diagonal', \
		'bounding box format not accepted: {}.'.format(form)

	for bbox, label in zip(box_list, label_list):

		if form == 'center':
			bbox = util.bbox_transform(bbox)

			xmin, ymin, xmax, ymax = [int(b) for b in bbox]

			l = label.split(':')[0] # text before "CLASS: (PROB)"
			if cdict and l in cdict:
				c = cdict[l]
			else:
				c = color

		    # draw box
			cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
		    # draw label
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(im, label, (xmin, ymax), font, 0.3, c, 1)


def video_demo(model_filename, weight_filename,input_filename,out_dir):

	mc = kitti_squeezeDet_config()                                        #load the kitti squeezeDet configuration 
	net = caffe.Net(model_filename, weight_filename, caffe.TEST)          #load the caffe model
	cap = cv2.VideoCapture(input_filename)                                #read video input 
	times = {}
	count = 0
	while cap.isOpened():
		t_start = time.time()
		count += 1
		out_im_name = os.path.join(out_dir, str(count).zfill(6)+'.jpg')
       		# Load images from video
		ret, frame = cap.read()
		if ret==True:
			#set the caffe transformer to preprocess images to load into caffe net
			im_input = frame.astype(np.float32) - mc.BGR_MEANS
			transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
			transformer.set_transpose('data', (2,0,1))
		else:
			break

		t_reshape = time.time()
		times['reshape']= t_reshape - t_start

		out = net.forward_all(data=np.asarray([transformer.preprocess('data', im_input)]))     
		det_boxes, det_probs, det_class=interpret_output(out['preds'],mc)                            #interpret the caffe net predictions to get the bounding box proposals and class predictions
		final_boxes, final_probs, final_class = filter_prediction(mc,det_boxes[0], det_probs[0], det_class[0])     # use NMS filtering to get the final bounding boxes and class predictions
		t_detect = time.time()
		times['detect']= t_detect - t_reshape

		keep_idx    = [idx for idx in range(len(final_probs)) \
			if final_probs[idx] > mc.PLOT_PROB_THRESH]
		final_boxes = [final_boxes[idx] for idx in keep_idx]
		final_probs = [final_probs[idx] for idx in keep_idx]
		final_class = [final_class[idx] for idx in keep_idx]
		t_filter = time.time()
		times['filter']= t_filter - t_detect

		cls2clr = {
				'car': (255, 191, 0),
				'cyclist': (0, 191, 255),
				'pedestrian':(255, 0, 191)
				}
		
		#draw bounding boxes on the frame
		out_frame=cv2.resize(frame, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
		_draw_box(
			out_frame, final_boxes,
			[mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
				for idx, prob in zip(final_class, final_probs)],
			cdict=cls2clr,
		)

		t_draw = time.time()
		times['draw']= t_draw - t_filter

		#save and display the frame with detections
		cv2.imwrite(out_im_name, out_frame)
		times['total']= time.time() - t_start
		time_str = 'Total time: {:.4f}, detection time: {:.4f}, filter time: '\
				'{:.4f}'. \
			format(times['total'], times['detect'], times['filter'])
		print (time_str)

		plt.imshow(out_frame)
		plt.draw()
		plt.pause(.1)
		plt.gcf().clear()

	#exit when all frames are processed
	cap.release()
		
	


def main(argv):

	#To run this demo from terminal: 'SqueezeDet_demo.py -m <model_file> -w <output_file> -i <img_file> -o <out_dir>'
	model_filename = ''
	weight_filename = ''
	img_filename = ''
	out_dir=''
	try:
		opts, args = getopt.getopt(argv, "hm:w:i:o:")
		print (opts)
	except getopt.GetoptError:
		print ('SqueezeDet_demo.py -m <model_file> -w <output_file> -i <img_file> -o <out_dir>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print ('SqueezeDet_demo.py -m <model_file> -w <weight_file> -i <img_file> -o <out_dir>')
			sys.exit()
		elif opt == "-m":
			model_filename = arg
		elif opt == "-w":
			weight_filename = arg
		elif opt == "-i":
			img_filename = arg
		elif opt == "-o":
			out_dir = arg			
	print ('model file is "', model_filename)
	print ('weight file is "', weight_filename)
	print ('image file is "', img_filename)
	print('output directory is "', out_dir)

	video_demo(model_filename, weight_filename,img_filename,out_dir)

if __name__=='__main__':	
	main(sys.argv[1:])
