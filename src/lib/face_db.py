import os
import math
import pickle
import numpy as np

FACE_DB_PATH = 'examples/face_recognition/facedb.pkl'

class FaceDb():
	def __init__(self, db_path= None):
		# load/create face database
		self.db_path = db_path if None != db_path else 'facedb.pkl'
		if os.path.exists(db_path):
			self.load()
		else: # start with an empty db
			self.indices_by_name = {}
			self.name_by_index = {}
			self.encodings = np.array([])

	def load(self):
		with open(self.db_path, 'rb') as f:
			self_dict = pickle.load(f)
			self.__dict__.update(self_dict) 


	def flush(self):
		with open(self.db_path, 'wb') as f:
			pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

	def add_encoding(self, name, enc, flush=True):
		"""
		Add a (name, image) pair or a list of [(name, image)] pairs to the face db.
		:param name - str or a list(str).
		:param enc - a single encoding (128 dimension numpy array or a list of such arrays).
		:param id - id for the face encoding (default None -> returns generated id).
		:param flush - whether or not to flush database to secondary storage (default True). 
		"""
		
		# add enc to known_face_encodings and get its row index
		if self.encodings.shape[0] == 0:
			self.encodings = enc.reshape(1,-1)
		else:
			self.encodings = np.vstack((self.encodings, enc))

		index = self.encodings.shape[0] - 1

		# update name_by_index dictionary
		self.name_by_index[index] = name

		# add the newly added index to the indices of name
		indices = self.indices_by_name.get(name)
		if None == indices:
			indices = []
			self.indices_by_name[name] = indices
		indices.append(index)

		# flush changes to disk
		if flush:
			self.flush()

	def get_name(self, id):
		"""return person name by id"""
		return self.name_by_index[id]

	def match(self, enc, threshold=None, optimize=False):
		"""
		Match enc to the known-face-encodings in the db.
		:param: enc - unknown face encoding to match.
		:return: a tuple (id, face_distance) of best match for the unknown encodings. 
		         use get_name(id) to get the label for the id.
		"""

		if len(self.encodings) == 0:
			# no encodings yet
			return -1, 1.0

		# compare enc to known-face-encodings to get all euclidean distances.
		distances = np.linalg.norm(self.encodings - enc, axis=1)

		# get the minimum distance		
		face_index = np.argmin(distances)
		min_distance = distances[face_index]

		# optimization if min_distance >= threshold
		if threshold and min_distance >= threshold:
			if not optimize:
				return -1, min_distance

			print('*** distance > threshold ({} > {})'.format(min_distance, threshold))
			top_two = np.argsort(distances)[:2]
			idx1 = top_two[0]
			name1 = self.get_name(idx1)
			print('\ttop 1: {} - {:.5f}'.format(name1, distances[idx1]))
			idx2 = top_two[1]
			name2 = self.get_name(idx2)
			print('\ttop 2: {} - {:.5f}'.format(name2, distances[idx2]))
			
			d1 = distances[idx1]
			d2 = distances[idx2]

			# discard if names differ
			if name1 != name2:
				if abs(d1 - d2) < 0.06:
					return -1, min_distance
			else: # name1 == name2
				# discard if same name but distance differ (2 after point)
				if int(d1 * 100) != int(d2 * 100):
					return -1, min_distance
			
		return face_index, min_distance
