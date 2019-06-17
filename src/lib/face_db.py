import os
import math
import pickle
import numpy as np

class FaceDb():
	def __init__(self, db_path= None):
		# load/create face database
		self.db_path = db_path if None != db_path else 'facedb.pkl'
		if os.path.exists(db_path):
			print('loading faces from:', db_path)
			with open(db_path, 'rb') as f:
				self.db = pickle.load(f)
				self.db['encodings'] = np.array(self.db['encodings'])
		else:
			# start with an empty db
			self.db = { 'indices_by_name': {}, 'name_by_index': {}, 'encodings': np.array([]) }

	def add_encoding(self, name, enc, flush=True):
		"""
		Add a (name, image) pair or a list of [(name, image)] pairs to the face db.
		:param name - str or a list(str).
		:param enc - a single encoding (128 dimension numpy array or a list of such arrays).
		:param id - id for the face encoding (default None -> returns generated id).
		:param flush - whether or not to flush database to secondary storage (default True). 
		"""
		indices_by_name = self.db['indices_by_name']
		name_by_index = self.db['name_by_index']
		known_face_encodings = self.db['encodings']
		
		# add enc to known_face_encodings and get its row index
		if known_face_encodings.shape[0] == 0:
			known_face_encodings = enc.reshape(1,-1)
		else:
			known_face_encodings = np.vstack((known_face_encodings, enc))

		self.db['encodings'] = known_face_encodings
		index = known_face_encodings.shape[0] - 1

		# update name_by_index dictionary
		name_by_index[index] = name

		# add the newly added index to the indices of name
		indices = indices_by_name.get(name)
		if None == indices:
			indices = []
			indices_by_name[name] = indices
		indices.append(index)

		# flush changes to disk
		if flush:
			self.flush()

	def flush(self):
		with open(self.db_path, 'wb') as f:
			pickle.dump(self.db, f, pickle.HIGHEST_PROTOCOL)

	def face_distance(self, known_face_encodings, unknown_encoding):
		"""
		Given a face encoding, compare it to known face encodings to get an euclidean distance.
		:param known_face_encodings: list of known face encodings
		:param unknown_encoding: a face encoding to match against known_face_encodings
		:return: An array with a distance for each 'known_face_encodings'
		"""
		return np.linalg.norm(known_face_encodings - unknown_encoding, axis=1)

	def match(self, unknown_face_encodings, threshold=None, optimize=False):
		"""
		Match a list of face encodings to known face encodings in the db.
		:param: unknown_face_encodings - a list of face encodings to match.
		:return: (id, face_distance) - use get_name(id) to get the label for the id.
		"""
		matches = []
		known_face_encodings = self.db['encodings']
		for unknown_enc in unknown_face_encodings:
			distances = self.face_distance(known_face_encodings, unknown_enc)
			face_index = np.argmin(distances)
			distance = distances[face_index]
			
			if threshold and distance >= threshold:
				if not optimize:
					continue

				print('*** distance > threshold ({} > {})'.format(distance, threshold))
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
					if abs(d1 - d2) < 0.08:
						matches.append((-1, distance))
						continue
				else: # name1 == name2
					# discard if same name but distance differ (2 after point)
					if int(d1 * 100) != int(d2 * 100):
						matches.append((-1, distance))
						continue
			
			matches.append((face_index, distance))
		return matches

	def get_name(self, id):
		"""return person name by id"""
		return self.db['name_by_index'][id]
