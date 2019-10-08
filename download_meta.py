import requests


class ISICApi(object):
    def __init__(self, hostname='https://isic-archive.com',
                 username=None, password=None):
        self.baseUrl = f'{hostname}/api/v1'
        self.authToken = None

        if username is not None:
            if password is None:
                password = input(f'Password for user "{username}":')
            self.authToken = self._login(username, password)

    def _makeUrl(self, endpoint):
        return f'{self.baseUrl}/{endpoint}'

    def _login(self, username, password):
        authResponse = requests.get(
            self._makeUrl('user/authentication'),
            auth=(username, password)
        )
        if not authResponse.ok:
            raise Exception(f'Login error: {authResponse.json()["message"]}')

        authToken = authResponse.json()['authToken']['token']
        return authToken

    def get(self, endpoint):
        url = self._makeUrl(endpoint)
        headers = {'Girder-Token': self.authToken} if self.authToken else None
        return requests.get(url, headers=headers)

    def getJson(self, endpoint):
        return self.get(endpoint).json()

    def getJsonList(self, endpoint):
        endpoint += '&' if '?' in endpoint else '?'
        LIMIT = 50
        offset = 0
        while True:
            resp = self.get(
                f'{endpoint}limit={LIMIT:d}&offset={offset:d}'
            ).json()
            if not resp:
                break
            for elem in resp:
                yield elem
            offset += LIMIT

api = ISICApi(username="dmitry@gurov.co", password="Lbvfvelfr1")

imageList = api.getJson('image?limit=24000&offset=0&sort=name')

meta_file = open('img_metas.csv', 'a')
header_str = 'id,name,type,w,h,dataset,diagnosis,diagnosis_confirm_type,diagnosis_another,age,sex,\n'
imageCount = len(imageList)
meta_file.write(header_str) 
for image in imageList:
	current_ind = imageList.index(image)
	print(current_ind, '/', imageCount, ' - ', round(((current_ind+1) / imageCount)*100, 3), '%' )
	imageDetail = api.getJson('image/%s' % image['_id'])
	if '_id' in imageDetail:
		i_id = str(imageDetail['_id'])
	else:
		i_id = 'null'
	if 'name' in imageDetail:
		i_name = str(imageDetail['name'])
	else:
		i_name = 'null'
	if 'dataset' in imageDetail:
		dataset_dict = imageDetail['dataset']
		if 'name' in dataset_dict:
			i_ds = str(dataset_dict['name'])
		else:
			i_ds = 'null'
	else:
		i_ds = 'null'
	if 'meta' in imageDetail:
		meta_dict = imageDetail['meta']
		if 'acquisition' in meta_dict:
			acquisition_dict = meta_dict['acquisition']
			if 'image_type' in acquisition_dict: 
				i_type = str(acquisition_dict['image_type'])
			else:
				i_type = 'null'
			if 'pixelsX' in acquisition_dict:
				i_w = str(acquisition_dict['pixelsX'])
			else:
				i_w = 'null'
			if 'pixelsY' in acquisition_dict:
				i_h = str(acquisition_dict['pixelsY'])
			else:
				i_h = 'null'
		else:
			i_type = 'null'
			i_w = 'null'
			i_h = 'null'
		if 'clinical' in meta_dict:
			clinical_dict = meta_dict['clinical']

			if 'diagnosis' in clinical_dict:
				i_diagnos = str(clinical_dict['diagnosis'])
			else:
				i_diagnos = 'null'
			if 'diagnosis_confirm_type' in clinical_dict:
				i_conf_type = str(clinical_dict['diagnosis_confirm_type'])
			else:
				i_conf_type = 'null'
			if 'age_approx' in clinical_dict:
				i_age = str(clinical_dict['age_approx'])
			else:
				i_age = 'null'
			if 'sex' in clinical_dict:
				i_sex = str(clinical_dict['sex'])
			else:
				i_sex = 'null'
		else:
			i_diagnos = 'null'
			i_conf_type = 'null'
			i_age = 'null'
			i_sex = 'null'
		if 'unstructured' in meta_dict:
			unstructured_dict = meta_dict['unstructured']
			if 'diagnosis' in unstructured_dict:
				i_another_diag = str(unstructured_dict['diagnosis'])
			else:
				i_another_diag = 'null'
		else:
			i_another_diag = 'null'
	else:
		i_type = 'null'
		i_w = 'null'
		i_h = 'null'
		i_diagnos = 'null'
		i_conf_type = 'null'
		i_age = 'null'
		i_sex = 'null'
		i_another_diag = 'null'
	
	image_str = i_id + ',' + i_name + ',' + i_type + ',' + i_w + ',' + i_h + ',' + i_ds + ',' + i_diagnos + ',' + i_conf_type + ',' + i_another_diag + ',' + i_age + ',' + i_sex + '\n'
	meta_file.write(image_str)
meta_file.close()

print('Done')