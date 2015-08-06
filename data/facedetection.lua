local FaceDetection, parent = torch.class("dp.FaceDetection", "dp.SmallImageSource")

function FaceDetection:__init(config)
   config = config or {}   
   config.image_size = config.image_size or {3, 32, 32}
   config.name = config.name or 'facedetection'
   config.train_dir = config.train_dir or 'face-dataset'
   config.test_dir = ''
   config.download_url = config.download_url 
      or 'https://engineering.purdue.edu/elab/files/face-dataset.zip'
   parent.__init(self, config)
end
