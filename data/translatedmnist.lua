local TranslatedMnist, parent = torch.class("dp.TranslatedMnist", "dp.SmallImageSource")

function TranslatedMnist:__init(config)
   config = config or {}   
   config.image_size = config.image_size or {1, 60, 60}
   config.name = config.name or 'translatedmnist'
   config.train_dir = config.train_dir or 'train'
   config.test_dir = 'test'
   config.download_url = config.download_url 
      or 'https://s3.amazonaws.com/torch7/data/translatedmnist.zip'
   parent.__init(self, config)
end
