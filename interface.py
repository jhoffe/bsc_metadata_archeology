import metadata_archealogy as ma
from lightning import Trainer
from torch.utils.data import DataLoader

ps = ma.wrap(dataset)

callback = ma.Callback()
trainer = Trainer(..., callbacks=[callback])

trainer.fit(model, DataLoader(ps, ...))

callback.analyze()