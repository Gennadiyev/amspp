from snntorch.spikevision import spikedata

train_DVS = spikedata.DVSGesture("data/dvsgesture", train=True,
num_steps=500, dt=1000)
test_DVS = spikedata.DVSGesture("data/dvsgesture", train=False,
num_steps=1800, dt=1000)

train_NMNIST = spikedata.NMNIST("data/nmnist", train=True,
num_steps=300, dt=1000)
test_NMNIST = spikedata.NMNIST("data/nmnist", train=False,
num_steps=300, dt=1000)

train_SHD = spikedata.SHD("data/shd", train=True, num_steps=500, dt=2000)
test_SHD = spikedata.SHD("data/shd", train=False, num_steps=500, dt=2000)