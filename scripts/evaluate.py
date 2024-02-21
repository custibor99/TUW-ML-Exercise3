from models import *
from utils import *
from networks import *

BASE_PATH = ""
IN_MEAMORY = False
EPOCHS = 1
BATCH_SIZE = 4
CHECKPOINT_PATH = "models/eccv16_cp.ckpt"

test_dataset = get_dataset(f"{BASE_PATH}data/validation/*.jpg", inMeamory=IN_MEAMORY)
model = eccv16()
model.load_weights(CHECKPOINT_PATH).expect_partial()
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_squared_error', metrics=["mean_absolute_error", "mean_squared_error"])

loss, mae, mse = model.evaluate(test_dataset.batch(32), None, verbose=2)
print(loss, mae, mse)

for l, ab in test_dataset.batch(1):
    pred = model.predict(l)
    pred = pred[0,:,:,:]
    draw_lab(l[0,:,:,:], pred)
    draw_lab_channels(l[0,:,:,:], pred)
    rg = convert_lab_to_rgb(l[0,:,:,:], pred)
    plt.imshow(rg)
    plt.show()
    break