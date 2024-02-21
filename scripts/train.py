from utils import *
from networks import eccv16, unet
import tensorflow as tf
import sys, getopt


def parse_arguments(argv: list) -> tuple[str, str, float, int]:
    params = {
        "batch_size": 4,
        "epochs": 1,
        "model": None,
        "in_meamory": False,
    }
    try:
        opts, args = getopt.getopt(argv,"e:b:m:i",["epochs=","batchsize=", "model=", "inmeamory="])
    except getopt.GetoptError:
        print(getopt.GetoptError)
        print("'processImages.py -e <epochs> -b <batchsize> -m <model> -i <in_meamory>'")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-e", "--epochs"):
            params["epochs"] = int(arg)
        if opt in ("-b", "--batchsize"):
            params["batch_size"] = int(arg)
        if opt in ("-m", "--model"):
            if arg == "eccv16":
                params["model"] = "eccv16"
            elif arg == "unet":
                params["model"] = "unet"
        if opt in ("-i", "--inmeamory"):
            if arg.lower() == "true":
                params["in_meamory"] = True
    return params    

@custom_logger
def train_model(train_dataset, test_dataset, batch_size, epochs, model, base_path = "", model_name="default_name"):
    log("\nBegin training eccv16")
    log(f"EPOCHS={epochs}, BATCH_SIZE={batch_size}")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f"models/{model_name}_cp.ckpt",
                                                    save_weights_only=True,
                                                    verbose=1)
    history = model.fit(x=train_dataset.batch(batch_size),
                        y=None, epochs = epochs,
                        verbose="auto", 
                        validation_data=test_dataset.batch(batch_size),
                        callbacks=[cp_callback])
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.savefig(f"{base_path}images/{model_name}_training_loss.jpg")


def main(args: list):
    settings = parse_arguments(args)
    print("\n\n\n")
    print(settings)

    BASE_PATH = ""
    IN_MEAMORY = settings["in_meamory"]
    EPOCHS = settings["epochs"]
    BATCH_SIZE = settings["batch_size"]

    eval_dataset = get_dataset(f"{BASE_PATH}data/validation/*.jpg", inMeamory=IN_MEAMORY)
    train_dataset = get_dataset(f"{BASE_PATH}data/train/*.jpg", inMeamory=IN_MEAMORY)

    model = None
    if settings["model"] == "eccv16":
        model = eccv16()
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_squared_error')
    
    elif settings["model"] == "unet":
        model = unet()
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_absolute_error')
    

    model.summary()
    train_model(train_dataset, eval_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, model=model, model_name=settings["model"])


if __name__ == "__main__":
    main(sys.argv[1:])    






