from timeflux import timeflux as tf
import multiprocessing as mp
import os

if __name__ == "__main__":
    mp.freeze_support()
    tf.main(app="graphs/main.yaml", env_file=os.path.join("environments","environment.env"))